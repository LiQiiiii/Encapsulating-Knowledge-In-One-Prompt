import argparse
from math import gamma
import os
import random
import shutil
import time
import warnings

import registry
import datafree

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
warnings.filterwarnings('ignore')
from visual_prompt import ExpansiveVisualPrompt, AdditiveVisualPrompt, ExpansiveVisualPrompt_one_channel
import numpy as np
from functools import partial
parser = argparse.ArgumentParser(description='KiOP')

def label_mapping_base(logits, mapping_sequence):
    modified_logits = logits[:, mapping_sequence]
    return modified_logits


class model_Fusion(nn.Module):
    def __init__(self, modelA, modelB, modelC, modelD):
        super(model_Fusion, self).__init__()
        self.modelA = modelA
        self.modelD = modelD
        self.modelB = modelB
        self.modelC = modelC
    
        for param in self.modelB.parameters():
            param.requires_grad = False

    def forward(self, x):
        x1 = self.modelA(x) 
        x4 = self.modelD(x1)
        x2 = self.modelB(x4) 
        x = self.modelC(x2)
        return x

    def train(self, mode=True):
        self.training = mode
        self.modelA.train(mode)
        self.modelD.train(mode)
        return self

    def eval(self):
        self.training = False
        self.modelA.eval()
        self.modelD.eval()
        return self 

class model_Fusion_1(nn.Module):
    def __init__(self, modelA, modelB):
        super(model_Fusion_1, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
    
        for param in self.modelB.parameters():
            param.requires_grad = False

    def forward(self, x):
        x1 = self.modelA(x) 
        x = self.modelB(x1) 
        return x

    def train(self, mode=True):
        self.training = mode
        self.modelA.train(mode)
        return self

    def eval(self):
        self.training = False
        self.modelA.eval()
        return self
    
def save_data(sampled_data, labels, path):
    """
    Save the sampled data and labels to a specified path.
    """
    torch.save({
        'sampled_data': sampled_data,
        'labels': labels
    }, path)

def test_accuracy(data, labels, model):
    model.eval() 
    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for i in range(len(data)):
            outputs = model(data[i].unsqueeze(0)) 
            _, predicted = torch.max(outputs.data, 1) 
            _, true_labels = torch.max(labels[i], 0)
            
            total += labels[i].size(0) 
            correct += (predicted == true_labels).sum().item() 
        
    accuracy = 100 * correct / total  
    print('Accuracy of the network on the test data: %d %%' % accuracy)
    return accuracy

# Data Free
parser.add_argument('--method', required=True, choices=['zskt', 'dfad', 'dafl', 'deepinv', 'dfq', 'cmi'])
parser.add_argument('--cn', default=3, type=int)
parser.add_argument('--adv', default=0, type=float, help='scaling factor for adversarial distillation')
parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
parser.add_argument('--oh', default=0, type=float, help='scaling factor for one hot loss (cross entropy)')
parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
parser.add_argument('--balance', default=0, type=float, help='scaling factor for class balance')
parser.add_argument('--save_dir', default='run/synthesis', type=str)
parser.add_argument('--save_dir_1', default='run/synthesis', type=str)

parser.add_argument('--cr', default=1, type=float, help='scaling factor for contrastive model inversion')
parser.add_argument('--cr_T', default=0.5, type=float, help='temperature for contrastive model inversion')
parser.add_argument('--cmi_init', default=None, type=str, help='path to pre-inverted data')

# Basic
parser.add_argument('--data_root', default='data')
parser.add_argument('--teacher', default='wrn40_2')
parser.add_argument('--backbone_t', default='ResNet18')
parser.add_argument('--backbone_s', default='ResNet50')
parser.add_argument('--student', default='wrn16_1')
parser.add_argument('--dataset', default='cifar10') # dataset to distill and train whole student model
parser.add_argument('--real_dataset', default='cifar100') # dataset for core net of student model
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate for KD')
parser.add_argument('--lr_decay_milestones', default="120,150,180", type=str,
                    help='milestones for learning rate decay')

parser.add_argument('--lr_g', default=1e-3, type=float, 
                    help='initial learning rate for generation')
parser.add_argument('--T', default=1, type=float)

parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--g_steps', default=1, type=int, metavar='N',
                    help='number of iterations for generation')
parser.add_argument('--kd_steps', default=400, type=int, metavar='N',
                    help='number of iterations for KD after generation')
parser.add_argument('--ep_steps', default=400, type=int, metavar='N',
                    help='number of total iterations in each epoch')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate_only', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--synthesis_batch_size', default=None, type=int,
                    metavar='N',
                    help='mini-batch size (default: None) for synthesis, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# Device
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
# TODO: Distributed and FP-16 training 
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--fp16', action='store_true',
                    help='use fp16')

# Misc
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--log_tag', default='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--img_size', default=32, type=int,
                    help='seed for initializing training.')
parser.add_argument('--vp1_size', default=36, type=int,
                    help='seed for initializing training.')
parser.add_argument('--vp2_size', default=224, type=int,
                    help='seed for initializing training.')
best_acc1 = 0


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.ngpus_per_node = ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    ############################################
    # GPU and FP16
    ############################################
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    if args.fp16:
        from torch.cuda.amp import autocast, GradScaler
        args.scaler = GradScaler() if args.fp16 else None 
        args.autocast = autocast
    else:
        args.autocast = datafree.utils.dummy_ctx


    ############################################
    # Logger
    ############################################
    if args.log_tag != '':
        args.log_tag = '-'+args.log_tag
    log_name = 'R%d-%s-%s-%s%s'%(args.rank, args.dataset, args.teacher, args.student, args.log_tag) if args.multiprocessing_distributed else '%s-%s-%s'%(args.dataset, args.teacher, args.student)
    args.logger = datafree.utils.logger.get_logger(log_name, output='checkpoints/datafree-%s/log-%s-%s-%s%s.txt'%(args.method, args.dataset, args.teacher, args.student, args.log_tag))
    if args.rank<=0:
        for k, v in datafree.utils.flatten_dict( vars(args) ).items(): # print args
            args.logger.info( "%s: %s"%(k,v) )

    ############################################
    # Setup dataset
    ############################################
    num_classes, ori_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
    num_classes_real, ori_dataset_real, val_dataset_real = registry.get_dataset(name=args.real_dataset, data_root=args.data_root)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    val_loader_real = torch.utils.data.DataLoader(
        val_dataset_real,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    ori_loader_real = torch.utils.data.DataLoader(
        ori_dataset_real,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    ori_data_loader = iter(ori_loader_real)
    
    
    evaluator = datafree.evaluators.classification_evaluator(val_loader)
    evaluator_prompt_v2 = datafree.evaluators.classification_prompt_evaluator_v2(val_loader_real)

    ############################################
    # Setup models
    ############################################
    def prepare_model(model):
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
            return model
        elif args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                return model
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
                return model
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            return model
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()
            return model
    args.normalizer = normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])

    if args.dataset == 'mnist' or args.dataset == 'fmnist':
        from torchvision.models import resnet18, resnet50, resnet101, ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, vgg13_bn
        if args.backbone_t == 'ResNet18':
            teacher = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            teacher.fc = nn.Linear(teacher.fc.in_features, 10)
            pretrained_tea = torch.load("{}_32_{}".format(args.backbone_t, args.dataset))
            teacher.load_state_dict(pretrained_tea)
        elif args.backbone_t == 'ResNet50':
            teacher = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            teacher.fc = nn.Linear(teacher.fc.in_features, 10)
            pretrained_tea = torch.load("{}_32_{}".format(args.backbone_t, args.dataset))
            teacher.load_state_dict(pretrained_tea)
        elif args.backbone_t == 'ResNet101':
            teacher = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            teacher.fc = nn.Linear(teacher.fc.in_features, 10)
            pretrained_tea = torch.load("{}_32_{}".format(args.backbone_t, args.dataset))
            teacher.load_state_dict(pretrained_tea)
        elif args.backbone_t == 'vgg13':
            teacher = vgg13_bn(pretrained=True)
            num_ftrs = teacher.classifier[6].in_features
            teacher.classifier[6] = nn.Linear(num_ftrs, num_classes)
            pretrained_tea = torch.load("{}_bn_{}".format(args.backbone_t, args.dataset))
            teacher.load_state_dict(pretrained_tea)
        
    else:
        from torchvision.models import resnet18, resnet50, resnet101, ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, vgg13_bn
        if args.backbone_t == 'ResNet18':
            teacher = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            num_ftrs = teacher.fc.in_features
            teacher.fc = nn.Linear(num_ftrs, num_classes)
            pretrained_tea = torch.load("{}_{}".format(args.backbone_t, args.dataset))
            teacher.load_state_dict(pretrained_tea)
        elif args.backbone_t == 'ResNet50':
            teacher = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            num_ftrs = teacher.fc.in_features
            teacher.fc = nn.Linear(num_ftrs, num_classes)
            pretrained_tea = torch.load("{}_{}".format(args.backbone_t, args.dataset))
            teacher.load_state_dict(pretrained_tea)
        elif args.backbone_t == 'ResNet101':
            teacher = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            num_ftrs = teacher.fc.in_features
            teacher.fc = nn.Linear(num_ftrs, num_classes)
            pretrained_tea = torch.load("{}_{}".format(args.backbone_t, args.dataset))
            teacher.load_state_dict(pretrained_tea)
        elif args.backbone_t == 'vgg13':
            teacher = vgg13_bn(pretrained=True)
            num_ftrs = teacher.classifier[6].in_features
            teacher.classifier[6] = nn.Linear(num_ftrs, num_classes)
            pretrained_tea = torch.load("{}_bn_{}".format(args.backbone_t, args.dataset))
            teacher.load_state_dict(pretrained_tea)

    if args.real_dataset == 'mnist' or args.real_dataset == 'fmnist':
        from torchvision.models import resnet18, resnet50, resnet101, ResNet18_Weights, ResNet50_Weights, ResNet101_Weights, vgg13_bn
        if args.backbone_s == 'ResNet18':
            student_core = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            student_core.fc = nn.Linear(student_core.fc.in_features, 10)
            pretrained_stu = torch.load("{}_32_{}".format(args.backbone_s, args.real_dataset))
            student_core.load_state_dict(pretrained_stu)

            student_init = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            student_init.fc = nn.Linear(student_init.fc.in_features, 10)
            pretrained_stu = torch.load("{}_32_{}".format(args.backbone_s, args.real_dataset))
            student_init.load_state_dict(pretrained_stu)

        elif args.backbone_s == 'ResNet50':
            student_core = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            student_core.fc = nn.Linear(student_core.fc.in_features, 10)
            pretrained_stu = torch.load("{}_32_{}".format(args.backbone_s, args.real_dataset))
            student_core.load_state_dict(pretrained_stu)

            student_init = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            student_init.fc = nn.Linear(student_init.fc.in_features, 10)
            pretrained_stu = torch.load("{}_32_{}".format(args.backbone_s, args.real_dataset))
            student_init.load_state_dict(pretrained_stu)

        elif args.backbone_s == 'ResNet101':
            student_core = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            student_core.fc = nn.Linear(student_core.fc.in_features, 10)
            pretrained_stu = torch.load("{}_32_{}".format(args.backbone_s, args.real_dataset))
            student_core.load_state_dict(pretrained_stu)

            student_init = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            student_init.fc = nn.Linear(student_init.fc.in_features, 10)
            pretrained_stu = torch.load("{}_32_{}".format(args.backbone_s, args.real_dataset))
            student_init.load_state_dict(pretrained_stu)
        
        elif args.backbone_s == 'vgg13':
            student_core = vgg13_bn(pretrained=True)
            num_ftrs = student_core.classifier[6].in_features
            student_core.classifier[6] = nn.Linear(num_ftrs, 10)
            pretrained_stu = torch.load("{}_bn_{}".format(args.backbone_s, args.real_dataset))
            student_core.load_state_dict(pretrained_stu)

            student_init = vgg13_bn(pretrained=True)
            num_ftrs = student_init.classifier[6].in_features
            student_init.classifier[6] = nn.Linear(num_ftrs, 10)
            pretrained_stu = torch.load("{}_bn_{}".format(args.backbone_s, args.real_dataset))
            student_init.load_state_dict(pretrained_stu)
        
        visual_prompt = ExpansiveVisualPrompt(args.vp1_size, mask=np.zeros((args.img_size, args.img_size)))
        visual_prompt_core = ExpansiveVisualPrompt(args.vp2_size, mask=np.zeros((args.vp1_size, args.vp1_size)))
        mapping_sequence = torch.randperm(num_classes_real)[:num_classes]
        label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
        student = model_Fusion(visual_prompt, student_core, label_mapping, visual_prompt_core)

        stu_prompt_core = model_Fusion_1(visual_prompt, student_core) ###

    else:
        from torchvision.models import resnet18, resnet50, resnet101, ResNet18_Weights, ResNet50_Weights, ResNet101_Weights
        if args.backbone_s == 'ResNet18':
            student_core = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            num_ftrs = student_core.fc.in_features
            student_core.fc = nn.Linear(num_ftrs, num_classes_real)
            pretrained_stu = torch.load("{}_{}".format(args.backbone_s, args.real_dataset))
            student_core.load_state_dict(pretrained_stu)

            student_init = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            student_init.fc = nn.Linear(student_init.fc.in_features, num_classes_real)
            pretrained_stu = torch.load("{}_{}".format(args.backbone_s, args.real_dataset))
            student_init.load_state_dict(pretrained_stu)


        elif args.backbone_s == 'ResNet50':
            student_core = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            num_ftrs = student_core.fc.in_features
            student_core.fc = nn.Linear(num_ftrs, num_classes_real)
            pretrained_stu = torch.load("{}_{}".format(args.backbone_s, args.real_dataset))
            student_core.load_state_dict(pretrained_stu)

            student_init = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            student_init.fc = nn.Linear(student_init.fc.in_features, num_classes_real)
            pretrained_stu = torch.load("{}_{}".format(args.backbone_s, args.real_dataset))
            student_init.load_state_dict(pretrained_stu)


        elif args.backbone_s == 'ResNet101':
            student_core = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            num_ftrs = student_core.fc.in_features
            student_core.fc = nn.Linear(num_ftrs, num_classes_real)
            pretrained_stu = torch.load("{}_{}".format(args.backbone_s, args.real_dataset))
            student_core.load_state_dict(pretrained_stu)

            student_init = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
            student_init.fc = nn.Linear(student_init.fc.in_features, num_classes_real)
            pretrained_stu = torch.load("{}_{}".format(args.backbone_s, args.real_dataset))
            student_init.load_state_dict(pretrained_stu)

        elif args.backbone_s == 'vgg13':
            student_core = vgg13_bn(pretrained=True)
            num_ftrs = student_core.classifier[6].in_features
            student_core.classifier[6] = nn.Linear(num_ftrs, num_classes_real)
            pretrained_stu = torch.load("{}_bn_{}".format(args.backbone_s, args.real_dataset))
            student_core.load_state_dict(pretrained_stu)

            student_init = vgg13_bn(pretrained=True)
            num_ftrs = student_init.classifier[6].in_features
            student_init.classifier[6] = nn.Linear(num_ftrs, num_classes_real)
            pretrained_stu = torch.load("{}_bn_{}".format(args.backbone_s, args.real_dataset))
            student_init.load_state_dict(pretrained_stu)

        visual_prompt = ExpansiveVisualPrompt(args.vp1_size, mask=np.zeros((args.img_size, args.img_size)))
        visual_prompt_core = ExpansiveVisualPrompt(args.vp2_size, mask=np.zeros((args.vp1_size, args.vp1_size)))
        mapping_sequence = torch.randperm(num_classes_real)[:num_classes]
        label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
        student = model_Fusion(visual_prompt, student_core, label_mapping, visual_prompt_core)

        stu_prompt_core = model_Fusion_1(visual_prompt, student_core) ###

    student = prepare_model(student)
    teacher = prepare_model(teacher)
    student_init = prepare_model(student_init)

    stu_prompt_core = prepare_model(stu_prompt_core) ###

    criterion = datafree.criterions.KLDiv(T=args.T)
    
    ############################################
    # Setup data-free synthesizers
    ############################################
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size
    
    if args.method=='deepinv':
        synthesizer = datafree.synthesis.DeepInvSyntheiszer(
                 teacher=teacher, student=student, num_classes=num_classes, 
                 img_size=(3, 32, 32), iterations=args.g_steps, lr_g=args.lr_g,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, tv=0.001, l2=0.0,
                 save_dir=args.save_dir, transform=ori_dataset.transform,
                 normalizer=args.normalizer, device=args.gpu)
    elif args.method in ['zskt', 'dfad', 'dfq', 'dafl']:
        nz = 512 if args.method=='dafl' else 256
        generator = datafree.models.generator.LargeGenerator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        criterion = torch.nn.L1Loss() if args.method=='dfad' else datafree.criterions.KLDiv()
        synthesizer = datafree.synthesis.GenerativeSynthesizer(
                 teacher=teacher, student=student, generator=generator, nz=nz, 
                 img_size=(3, 32, 32), iterations=args.g_steps, lr_g=args.lr_g,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, act=args.act, balance=args.balance, criterion=criterion,
                 normalizer=args.normalizer, device=args.gpu)
    elif args.method == 'cmi' and args.cn==3:
        # cifar10: nz=256, ngf=64
        nz = 256
        # nz = 512 # big
        generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        feature_layers = None # use all conv layers
        if args.teacher=='resnet34': # only use blocks
            feature_layers = [teacher.layer1, teacher.layer2, teacher.layer3, teacher.layer4]
        if args.dataset == 'mnist' or args.dataset == 'fmnist':
            img_size = 32
        else:
            img_size = args.img_size
        synthesizer = datafree.synthesis.CMISynthesizer(teacher, student, generator, 
                 nz=nz, num_classes=num_classes, img_size=(3, img_size, img_size), 
                 # if feature layers==None, all convolutional layers will be used by CMI.
                 feature_layers=feature_layers, cn=args.cn, bank_size=40960, n_neg=4096, head_dim=256, init_dataset=args.cmi_init,
                 iterations=args.g_steps, lr_g=args.lr_g, progressive_scale=False,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, cr=args.cr, cr_T=args.cr_T,
                 save_dir=args.save_dir, transform=ori_dataset.transform,
                 normalizer=args.normalizer, device=args.gpu)
        
        synthesizer_1 = datafree.synthesis.CMISynthesizer(student_init, stu_prompt_core, generator, 
                 nz=nz, num_classes=num_classes_real, img_size=(3, img_size, img_size), ###
                 feature_layers=feature_layers, cn=args.cn, bank_size=40960, n_neg=4096, head_dim=256, init_dataset=args.cmi_init,
                 iterations=args.g_steps, lr_g=args.lr_g, progressive_scale=False,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, cr=args.cr, cr_T=args.cr_T,
                 save_dir=args.save_dir_1, transform=ori_dataset_real.transform,
                 normalizer=args.normalizer, device=args.gpu)

    elif args.method == 'cmi' and args.cn==1:
        # cifar10: nz=256, ngf=64
        nz = 128
        print("get 1")
        # nz = 512 # big
        generator = datafree.models.generator.Generator(nz=nz, ngf=56, img_size=28, nc=1)
        # generator = datafree.models.generator.LargeGenerator(nz=nz, ngf=64, img_size=32, nc=3)
        # generator = datafree.models.generator.Generator(nz=nz, ngf=256, img_size=128, nc=3) # big
        generator = prepare_model(generator)
        feature_layers = None # use all conv layers
        if args.teacher=='resnet34': # only use blocks
            feature_layers = [teacher.layer1, teacher.layer2, teacher.layer3, teacher.layer4]
        synthesizer = datafree.synthesis.CMISynthesizer(teacher, student, generator, 
                 nz=nz, num_classes=num_classes, img_size=(1, 28, 28), 
                 # if feature layers==None, all convolutional layers will be used by CMI.
                 feature_layers=feature_layers, cn=args.cn, bank_size=40960, n_neg=4096, head_dim=256, init_dataset=args.cmi_init,
                 iterations=args.g_steps, lr_g=args.lr_g, progressive_scale=False,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, cr=args.cr, cr_T=args.cr_T,
                 save_dir=args.save_dir, transform=ori_dataset.transform,
                 normalizer=args.normalizer, device=args.gpu)
    else: raise NotImplementedError
        
    ############################################
    # Setup optimizer
    ############################################
    optimizer = torch.optim.SGD(student.parameters(), args.lr, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer, T_max=args.epochs)

    ############################################
    # Resume
    ############################################
    args.current_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            if isinstance(student, nn.Module):
                student.load_state_dict(checkpoint['state_dict'])
            else:
                student.module.load_state_dict(checkpoint['state_dict'])
            best_acc1 = checkpoint['best_acc1']
            try: 
                args.start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            except: print("Fails to load additional model information")
            print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                  .format(args.resume, checkpoint['epoch'], best_acc1))
        else:
            print("[!] no checkpoint found at '{}'".format(args.resume))
        
    ############################################
    # Evaluate
    ############################################
    if args.evaluate_only:
        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        print('[Eval] Acc={acc:.4f}'.format(acc=eval_results['Acc']))
        return

    ############################################
    # Train Loop
    ############################################
    # all_images = []
    # all_labels = []
    for epoch in range(args.start_epoch, args.epochs):
        #if args.distributed:
        #    train_sampler.set_epoch(epoch)
        args.current_epoch=epoch

        for _ in range( args.ep_steps//args.kd_steps ): # total kd_steps < ep_steps
            # 1. Data synthesis
            vis_results, images_to_save, labels_to_save = synthesizer.synthesize() # g_steps
            vis_results_1, images_to_save_1, labels_to_save_1 = synthesizer_1.synthesize() # g_steps
            # 2. Knowledge distillation
            train( synthesizer, synthesizer_1, [student, teacher, student_init], args.dataset, args.real_dataset, ori_data_loader, ori_loader_real, criterion, optimizer, args) # # kd_steps
        
        for vis_name, vis_image in vis_results.items():
            datafree.utils.save_image_batch( vis_image, 'checkpoints/datafree-%s/%s%s.png'%(args.method, vis_name, args.log_tag) )
        
        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        eval_results_tea = evaluator_prompt_v2(student.modelA, student.modelB, device=args.gpu)
        (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        (acc1_tea, acc5_tea), val_loss_tea = eval_results_tea['Acc'], eval_results_tea['Loss']
        args.logger.info('[Eval] Epoch={current_epoch} Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f} Acc@1_tea={acc1_tea:.4f} Acc@5_tea={acc5_tea:.4f} Loss_tea={loss_tea:.4f} Lr={lr:.4f}'
                .format(current_epoch=args.current_epoch, acc1=acc1, acc5=acc5, loss=val_loss, acc1_tea = acc1_tea, acc5_tea = acc5_tea, loss_tea = val_loss_tea, lr=optimizer.param_groups[0]['lr']))
        scheduler.step()
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        _best_ckpt = 'checkpoints/datafree-%s/%s-%s-%s.pth'%(args.method, args.dataset, args.teacher, args.student)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.student,
                'state_dict': student.state_dict(),
                'best_acc1': float(best_acc1),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, _best_ckpt)


    if args.rank<=0:
        args.logger.info("Best: %.4f"%best_acc1)


def train(synthesizer, synthesizer_1, model, dts, r_dts, ori_data_loader, ori_loader_real, criterion, optimizer, args):
    loss_metric = datafree.metrics.RunningLoss(datafree.criterions.KLDiv(reduction='sum'))
    loss_metric_1 = datafree.metrics.RunningLoss(datafree.criterions.KLDiv(reduction='sum'))
    acc_metric = datafree.metrics.TopkAccuracy(topk=(1,5))
    acc_metric_1 = datafree.metrics.TopkAccuracy(topk=(1,5))
    student, teacher, student_init = model
    optimizer = optimizer
    student.train()
    teacher.eval()
    student_init.eval()
    for i in range(args.kd_steps):
        images = synthesizer.sample()
        images_stu = synthesizer_1.sample()
        if dts == 'mnist' or dts == 'fmnist':
            images = images.reshape(*images.shape[:2], -1)
            images_parts = images.chunk(3, dim=1) 
            images_reduced = torch.stack([part.mean(dim=1, keepdim=True) for part in images_parts], dim=1)
            images_reduced = images_reduced.reshape(*images_reduced.shape[:2], int(images_reduced.shape[-1]**0.5), int(images_reduced.shape[-1]**0.5)) 
            images = images_reduced.squeeze() 
        if r_dts == 'mnist' or r_dts == 'fmnist':
            images_stu = images_stu.reshape(*images_stu.shape[:2], -1) 
            images_parts = images_stu.chunk(3, dim=1)
            images_reduced = torch.stack([part.mean(dim=1, keepdim=True) for part in images_parts], dim=1)
            images_reduced = images_reduced.reshape(*images_reduced.shape[:2], int(images_reduced.shape[-1]**0.5), int(images_reduced.shape[-1]**0.5)) 
            images_stu = images_reduced.squeeze()  

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            images_stu = images_stu.cuda(args.gpu, non_blocking=True)
        with args.autocast():
            with torch.no_grad():
                t_out = teacher(images)
            s_out = student(images.detach())

            ss_prompt_out = student.modelB(student.modelA(images_stu.detach()))
            ss_out = student.modelB(images_stu.detach())

            loss_st_prompt = criterion(s_out, t_out.detach())
            loss_ss_prompt = criterion(ss_out, ss_prompt_out)
            loss_s = loss_st_prompt + loss_ss_prompt 
        optimizer.zero_grad()
        if args.fp16:
            scaler_s = args.scaler_s
            scaler_s.scale(loss_s).backward()
            scaler_s.step(optimizer)
            scaler_s.update()
        else:
            loss_s.backward()
            optimizer.step()

        acc_metric.update(s_out, t_out.max(1)[1])
        acc_metric_1.update(ss_prompt_out, ss_out.max(1)[1])

        loss_metric.update(s_out, t_out)
        loss_metric_1.update(ss_prompt_out, ss_out)
        if args.print_freq>0 and i % args.print_freq == 0:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            (train_acc1_1, train_acc5_1), train_loss_1 = acc_metric_1.get_results(), loss_metric_1.get_results()
            args.logger.info('[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, train_acc@1_1={train_acc1_1:.4f}, train_acc@5_1={train_acc5_1:.4f}, train_Loss_1={train_Loss_1:.4f}, Lr={lr:.4f}'
              .format(current_epoch=args.current_epoch, i=i, total_iters=len(args.kd_steps), train_acc1=train_acc1, train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset()
    
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)

if __name__ == '__main__':
    main()
