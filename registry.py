from datafree.models import classifiers, deeplab
from torchvision import datasets, transforms as T
from datafree.utils import sync_transforms as sT

from PIL import PngImagePlugin, Image
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


import os
import torch
import torchvision
import datafree
import lmdb
import six
import pickle
import json
import torch.nn as nn 
from PIL import Image
from collections import OrderedDict
import torch.utils.data as data

import gzip
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose

class MNISTDataset(Dataset):
    def __init__(self, data_folder, train=True, transform=None):
        self.transform = transform
        self.train = train

        if self.train:
            with gzip.open(os.path.join(data_folder, 'train-images-idx3-ubyte.gz'), 'r') as f_in:
                self.images = np.frombuffer(f_in.read(), np.uint8, offset=16).reshape(-1, 28, 28)
            with gzip.open(os.path.join(data_folder, 'train-labels-idx1-ubyte.gz'), 'r') as f_in:
                self.labels = np.frombuffer(f_in.read(), np.uint8, offset=8)
        else:
            with gzip.open(os.path.join(data_folder, 't10k-images-idx3-ubyte.gz'), 'r') as f_in:
                self.images = np.frombuffer(f_in.read(), np.uint8, offset=16).reshape(-1, 28, 28)
            with gzip.open(os.path.join(data_folder, 't10k-labels-idx1-ubyte.gz'), 'r') as f_in:
                self.labels = np.frombuffer(f_in.read(), np.uint8, offset=8)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image_pil = Image.fromarray(image, 'L')
            image = self.transform(image_pil)

        return image, label

def loads_data(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pickle.loads(buf)

class LMDBDataset(data.Dataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super().__init__()
        db_path = os.path.join(root, f"{split}.lmdb")
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_data(txn.get(b'__len__'))
            self.keys = loads_data(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_data(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf)

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # return img, target
        return img, target

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

class COOPLMDBDataset(LMDBDataset):
    def __init__(self, root, split="train", transform=None) -> None:
        super().__init__(root, split, transform=transform)
        with open(os.path.join(root, "split.json")) as f:
            split_file = json.load(f)
        idx_to_class = OrderedDict(sorted({s[-2]: s[-1] for s in split_file["test"]}.items()))
        self.classes = list(idx_to_class.values())

NORMALIZE_DICT = {
    'mnist':    dict( mean=(0.1307,),                std=(0.3081,) ),
    'fmnist':    dict( mean=(0.1307,),                std=(0.3081,) ),
    'cifar10':  dict( mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010) ),
    'cifar100': dict( mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761) ),
    'imagenet': dict( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    'tinyimagenet': dict( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    
    'cub200':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'stanford_dogs':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'stanfordcars':   dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365_32x32': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365_64x64': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'places365': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'svhn': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'tiny_imagenet': dict( mean=(0.4802, 0.4481, 0.3975), std=(0.2770, 0.2691, 0.2821) ),
    'imagenet_32x32': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'gtsrb': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'eurosat': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'oxfordpets': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    
    # for semantic segmentation
    'camvid': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
    'nyuv2': dict( mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5) ),
}


MODEL_DICT = {
    # https://github.com/polo5/ZeroShotKnowledgeTransfer
    'wrn16_1': classifiers.wresnet.wrn_16_1,
    'wrn16_2': classifiers.wresnet.wrn_16_2,
    'wrn40_1': classifiers.wresnet.wrn_40_1,
    'wrn40_2': classifiers.wresnet.wrn_40_2,

    # https://github.com/HobbitLong/RepDistiller
    'resnet8': classifiers.resnet_tiny.resnet8,
    'resnet20': classifiers.resnet_tiny.resnet20,
    'resnet32': classifiers.resnet_tiny.resnet32,
    'resnet56': classifiers.resnet_tiny.resnet56,
    'resnet110': classifiers.resnet_tiny.resnet110,
    'resnet8x4': classifiers.resnet_tiny.resnet8x4,
    'resnet32x4': classifiers.resnet_tiny.resnet32x4,
    'vgg8': classifiers.vgg.vgg8_bn,
    'vgg11': classifiers.vgg.vgg11_bn,
    'vgg13': classifiers.vgg.vgg13_bn,
    'shufflenetv2': classifiers.shufflenetv2.shuffle_v2,
    'mobilenetv2': classifiers.mobilenetv2.mobilenet_v2,
    
    # https://github.com/huawei-noah/Data-Efficient-Model-Compression/tree/master/DAFL
    'resnet50':  classifiers.resnet.resnet50,
    'resnet18':  classifiers.resnet.resnet18,
    'resnet34':  classifiers.resnet.resnet34,
}

IMAGENET_MODEL_DICT = {
    'resnet50_imagenet': classifiers.resnet_in.resnet50,
    'resnet18_imagenet': classifiers.resnet_in.resnet18,
    'mobilenetv2_imagenet': torchvision.models.mobilenet_v2,
}

SEGMENTATION_MODEL_DICT = {
    'deeplabv3_resnet50':  deeplab.deeplabv3_resnet50,
    'deeplabv3_mobilenet': deeplab.deeplabv3_mobilenet,
}


def get_model(name: str, num_classes, pretrained=False, **kwargs):
    if 'imagenet' in name:
        model = IMAGENET_MODEL_DICT[name](pretrained=pretrained)
        if num_classes!=1000:
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'deeplab' in name:
        model = SEGMENTATION_MODEL_DICT[name](num_classes=num_classes, pretrained_backbone=kwargs.get('pretrained_backbone', False))
    else:
        model = MODEL_DICT[name](num_classes=num_classes)
    return model 


def get_dataset(name: str, data_root: str='data', return_transform=False, split=['A', 'B', 'C', 'D']):
    name = name.lower()
    data_root = os.path.expanduser( data_root )

    if name=='mnist':
        num_classes = 10
        train_transform = Compose([
        # T.Resize((28, 28)), 
        T.ToTensor(), 
        T.Resize((32, 32)),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        # T.Normalize(**NORMALIZE_DICT['mnist'])
        ])

        val_transform = Compose([
            # T.Resize((28, 28)), 
            T.ToTensor(), 
            T.Resize((32, 32)),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            # T.Normalize(**NORMALIZE_DICT['mnist'])
        ])
        data_root = os.path.join( data_root, 'torchdata/MNIST/raw' ) 
        train_dst = MNISTDataset(data_root, train=True, transform=train_transform)
        val_dst = MNISTDataset(data_root, train=False, transform=val_transform)
    
    elif name == 'fmnist':
        num_classes = 10
        train_transform = Compose([
        T.ToTensor(), 
        T.Resize((32, 32)),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])

        val_transform = Compose([
            # T.Resize((28, 28)), 
            T.ToTensor(), 
            T.Resize((32, 32)),
            T.Lambda(lambda x: x.repeat(3, 1, 1)),
            # T.Normalize(**NORMALIZE_DICT['fmnist'])
        ])
        data_root = os.path.join( data_root, 'torchdata/FashionMNIST/raw' ) 
        train_dst = MNISTDataset(data_root, train=True, transform=train_transform)
        val_dst = MNISTDataset(data_root, train=False, transform=val_transform)

    elif name=='cifar10':
        num_classes = 10
        train_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR10(data_root, train=False, download=True, transform=val_transform)
    elif name=='gtsrb':
        num_classes = 43
        preprocess = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.GTSRB(data_root, split="train", download = True, transform = preprocess)
        val_dst = datasets.GTSRB(data_root, split="test", download = True, transform = preprocess)
    
    elif name == 'eurosat':
        num_classes = 10
        preprocess = T.Compose([
            T.Lambda(lambda x: x.convert("RGB")),
            T.Resize((128, 128)),
            T.ToTensor(),
        ])
        data_root = os.path.join( data_root, 'torchdata/{}'.format(name) ) 
        train_dst = COOPLMDBDataset(data_root, split="train", transform = preprocess)
        val_dst = COOPLMDBDataset(data_root, split="test", transform = preprocess)
    
    elif name == 'flowers102':
        num_classes = 102
        preprocess = T.Compose([
            T.Lambda(lambda x: x.convert("RGB")),
            T.Resize((128, 128)),
            T.ToTensor(),
        ])
        data_root = os.path.join( data_root, 'torchdata/{}'.format(name) ) 
        train_dst = COOPLMDBDataset(data_root, split="train", transform = preprocess)
        val_dst = COOPLMDBDataset(data_root, split="test", transform = preprocess)

    elif name == 'oxfordpets':
        num_classes = 37
        preprocess = T.Compose([
            T.Lambda(lambda x: x.convert("RGB")),
            T.Resize((128, 128)),
            T.ToTensor(),
        ])
        data_root = os.path.join( data_root, 'torchdata/{}'.format(name) ) 
        train_dst = COOPLMDBDataset(data_root, split="train", transform = preprocess)
        val_dst = COOPLMDBDataset(data_root, split="test", transform = preprocess)

    elif name=='c10+p365':
        num_classes = 10
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            #T.Resize((224, 224), Image.BICUBIC),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst_1 = datasets.CIFAR10(data_root, train=True, download=True, transform=train_transform)
        val_dst_1 = datasets.CIFAR10(data_root, train=False, download=True, transform=val_transform)
        
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join( data_root, 'Places365_32x32' ) 
        train_dst_2 = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_transform)
        val_dst_2 = datasets.ImageFolder(os.path.join(data_root, 'val'), transform=val_transform)
        train_dst = torch.utils.data.ConcatDataset([train_dst_1, train_dst_2])
        val_dst = torch.utils.data.ConcatDataset([val_dst_1, val_dst_2])
    elif name=='cifar100':
        num_classes = 100
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.CIFAR100(data_root, train=True, download=True, transform=train_transform)
        val_dst = datasets.CIFAR100(data_root, train=False, download=True, transform=val_transform)
    elif name=='svhn':
        num_classes = 10
        train_transform = T.Compose([
            T.ToTensor(),
            # T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            # T.Normalize( **NORMALIZE_DICT[name] ),
        ])
        data_root = os.path.join( data_root, 'torchdata' ) 
        train_dst = datasets.SVHN(data_root, split='train', download=True, transform=train_transform)
        val_dst = datasets.SVHN(data_root, split='test', download=True, transform=val_transform)
    elif name=='imagenet' or name=='imagenet-0.5':
        num_classes=1000
        train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join( data_root, 'ILSVRC2012' ) 
        train_dst = datasets.ImageNet(data_root, split='train', transform=train_transform)
        val_dst = datasets.ImageNet(data_root, split='val', transform=val_transform)
    elif name=='imagenet_32x32':
        num_classes=1000
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join( data_root, 'ImageNet_32x32' ) 
        train_dst = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_transform)
        val_dst = datasets.ImageFolder(os.path.join(data_root, 'val'), transform=val_transform)
    elif name=='places365_32x32':
        num_classes=365
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join( data_root, 'Places365_32x32' ) 
        train_dst = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_transform)
        val_dst = datasets.ImageFolder(os.path.join(data_root, 'val'), transform=val_transform)
    elif name=='places365_64x64':
        num_classes=365
        train_transform = T.Compose([
            T.RandomCrop(64, padding=8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join( data_root, 'Places365_64x64' ) 
        train_dst = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_transform)
        val_dst = None #datasets.ImageFolder(os.path.join(data_root, 'val'), transform=val_transform)
    elif name=='places365':
        num_classes=365
        train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_DICT[name]),
        ])
        data_root = os.path.join( data_root, 'Places365' ) 
        train_dst = datasets.ImageFolder(os.path.join(data_root, 'train'), transform=train_transform)
        val_dst = datasets.ImageFolder(os.path.join(data_root, 'val'), transform=val_transform)
    elif name=='cub200':
        num_classes=200
        train_transform = T.Compose([
            T.Resize(64),
            T.RandomCrop(64, padding=8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        val_transform = T.Compose([
            T.Resize(64),
            T.CenterCrop(64),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )       
        data_root = os.path.join(data_root, 'CUB200')
        train_dst = datafree.datasets.CUB200(data_root, split='train', transform=train_transform)
        val_dst = datafree.datasets.CUB200(data_root, split='val', transform=val_transform)
    elif name=='stanford_dogs':
        num_classes=120
        train_transform = T.Compose([
            T.Resize(64),
            T.RandomCrop(64, padding=8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        val_transform = T.Compose([
            T.Resize(64),
            T.CenterCrop(64),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )       
        data_root = os.path.join(data_root, 'StanfordDogs')
        train_dst = datafree.datasets.StanfordDogs(data_root, split='train', transform=train_transform)
        val_dst = datafree.datasets.StanfordDogs(data_root, split='test', transform=val_transform)
    elif name=='stanfordcars':
        num_classes=196
        train_transform = T.Compose([
            T.Resize(64),
            T.RandomCrop(64, padding=8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )
        val_transform = T.Compose([
            T.Resize(64),
            T.CenterCrop(64),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )       
        data_root = os.path.join(data_root, 'StanfordCars')
        train_dst = datafree.datasets.StanfordCars(data_root, split='train', transform=train_transform)
        val_dst = datafree.datasets.StanfordCars(data_root, split='test', transform=val_transform)
    elif name=='tiny_imagenet':
        num_classes=200
        train_transform = T.Compose([T.Resize((32, 32)),
	    T.RandomHorizontalFlip(), 
	    T.RandomCrop(32, padding=4),
        T.ToTensor(), 
        T.Normalize(**NORMALIZE_DICT[name])] 
        )
        val_transform = T.Compose([
            T.Resize((32, 32)),
            T.ToTensor(),
            T.Normalize( **NORMALIZE_DICT[name] )]
        )       
        data_root = os.path.join(data_root, 'torchdata/tiny-imagenet-200')
        train_dst = datasets.ImageFolder(data_root, transform=train_transform)
        val_dst = datasets.ImageFolder(data_root, transform=val_transform)

    # For semantic segmentation
    elif name=='nyuv2':
        num_classes=13
        train_transform = sT.Compose([
            sT.Multi( sT.Resize(256), sT.Resize(256, interpolation=Image.NEAREST)),
            #sT.Multi( sT.ColorJitter(0.2, 0.2, 0.2), None),
            sT.Sync(  sT.RandomCrop(128),  sT.RandomCrop(128)),
            sT.Sync(  sT.RandomHorizontalFlip(), sT.RandomHorizontalFlip() ),
            sT.Multi( sT.ToTensor(), sT.ToTensor( normalize=False, dtype=torch.uint8) ),
            sT.Multi( sT.Normalize( **NORMALIZE_DICT[name] ), None)
        ])
        val_transform = sT.Compose([
            sT.Multi( sT.Resize(256), sT.Resize(256, interpolation=Image.NEAREST)),
            sT.Multi( sT.ToTensor(),  sT.ToTensor( normalize=False, dtype=torch.uint8 ) ),
            sT.Multi( sT.Normalize( **NORMALIZE_DICT[name] ), None)
        ])
        data_root = os.path.join( data_root, 'NYUv2' )
        train_dst = datafree.datasets.NYUv2(data_root, split='train', transforms=train_transform)
        val_dst = datafree.datasets.NYUv2(data_root, split='test', transforms=val_transform)
    else:
        raise NotImplementedError
    if return_transform:
        return num_classes, train_dst, val_dst, train_transform, val_transform
    return num_classes, train_dst, val_dst
