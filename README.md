<div align="center">
<h1>Encapsulating Knowledge in One Prompt</h1>

<div>
Qi Li&emsp;Runpeng Yu&emsp;Xinchao Wang<sup>&dagger;</sup>
</div>
<div>
    LV-Lab, National University of Singapore&emsp;
    <sup>&dagger;</sup>corresponding author 
</div>

---
![overall_structure](./datafree/ECCV2024_KiOP_pipeline.jpg)


## Installation & Preparation

<div style="text-align: left;">
1. Clone the repo and prepare the virtual environment.
</div>

```
git clone https://github.com/LiQiiiii/Encapsulating-Knowledge-In-One-Prompt.git
```

```
cd Encapsulating-Knowledge-In-One-Prompt
```

```
pip install -r requirements.txt
```

<div style="text-align: left;">
2. Prepare the dataset and models. You can use your own models and dataset. For quick stark, we provide several models and datasets, which can be download directly from google drive:
</div>

```
gdown https://drive.google.com/file/d/18XDK2fdhCQuwGm4sJntfSvESpbZEv1bY/view?usp=drive_link
```

```
gdown https://drive.google.com/file/d/19o2EItRw-LOJUdjDf-mOz0zh0QalF8wj/view?usp=drive_link
```

```
unzip KiOP_models.zip
```

```
unzip KiOP_data.zip
```

## Training & Evaluation

We provide several scripts in ```./scripts```. For example, for running KiOP-B, you may use the ```KiOP_B.sh``` as follows:

```
sh ./scripts/KiOP_B.sh
```

You can adjust the hyperparameters to customize your setup.
