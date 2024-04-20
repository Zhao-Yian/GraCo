<div style="text-align: center; margin: 10px">
    <h1> ⭐ GraCo: Granularity-Controllable Interactive Segmentation </h1>
</div>
<p align="center">
    <a href="https://zhao-yian.github.io/GraCo"><img alt="Static Badge" src="https://img.shields.io/badge/Project_page-openproject.svg?logo=openproject&color=%230770B8"></a>
    <a href="">
    <img alt="Static Badge" src="https://img.shields.io/badge/Paper-arXiv.svg?logo=arxiv&labelColor=%23B31B1B&color=%23B31B1B">
    </a>
    <a href="https://youtu.be/QE8Mi0k2nKg?si=yJXbYAzTG1qHF_uK">
    <img src="https://img.shields.io/badge/Video-FC2947.svg?logo=YouTube" style="display:inline;"></a>
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/Zhao-Yian/GraCo">
    <a href="">
    <img alt="Static Badge" src="https://img.shields.io/badge/Demo-buffer.svg?logo=buffer">
    </a>
    <a href="mailto: zhaoyian.zh@gmail.com">
    <img alt="Static Badge" src="https://img.shields.io/badge/contact_me-email-yellow">
    </a>
</p>

![GraCo_overview](./assets/overview.jpg)

This is the official implementation for our <span style='color: #EB5353;font-weight:bold'>CVPR'24 highlight</span> paper "GraCo: Granularity-Controllable Interactive Segmentation".

## 💡 Introduction

Current IS pipelines fall into two categories: single-granularity output and multi-granularity output. The latter aims to alleviate the spatial ambiguity present in the former.
However, the multi-granularity output pipeline suffers from limited interaction flexibility and produces redundant results.
We introduce Granularity-Controllable Interactive Segmentation (GraCo), 
a novel approach that allows precise control of prediction granularity by introducing additional parameters 
to input. This enhances the customization of the interactive system and eliminates redundancy while 
resolving ambiguity. 
Nevertheless, the exorbitant cost of annotating multi-granularity masks and the lack of available datasets with granularity annotations make it difficult for models to acquire the necessary guidance to control output granularity.
To address this problem, we design an any-granularity mask generator that exploits the semantic property of the pre-trained IS model to automatically generate abundant mask-granularity pairs without requiring additional manual annotation. 
Based on these pairs, we propose a granularity-controllable learning strategy that efficiently imparts the granularity controllability to the IS model.

<div align="center">
  <img src="./assets/motivation.jpg" width=500 >
</div>

## TODO

- Release code 
- Upload pre-trained weight

## 🚀 Quick start

### 📍 Install

```bash
pip install -r requirements.txt
```

### 🏕️ Any-Granularity mask Generator

```bash
python any_granularity_generator.py --checkpoint weights/simpleclick/sbd_vit_base.pth  \
    --save-path part_output --save-name proposal.pkl
```

### 🦄 Train and Evaluation

- Download pre-trained weights

[SimpleClick models](https://drive.google.com/drive/folders/1qpK0gtAPkVMF7VC42UA9XF4xMWr5KJmL?usp=sharing)

- Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py models/plainvit_base448_graco.py --load_gra \
    --part_path part_output/proposals.pkl --enable_lora \
    --weights weights/simpleclick/sbd_vit_base.pth \
    --gpus 0,1,2,3
```

- Evaluation on object-level benchmarks
```bash
python evaluate.py NoBRS --datasets GrabCut,Berkeley,DAVIS,SBD \
    --checkpoint weights/simpleclick/sbd_vit_base.pth \
    --lora_checkpoint path/to/checkpoints/last_checkpoint.pth --gra-oracle
```

- Evaluation on PartImageNet, SA-1B
```bash
python evaluate.py NoBRS --datasets PartImageNet,SA-1B \
    --checkpoint weights/simpleclick/sbd_vit_base.pth \
    --lora_checkpoint path/to/checkpoints/last_checkpoint.pth --gra-oracle
```

- Evaluation on PascalPart (five categories)
```bash
for c in "sheep" "cat" "dog" "cow" "aeroplane" "bus"; 
do 
  python evaluate.py NoBRS --datasets PascalPart \
    --checkpoint weights/simpleclick/sbd_vit_base.pth \
    --lora_checkpoint path/to/checkpoints/last_checkpoint.pth --gra-oracle --class-name $c; 
done
```

### 🌋 Evaluation of SAM

- Download SAM

[ViT-B SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)

[ViT-L SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth)

```bash
python evaluate.py NoBRS --datasets GrabCut,Berkeley,DAVIS,SBD,PartImageNet,SA-1B \
    --checkpoint weights/sam/sam_vit_b_01ec64.pth \
    --sam-model vit_b --sam-type SAM --oracle
```

```bash
for c in "sheep" "cat" "dog" "cow" "aeroplane" "bus"; 
do 
  python evaluate.py NoBRS --datasets PascalPart \
    --checkpoint weights/sam/sam_vit_b_01ec64.pth \
    --sam-model vit_b --sam-type SAM --oracle --class-name $c; 
done
```

## Acknowledgements
This repository is built upon [SimpleClick](https://github.com/uncbiag/SimpleClick). The project page is built using the template of [Nerfies](https://nerfies.github.io/). 
Thank the authors of these open source repositories for their efforts. And thank the ACs and reviewers for their effort when dealing with our paper.