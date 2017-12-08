# Image-Retrieval-by-Finetuning-CNN

Pytorch Code for Image Retrieval

## Prerequisites

- Python 3.6
- CUDA 8.0

## Getting started
### Installation
- Install Pytorch from http://pytorch.org/
- Install Torchvision from the source
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
Beacause pytorch and torchvision are ongoing projects.

Here I noted that our code is tested based on Pytorch 0.2.0 and Torchvision 0.2.0.

### 1.Train
```bash
python train.py --gpu_ids 0 
```

### 2.Demo
```bash
python demo.py
```

### 3.Test
```bash
python test.py
```

