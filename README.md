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


### 0.You may try our trained model first
```bash
python demo.py --gpu_ids 0 --name ft_ResNet50 --test_dir ./test/gallery  --query_path ./demo.jpg
```
`gpu_ids` which gpu to run.

`name` the name of model.

`query_path` the path of the query image.

`test_dir` the dir including all candidate images. For example,

```
        gallery/
             dog/xxx.jpg
             dog/xxy.png
             cat/123.png
             cat/asd932_.png
             ...
             dog/xxz.jpg
             cat/nsdf3.jpeg
```

Run `demo.py` under the terminal or graphical user interface (i.e. spyder)
In the terminal, it will output the paths of the top-10 related images.
In addition to top-10 image paths, the images will be showed if you run the code under the graphical user interface.

### 1.Collect Training data from Google Image
Google Image contains lots of data. 
Despite of noisy images, most images are somehow related to the keyword.
So our method use web images collected from Google Image.

### 2.Train
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --data_dir ./noisy
```
`gpu_ids` which gpu to run.

`name` the name of output model.

`data_dir` the training dataset dir. 

For example, `data_dir` should include a training dir `train` and a validation dir `val`.
Every class is store in a separate folder under the train/val dir.
```
        noisy/
             train/dog/xxx.jpg
             train/dog/xxy.png
             train/cat/123.png
             train/cat/asd932_.png
             ...
             val/dog/xxz.jpg
             val/cat/nsdf3.jpeg
```


### 4.Test
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir ./test
```
To get a comrephensive evaluation, we select 30 images as query images. 

`gpu_ids` which gpu to run.

`name` the name of output model.

`test_dir` the dir containing query and gallery.

```
    test/
       query/
             dog/1.jpg
             ...
             cat/2.png   
        gallery/
             dog/xxx.jpg
             dog/xxy.png
             cat/123.png
             cat/asd932_.png
             ...
             dog/xxz.jpg
             cat/nsdf3.jpeg     
```

