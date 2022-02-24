# Learning Network Architecture for Open-set Recognition

This repository contains the code for our AAAI 2020 paper **Learning Network Architecture for Open-set Recognition**

![alt text](https://github.com/zxl101/NAS_OSR/blob/master/images/pipeline.png "Pipeline")

# Requirements
**Environment**
1. Python 3.6.8
2. CUDA 11.0
3. PyTorch
4. TorchVision
5. R \

**Installation**
+ Clone this repository \
  git clone `https://github.com/zxl101/NAS_OSR.git` \
  cd NAS_OSR
+ Install Dependencies
  pip install -r requirements.txt \
  conda install r-base=3.6 \
  pip install rpy2 \
  R \
  chooseCRANmirror(graphics=F) \
  install.packages("mvtnorm")
  \### Then change the `os.environ['R_HOME'] = 'xxxx/lib/R'` in qmv.py in both search and train folders to your environment path
# Datasets
The following datasets are used to train/evaluate our network
+ MNIST
+ SVHN
+ CIFAR10
+ CIFAR100 \

By default they are downloaed to data folders under search and train folders respectively. If you wish to change the location of the datasets, please modify dataloader.py in search and train folders.

# Training
## Architecture Search
1. Search \
  cd search \
  sh search.sh \
2. Train from scratch on target dataset \
  cd NAS_OSR/train \ 
  + Copy the folder which contains the searched architecture into NAS_OSR/train/
  + Set the name of your searched folder *C.load_path = "used_searched_models/4loss_os_f1/search/"* in config_train.py on line 66 \
  sh train.sh \### Please choose the dataset you want to train on and comment all other lines \
# Acknowledgements
This repository makes use of code from [FasterSeg](https://github.com/VITA-Group/FasterSeg) and [GCM-CF](https://github.com/yue-zhongqi/gcm-cf/tree/main/osr).
