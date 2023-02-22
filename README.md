This repository provides code for the paper, Learning Unforgotten Domain-Invariant Representations for Online Unsupervised Domain Adaptation.

## Environment
Python 3.6.9, Pytorch 1.2.0, Torch Vision 0.4, [Apex](https://github.com/NVIDIA/apex). See requirement.txt.
 We used the nvidia apex library for memory efficient high-speed training.

## Data Preparation

[Office Dataset] (https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
[OfficeHome Dataset](http://hemanthdv.org/OfficeHome-Dataset/) 
[Image-clef Dataset]（https://drive.google.com/file/d/0B9kJH0-rJ2uRS3JILThaQXJhQlk/view）

Prepare dataset in data directory as follows.
```
./dataset/office/ ## Office
./dataset/office-home/ ## OfficeHome
./dataset/image-clef/## Image-clef
```
## Training
All the parameters are set to optimal in our experiments. The following are the command for each task. The test_interval can be changed, which is the number of iterations between near test.
```
Office-31

pythonn Online_UDA_SFD.py --gpu_id id --net ResNet50 --dset office --test_interval 500 --s_dset_path ./data/office/amazon_list.txt --t_dset_path ./data/office/webcam_list.txt
```
```
Office-Home

pythonn Online_UDA_SFD.py --gpu_id id --net ResNet50 --dset office-home --test_interval 2000 --s_dset_path ./data/office-home/Art.txt --t_dset_path ./data/office-home/Clipart.txt
```
Image-clef

pythonn Online_UDA_SFD.py --gpu_id id --net ResNet50 --dset image-clef --test_interval 500 --s_dset_path ./data/image-clef/b_list.txt --t_dset_path ./data/image-clef/i_list.txt
```
