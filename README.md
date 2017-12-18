# Counting Everyday Objects in Everyday Scenes

## Introduction

**Counting Everyday Objects in Everyday Scenes**  
Prithvijit Chattopadhyay*, Ramakrishna Vedantam*, Ramprasaath Ramaswamy Selvaraju, Dhruv Batra, Devi Parikh  
[CVPR 2017][3]

This repository contains code for training one-shot and contextual deep counting models mentioned in the CVPR 2017 version of the paper.

## Abstract

![](https://github.com/prithv1/cvpr2017_counting/img/seq_new.png?raw=true)

We are interested in counting the number of instances of object classes in natural, everyday images. Previous counting approaches tackle the problem in restricted domains such as counting pedestrians in surveillance videos. Counts can also be estimated from outputs of other vision tasks like object detection. In this work, we build dedicated models for counting designed to tackle the large variance in counts, appearances, and scales of objects found in natural scenes. Our approach is inspired by the phenomenon of subitizing – the ability of humans to make quick assessments of counts given a perceptual signal, for small count values. Given a natural scene, we employ a divide and conquer strategy while incorporating context across the scene to adapt the subitizing idea to counting. Our approach offers consistent improvements over numerous baseline approaches for counting on the PASCAL VOC 2007 and COCO datasets. Subsequently, we study how counting can be used to improve object detection. We then show a proof of concept application of our counting methods to the task of Visual Question Answering, by studying the ‘how many?’ questions in the VQA and COCO-QA datasets.


## Instructions

### Install Torch

```shell
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; bash install-deps;
./install.sh
source ~/.bashrc
```

### Required Packages
* [torch-hdf5][4]
* [Element-Research/rnn][5]
* [Element-Research/dpnn][6]


## Cite this work

If you find this code useful, consider citing our work:

```
@InProceedings{Chattopadhyay_2017_CVPR,
author = {Chattopadhyay, Prithvijit and Vedantam, Ramakrishna and Selvaraju, Ramprasaath R. and Batra, Dhruv and Parikh, Devi},
title = {Counting Everyday Objects in Everyday Scenes},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {July},
year = {2017}
}
```

## Authors

* [Prithvijit Chattopadhyay][2] (prithvijit3@gatech.edu)
* [Ramakrishna Vedantam][1] (vrama@gatech.edu)

## License

BSD

[1]: http://vrama91.github.io/
[2]: http://prithv1.github.io
[3]: http://openaccess.thecvf.com/content_cvpr_2017/papers/Chattopadhyay_Counting_Everyday_Objects_CVPR_2017_paper.pdf
[4]: https://github.com/deepmind/torch-hdf5
[5]: https://github.com/Element-Research/rnn
[6]: https://github.com/Element-Research/dpnn