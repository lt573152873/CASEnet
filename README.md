CASENet: Deep Category-Aware Semantic Edge Detection
====================================================

Legal Remarks
-------------

Copyright 2017 Mitsubishi Electric Research Laboratories All
Rights Reserved.

Permission to use, copy and modify this software and its
documentation without fee for educational, research and non-profit
purposes, is hereby granted, provided that the above copyright
notice, this paragraph, and the following three paragraphs appear
in all copies.

To request permission to incorporate this software into commercial
products contact: Director; Mitsubishi Electric Research
Laboratories (MERL); 201 Broadway; Cambridge, MA 02139.

IN NO EVENT SHALL MERL BE LIABLE TO ANY PARTY FOR DIRECT,
INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
DOCUMENTATION, EVEN IF MERL HAS BEEN ADVISED OF THE POSSIBILITY OF
SUCH DAMAGES.

MERL SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN
"AS IS" BASIS, AND MERL HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE,
SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

Overview
--------

This source code package contains the C++/Python implementation of our CASENet based on [Caffe](http://github.com/BVLC/caffe) for multi-label semantic edge detection training/testing. There are two folders in this package:

* caffe

    Our modified Caffe (based on the official Caffe [commit](https://github.com/BVLC/caffe/commit/4efdf7ee49cffefdd7ea099c00dc5ea327640f04) on June 20, 2017), with the C++ *MultiChannelReweightedSigmoidCrossEntropyLossLayer* implementing the multi-label loss function as explained in equation (1) of the CASENet paper.   
    
    We also modified the C++ *ImageSegDataLayer* from [DeepLab-v2](https://bitbucket.org/aquariusjay/deeplab-public-ver2) for reading multi-label edge ground truth stored as binary file.
    All modifications can be checked by "git log".

* CASENet

    Python scripts and network configurations for training/testing/visaulization.   
    Note that initial and trained model weights (caffemodel files) for SBD/Cityscapes dataset can be downloaded separately from [MERL's FTP](ftp://ftp.merl.com/pub/cfeng/CASENet/))
    

**If you use this package, please cite our CVPR 2017 paper:**

```
@inproceedings{yu2017casenet, 
  title={uppercase{CASEN}et: Deep Category-Aware Semantic Edge Detection}, 
  author={Z. Yu and C. Feng and M. Y. Liu and S. Ramalingam}, 
  booktitle={IEEE Conf. on Computer Vision and Pattern Recognition}, 
  year={2017}
}
```

Version
-------

1.0

Installation
------------

#### Compile

1. Unzip the CASENet_Codes.zip file to ${PACKAGE_ROOT} so the folder structure looks like this:

```
${PACKAGE_ROOT}
├── caffe
│   ├── build
│   ├── cmake
│   ├── data
│   ├── docker
│   ├── docs
│   ├── examples
│   ├── include
│   ├── matlab
│   ├── models
│   ├── python
│   ├── scripts
│   ├── src
│   └── tools
└── CASENet
    ├── cityscapes
    │   ├── config
    │   └── model
    └── sbd
        ├── config
        └── model
```

2. Follow the official Caffe's [installation guie](http://caffe.berkeleyvision.org/install_apt.html) to compile the modified Caffe in ${PACKAGE_ROOT}/caffe (building with cuDNN is supported).

3. Make sure to build pycaffe.

#### Using Trained Weights

We have supplied trained weights for easier testing. To use them, simply download them separately from [MERL's FTP](ftp://ftp.merl.com/pub/cfeng/CASENet/)) to ${PACKAGE_ROOT}/CASENet/sbd/model. Similarly for Cityscapes.

Experiments
-----------

Assume pycaffe is installed in ${PACKAGE_ROOT}/caffe/build/install/python. Following instructions use CASENet on Cityscapes dataset as an example. Baselines (Basic/DSN/CASENet-) run similarly. For SBD dataset, change all "cityscapes" to "sbd" in the following instructions.

1. If you want to train the network for other datasets, in the ${PACKAGE_ROOT}/CASENet/cityscapes/config/train_CASENet.prototxt, modify the *root_folder* and *source* at lines 27-28 to point to your dataset.

2. Run the following commands to perform training and testing:
```
cd ${PACKAGE_ROOT}/CASENet/cityscapes
# Training
python solve.py ./config/solver_CASENet.prototxt -c ../../caffe/build/install/python

# Testing
python test.py ./config/test_CASENet.prototxt ./model/CASENet_iter_40000.caffemodel -c ../../caffe/build/install/python -l ${Cityscapes_DATASET}/val.txt -d ${Cityscapes_DATASET} -o ./result_CASENet

# Visualization (note visualization for SBD is slightly different)
python visualize_multilabel.py ${Cityscapes_DATASET}/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png
```

3. Check ${PACKAGE_ROOT}/CASENet/cityscapes/model folder for trained weights and check ${PACKAGE_ROOT}/CASENet/cityscapes/result_CASENet for testing results. Check ${PACKAGE_ROOT}/CASENet/cityscapes/visualize for visualization results.

#### Training Notes

If you want to train CASENet on your own dataset, you will need to generate multi-label ground truth that is readable by our modified ImageSegDataLayer, which is essentially a memory buffer dumped in binary format that stores multi-label ground truth image in row-major order, where each pixel of this multi-label image has 4 x num_label_chn **bytes**, i.e., 32 x num_label_chn **bits** (num_label_chn as specified in image_data_param in the training prototxt file).

For example, a toy multi-label ground truth image with num_label_chn=1 corresponding to a 2x3 input RGB image can be the following bits in memory:
```
1000000000000000000000000000000000 0000000000000000000000000000000001 0000000000000000000000000000000010
0000000000000000000000000000000101 0000000000000000000000000000001110 0000000000000000000000000000000000
```
which means the following pixel labels:
```
ignored,           edge-type-0,             edge-type-1
edge-type-0-and-2, edge-type-1-and-2-and-3, non-edge
```
Basically, we use a single bit to encode a single label of a pixel, and the highest bit (the 32-th bit) is used to label ignored pixels excluded from loss computation. More details can be found in line 265-273 of the image_seg_data_layer.cpp file.

BTW, our *MultiChannelReweightedSigmoidCrossEntropyLossLayer* currently only supports ignoring the 32-th bit (which is enough for the Cityscapes dataset): so if your max number of labels is more than 31 (e.g., the [ADE20K dataset](http://groups.csail.mit.edu/vision/datasets/ADE20K/)), your num_label_chn has to be larger than 1, and you will need to modify *MultiChannelReweightedSigmoidCrossEntropyLossLayer* correspondingly. More details can be found in line 54 and line 130 of the multichannel_reweighted_sigmoid_cross_entropy_loss_layer.cpp file.

#### Generating Ground Truth Multi-label Edge Map
To generate such ground truth multi-label edge map from ground truth semantic segmentation in Cityscapes and SBD, we provide a separate code package downloadable in our [GitHub account](https://github.com/Chrisding).

Contact
-------

[Zhiding Yu](yzhiding@andrew.cmu.edu)   
[Chen Feng](simbaforrest@gmail.com)

**Feel free to email any bugs or suggestions to help us improve the code. Thank you!**