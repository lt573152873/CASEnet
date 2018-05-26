#Copyright (c) 2017 Mitsubishi Electric Research Laboratories (MERL).   All rights reserved.
#
#The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications.  MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose.  In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.
#
#As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes. 

# test CASENet:
# python test.py ./config/test_CASENet.prototxt ./model/CASENet_iter_22000.caffemodel -l ~/Dataset/SBD_Aug/val.txt -d ~/Dataset/SBD_Aug -o ./result_CASENet -c ../../caffe/build/install/python

import os
import sys
import argparse

import numpy as np
import cv2

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('deploy_prototxt_file', type=str,
                    help="path to the deploy prototxt file")
parser.add_argument('model', type=str,
                    help="path to the caffemodel containing the trained weights")
parser.add_argument('-l', '--image_list', type=str, default='',
                    help="list of image files to be tested")
parser.add_argument('-f', '--image_file', type=str, default='',
                    help="a single image file to be tested")
parser.add_argument('-d', '--image_dir', type=str, default='',
                    help="root folder of the image files in the list or the single image file")
parser.add_argument('-o', '--output_dir', type=str, default='.',
                    help="folder to store the test results")
parser.add_argument('-c', '--pycaffe_folder', type=str, default='../../code/python',
                    help="pycaffe folder that contains the caffe/_caffe.so file")
parser.add_argument('-g', '--gpu', type=int, default=0,
                    help="use which gpu device (default=0)")
args = parser.parse_args(sys.argv[1:])

assert(os.path.exists(args.deploy_prototxt_file))
assert(os.path.exists(args.model))

if os.path.exists(os.path.join(args.pycaffe_folder,'caffe/_caffe.so')):
    sys.path.insert(0, args.pycaffe_folder)
import caffe

caffe.set_mode_gpu()
caffe.set_device(args.gpu)

# load input path
if os.path.exists(args.image_list):
    with open(args.image_list) as f:
        test_lst = [x.strip().split()[0] for x in f.readlines()]
        if args.image_dir!='':
            test_lst = [
                args.image_dir+x if os.path.isabs(x)
                else os.path.join(args.image_dir, x)
                for x in test_lst]
else:
    image_file = os.path.join(args.image_dir, args.image_file)
    if os.path.exists(image_file):
        test_lst = [os.path.join(args.image_dir, os.path.basename(image_file))]
    else:
        raise IOError('nothing to be tested!')

# load net
net = caffe.Net(args.deploy_prototxt_file, args.model, caffe.TEST)
num_cls = 20
crop_size = 512
mean_value = (104.008, 116.669, 122.675) #BGR

for idx_img in xrange(len(test_lst)):
    in_ = cv2.imread(test_lst[idx_img]).astype(np.float32)
    width, height = in_.shape[1], in_.shape[0]
    if(crop_size < width or crop_size < height):
        raise ValueError('Input image size must be smaller than crop size!')
    pad_x = crop_size - width
    pad_y = crop_size - height
    in_ = cv2.copyMakeBorder(in_, 0, pad_y, 0, pad_x, cv2.BORDER_CONSTANT, value=mean_value)
    in_ -= np.array(mean_value)
    in_ = in_.transpose((2,0,1))    # HxWx3 -> 3xHxW
    in_ = in_[np.newaxis, ...]      # 3xHxW -> 1x3xHxW
    net.blobs['data'].reshape(*in_.shape)
    net.blobs['data'].data[...] = in_
    net.forward()

    img_base_name = os.path.basename(test_lst[idx_img])
    img_result_name = os.path.splitext(img_base_name)[0]+'.png'
    for idx_cls in xrange(num_cls):
        score_pred = net.blobs['score_output'].data[0][idx_cls, 0:height, 0:width]
        im = (score_pred*255).astype(np.uint8)
        result_root = os.path.join(args.output_dir, 'class_'+str(idx_cls+1))
        if not os.path.exists(result_root):
            os.makedirs(result_root)
        cv2.imwrite(
            os.path.join(result_root, img_result_name),
            im)

    print 'processed: '+test_lst[idx_img]
    sys.stdout.flush()

print 'Done!'
