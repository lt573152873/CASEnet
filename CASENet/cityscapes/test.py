#Copyright (c) 2017 Mitsubishi Electric Research Laboratories (MERL).   All rights reserved.
#
#The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications.  MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose.  In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.
#
#As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes. 

# test CASENet:
# python test.py ./config/test_CASENet.prototxt ./model/CASENet_iter_40000.caffemodel -l ~/datasets/Cityscapes/valEdgeBin.txt -d ~/Dataset/Cityscapes -o ./result_CASENet -c ../../caffe/build/install/python
# valEdgeBin.txt file can be simply a list file of all the validation images in Cityscapes, each line storeing a single file relative to the Cityscapes dataset's root folder, such as:
# /leftImg8bit/val/frankfurt/frankfurt_000000_002963_leftImg8bit.png

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
num_cls = 19
image_h = 1024 # Need to pre-determine test image size
image_w = 2048 # Need to pre-determine test image size
patch_h = 512
patch_w = 512
step_size_y = 256
step_size_x = 384
pad = 16
if ((2*pad)%8)!=0:
    raise ValueError('Pad number must be able to be divided by 8!')
step_num_y = (image_h-patch_h+0.0)/step_size_y
step_num_x = (image_w-patch_w+0.0)/step_size_x
if(round(step_num_y)!=step_num_y):
    raise ValueError('Vertical sliding size can not be divided by step size!')
if(round(step_num_x)!=step_num_x):
    raise ValueError('Horizontal sliding size can not be divided by step size!')
step_num_y=int(step_num_y)
step_num_x=int(step_num_x)
mean_value = (104.008, 116.669, 122.675) #BGR

for idx_img in xrange(len(test_lst)):
    in_ = cv2.imread(test_lst[idx_img]).astype(np.float32)
    width, height, chn = in_.shape[1], in_.shape[0], in_.shape[2]
    im_array = cv2.copyMakeBorder(in_, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    if(height!=image_h or width!=image_w):
        raise ValueError('Input image size must be'+str(image_h)+'x'+str(image_w)+'!')
    
    # Perform patch-by-patch testing
    score_pred = np.zeros((height, width, num_cls))
    mat_count = np.zeros((height, width, 1))
    for i in range(0, step_num_y+1):
        offset_y = i*step_size_y
        for j in range(0, step_num_x+1):
            offset_x = j*step_size_x
            # crop overlapped regions from the image
            in_ = np.array(im_array[offset_y:offset_y+patch_h+2*pad, offset_x:offset_x+patch_w+2*pad, :])
            in_ -= np.array(mean_value)
            in_ = in_.transpose((2,0,1))    # HxWx3 -> 3xHxW
            in_ = in_[np.newaxis, ...]      # 3xHxW -> 1x3xHxW
            net.blobs['data'].reshape(*in_.shape)
            net.blobs['data'].data[...] = in_
            net.forward()
            
            # add the prediction to score_pred and increase count by 1
            score_pred[offset_y:offset_y+patch_h, offset_x:offset_x+patch_w, :] += \
                np.transpose(net.blobs['score_output'].data[0], (1, 2, 0))[pad:-pad,pad:-pad,:]
            mat_count[offset_y:offset_y+patch_h, offset_x:offset_x+patch_w, 0] += 1.0
    score_pred = np.divide(score_pred, mat_count)
    
    img_base_name = os.path.basename(test_lst[idx_img])
    img_result_name = os.path.splitext(img_base_name)[0]+'.png'
    for idx_cls in xrange(num_cls):
        im = (score_pred[:,:,idx_cls]*255).astype(np.uint8)
        result_root = os.path.join(args.output_dir, 'class_'+str(idx_cls))
        if not os.path.exists(result_root):
            os.makedirs(result_root)
        cv2.imwrite(
            os.path.join(result_root, img_result_name),
            im)

    print 'processed: '+test_lst[idx_img]
    sys.stdout.flush()

print 'Done!'
