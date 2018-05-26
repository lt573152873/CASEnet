#Copyright (c) 2017 Mitsubishi Electric Research Laboratories (MERL).   All rights reserved.
#
#The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications.  MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose.  In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.
#
#As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes. 

#visualize CASENet (Simple, fast)
# python visualize_multilabel.py ~/datasets/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png
#visualize CASENet (Full, slow)
# python visualize_multilabel.py ~/datasets/Cityscapes/leftImg8bit/val/frankfurt/frankfurt_000000_000294_leftImg8bit.png -g ~/datasets/Cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_edge.bin
import os
import numpy as np
import numpy.random as npr
import cv2
import struct
import sys
import time
import multiprocessing
import argparse


class Timer:
    def __init__(self, unit=1):
        self.start = 0
        self.unit = unit
        self.unit_name = 's' if unit==1 else 'ms' if unit==1000 else '(%fs)'%(1.0/unit)

    def tic(self):
        self.start = time.time()

    def toc(self, verbose=True, tag=''):
        end = time.time()
        ret = (end-self.start) * self.unit
        if verbose or len(tag)>0:
            print(tag+': %.3f'%ret+self.unit_name)
        return ret


def output_color_definition_for_latex(hsv_class, labeli=None,names=None):
    if names is None:
        names = get_class_name_cityscape()
    n_colors = len(hsv_class)
    if labeli is None:
        rgb_list=np.array([[[hi,255,255] for hi in hsv_class]], dtype=np.uint8)
        rgb_list = cv2.cvtColor(rgb_list, cv2.COLOR_HSV2RGB).tolist()[0]
        for (rgb,i) in zip(rgb_list, range(0, n_colors)):
            print '\definecolor{blk_color_%d}{rgb}{%.3f,%.3f,%.3f}' % (i, rgb[0]/255., rgb[1]/255., rgb[2]/255.)
        for i in range(0, n_colors):
            print '\crule[blk_color_%d]{%.3f\columnwidth}{%.3f\columnwidth}'%(i, 1.0/n_colors, 1.0/n_colors)
    else:
        n_labels = len(labeli)
        the_names=[]
        the_rgbs=[]
        for i in range(0,n_labels):
            the_label = labeli[i]
            the_hue = 0.0
            the_cnt = 0
            the_name = ''
            for k in range(0,n_colors):
                if ((the_label >> k) & 1) == 1:
                    the_hue += hsv_class[k]
                    the_cnt += 1
                    the_name += names[k]+'+'
            the_hue /= the_cnt if the_cnt>1 else 1
            rgb = np.array([[[the_hue,255,255]]],dtype=np.uint8)
            the_rgbs.append(cv2.cvtColor(rgb,cv2.COLOR_HSV2RGB)[0,0].tolist())
            the_names.append(the_name)
        for (rgb,i) in zip(the_rgbs, range(0,n_labels)):
            print '\definecolor{blk_color_%d}{rgb}{%.3f,%.3f,%.3f}' % (i, rgb[0] / 255., rgb[1] / 255., rgb[2] / 255.)
        for (name,i) in zip(the_names, range(0,n_labels)):
            print ('\cellcolor{blk_color_%d} '%i)+name[:-1]+' & '


def gen_hsv_class_cityscape():
    return np.array([
        359, # road
        320, # sidewalk
        40, # building
        80, #  wall
        90, #  fence
        10, # pole
        20, #  traffic light
        30, #  traffic sign
        140, # vegetation
        340, # terrain
        280,  # sky
        330, # person
        350, #  rider
        120, # car
        110, #  truck
        130, #  bus
        150, #  train
        160, #  motorcycle
        170 #  bike
    ])/2.0


def get_class_name_cityscape():
    return ['road',
            'sidewalk',
            'building',
            'wall',
            'fence',
            'pole',
            'traffic light',
            'traffic sign',
            'vegetation',
            'terrain',
            'sky',
            'person',
            'rider',
            'car',
            'truck',
            'bus',
            'train',
            'motorcycle',
            'bicycle']


def gen_hsv_class(K_class, h_min=1, h_max=179, do_shuffle=False, do_random=False):
    hsv_class = np.linspace(h_min, h_max, K_class, False, dtype=np.int32)
    if do_shuffle:
        npr.shuffle(hsv_class)
    if do_random:
        hsv_class = npr.randint(h_min, h_max, K_class, dtype=np.int32)
    return hsv_class


def save_each_blended_probk(fname_prefix, prob_gt, prob, thresh=0.5, names=None):
    if names is None:
        names = get_class_name_cityscape()
    rows,cols,nchns = prob.shape
    H_TP = 60.0
    H_FN = 120.0
    # H_FP_OR_TN = 0
    for k in range(0, nchns):
        namek = fname_prefix + names[k] + '.png'
        probk = prob[:,:,k]
        probk_gt = prob_gt[:,:,k]

        label_pos = probk_gt>0
        label_neg = probk_gt<=0
        pred_pos = probk>thresh
        pred_neg = probk<=thresh

        blendk_hsv = np.zeros((rows,cols,3), dtype=np.float32)
        blendk_hsv[:,:,0] = label_pos*(pred_pos*H_TP + pred_neg*H_FN) #classification result type
        blendk_hsv[:,:,1] = label_pos*(pred_pos*probk*255.0 + pred_neg*(1-probk)*255.0) + label_neg*probk*255.0 #probability
        blendk_hsv[:,:,2] = 255
        blend = cv2.cvtColor(blendk_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        cv2.imwrite(namek, blend.astype(np.uint8))


def thresh_and_select_k_largest_only(prob, k=2, thresh=0.0):
    prob[prob <= thresh] = 0

    rows = prob.shape[0]
    cols = prob.shape[1]
    chns = prob.shape[2]

    if k<=0 or k>=chns:
        return prob

    ii, jj = np.meshgrid(range(0,rows), range(0,cols), indexing='ij')

    prob_out = np.zeros(prob.shape, dtype=np.float32)
    for ik in range(0, k):
        idx_max = np.argmax(prob, axis=2)
        prob_out[ii, jj, idx_max] = prob[ii, jj, idx_max]
        prob[ii, jj, idx_max] = -1
    return prob_out


def vis_multilabel(prob, img_h, img_w, K_class, hsv_class=None, use_white_background=False):
    label_hsv = np.zeros((img_h, img_w, 3), dtype=np.float32)
    prob_sum = np.zeros((img_h, img_w), dtype=np.float32)
    prob_edge = np.zeros((img_h, img_w), dtype=np.float32)

    use_abs_color = True
    if hsv_class is None:
        n_colors = 0
        use_abs_color = False
        for k in range(0, K_class):
            if prob[:, :, k].max() > 0:
                n_colors += 1
        hsv_class = gen_hsv_class(n_colors)
        print hsv_class

    i_color = 0
    for k in range(0, K_class):
        prob_k = prob[:, :, k].astype(np.float32)
        if prob_k.max() == 0:
            continue
        hi = hsv_class[ k if use_abs_color else i_color ]
        i_color += 1
        # print '%d: %f' % (k, hi)
        label_hsv[:, :, 0] += prob_k * hi  # H
        prob_sum += prob_k
        prob_edge = np.maximum(prob_edge, prob_k)

    prob_sum[prob_sum == 0] = 1.0
    label_hsv[:, :, 0] /= prob_sum
    if use_white_background:
        label_hsv[:, :, 1] = prob_edge * 255
        label_hsv[:, :, 2] = 255
    else:
        label_hsv[:, :, 1] = 255
        label_hsv[:, :, 2] = prob_edge * 255

    label_bgr = cv2.cvtColor(label_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return label_bgr


def bits2array(vals):
    # print vals
    return [[(valj >> k) & 1
             for k in xrange(0, 32)]
            for valj in vals]


def load_gt(fname_gt, img_h, img_w, use_inner_pool=False):
    # timer = Timer(1000)
    # timer.tic()
    with open(fname_gt, 'rb') as fp:
        bytes = fp.read(img_h * img_w * 4)
    # timer.toc(tag='[load_gt.open]')

    # timer.tic()
    labeli = np.array(struct.unpack('%dI' % (len(bytes) / 4), bytes)).reshape((img_h, img_w))
    # timer.toc(tag='[load_gt.unpack]')

    #find unique labels
    # timer.tic()
    labeli_unique, labeli_count = np.unique(labeli, return_counts=True)
    sidx = np.argsort(labeli_count)[::-1]
    labeli_unique=labeli_unique[sidx]
    # labeli_count=labeli_count[sidx]
    # timer.toc(tag='[load_gt.unique]')

    # timer.tic()
    nbits = 32

    if use_inner_pool:
        # prob = np.float32(map(bits2array,labeli))
        pool = multiprocessing.Pool()
        prob = pool.map(bits2array, labeli.tolist())
        pool.close()
        pool.join()
        prob = np.float32(prob)
    else:
        prob = np.float32([[[(labeli[i, j] >> k) & 1
                             for k in xrange(0, nbits)]
                            for j in xrange(0, img_w)]
                           for i in xrange(0, img_h)])
    # timer.toc(tag='[load_gt.for]')
    return prob, labeli_unique[1:] if labeli_unique[0]==0 else labeli_unique


def load_result(img_h, img_w, result_fmt, K_class, idx_base):
    prob = np.zeros((img_h, img_w, K_class), dtype=np.float32)
    for k in xrange(K_class):
        prob_k = cv2.imread(result_fmt%(k+idx_base), cv2.IMREAD_GRAYSCALE)
        prob[:,:,k] = prob_k.astype(np.float32) / 255.
    return prob


def main(out_name=None, raw_name=None, gt_name=None,
         result_fmt=None, thresh=0.5, do_gt=True, select_only_k = 2,
         do_each_comp = False, K_class = 19, idx_base = 0, save_input=False, dryrun=False):
    oldstdout = sys.stdout
    if do_gt:
        sys.stdout = open(out_name+'.log', 'w')

    hsv_class = gen_hsv_class_cityscape()
    class_names = get_class_name_cityscape()

    timer = Timer(1000)

    #load files
    image = cv2.imread(raw_name)
    img_h, img_w = (image.shape[0], image.shape[1])
    if do_gt:
        # timer.tic()
        prob_gt, labeli = load_gt(gt_name, img_h, img_w)
        # timer.toc(tag='[load_gt]')
        output_color_definition_for_latex(hsv_class, labeli,names=class_names)
    prob = load_result(img_h, img_w, result_fmt, K_class, idx_base)

    # vis gt and blended_gt
    if do_gt:
        # timer.tic()
        label_bgr = vis_multilabel(prob_gt, img_h, img_w, K_class, hsv_class, use_white_background=True)
        # timer.toc(tag='[vis_multilabel for gt]')
        if dryrun:
            print 'writing: '+out_name+'.gt.png'
        else:
            cv2.imwrite(out_name+'.gt.png', label_bgr)
    if save_input:
        if dryrun:
            print 'writing: '+out_name+'.input.png'
        else:
            cv2.imwrite(out_name+'.input.png', image)
        # blended = cv2.addWeighted(image, 0.2, label_bgr, 0.8, 0)
        # cv2.imwrite(fout_prefix+'_blend_gt.png', blended)

    # # vis result
    if select_only_k>0 or thresh>0.0:
        # timer.tic()
        prob_out = thresh_and_select_k_largest_only(prob, select_only_k, thresh)
        prob = prob_out
        # timer.toc(tag='[thresh_and_select_k_largest_only]')
    if do_each_comp and do_gt:
        # timer.tic()
        save_each_blended_probk(out_name+'.comp.',prob_gt,prob,names=class_names)
        # timer.toc(tag='[save_each_blended_probk]')
    label_bgr = vis_multilabel(prob, img_h, img_w, K_class, hsv_class, use_white_background=True)
    if dryrun:
        print 'writing: '+out_name
    else:
        cv2.imwrite(out_name, label_bgr)

    print 'finished: '+raw_name
    sys.stdout = oldstdout


if __name__=='__main__':
    parser = argparse.ArgumentParser(sys.argv[0])
    parser.add_argument('raw_name', type=str, help="input rgb filename")
    parser.add_argument('-o','--output_folder', type=str, default='./visualize/thresh0.6/CASENet',
                        help="visualization output folder")
    parser.add_argument('-g', '--gt_name', type=str, default=None,
                        help="full path to the corresponding multi-label ground truth file")
    parser.add_argument('-f', '--result_fmt', type=str, default='./result_CASENet/class_%d/',
                        help="folders storing testing results for each class")
    parser.add_argument('-t', '--thresh', type=float, default=0.6,
                        help="set any probability<=thresh to 0")
    parser.add_argument('-c', '--do_each_comp', type=int, default=1,
                        help="if gt_name is not None, whether to visualize each class component (1) or not (0)")
    args = parser.parse_args(sys.argv[1:])

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    namei = os.path.basename(args.raw_name)
    print namei
    main(out_name=os.path.join(args.output_folder, namei),
         raw_name=args.raw_name,
         gt_name=args.gt_name,
         result_fmt=args.result_fmt+'%s'%namei,
         do_gt=args.gt_name is not None,
         thresh=args.thresh,
         select_only_k=2,
         do_each_comp = args.do_each_comp)
    print '========================================'
