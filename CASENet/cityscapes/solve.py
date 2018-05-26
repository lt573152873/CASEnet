#Copyright (c) 2017 Mitsubishi Electric Research Laboratories (MERL).   All rights reserved.
#
#The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications.  MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose.  In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages.
#
#As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes. 

import os
import sys
import argparse

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('solver_prototxt_file', type=str,
                    help="path to the solver prototxt file")
parser.add_argument('-c', '--pycaffe_folder', type=str, default='../../code/python',
                    help="pycaffe folder that contains the caffe/_caffe.so file")
parser.add_argument('-m', '--init_model', type=str, default='./model/init_res_coco.caffemodel',
                    help="path to the initial caffemodel")
parser.add_argument('-g', '--gpu', type=int, default=0,
                    help="use which gpu device (default=0)")
args = parser.parse_args(sys.argv[1:])

assert(os.path.exists(args.solver_prototxt_file))
assert(os.path.exists(args.init_model))

if os.path.exists(os.path.join(args.pycaffe_folder,'caffe/_caffe.so')):
    sys.path.insert(0, args.pycaffe_folder)
import caffe

caffe.set_mode_gpu()
caffe.set_device(args.gpu)

solver = caffe.SGDSolver(args.solver_prototxt_file)
solver.net.copy_from(args.init_model)

solver.solve()
