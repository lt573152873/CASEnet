CASENet Python Scripts Usage
==============

Training Script: solve.py
--------------

```
usage: solve.py [-h] [-c PYCAFFE_FOLDER] [-m INIT_MODEL] [-g GPU]
                solver_prototxt_file

positional arguments:
  solver_prototxt_file  path to the solver prototxt file

optional arguments:
  -h, --help            show this help message and exit
  -c PYCAFFE_FOLDER, --pycaffe_folder PYCAFFE_FOLDER
                        pycaffe folder that contains the caffe/_caffe.so file
  -m INIT_MODEL, --init_model INIT_MODEL
                        path to the initial caffemodel
  -g GPU, --gpu GPU     use which gpu device (default=0)
```

Testing Script: test.py
-------------

```
usage: test.py [-h] [-l IMAGE_LIST] [-f IMAGE_FILE] [-d IMAGE_DIR]
               [-o OUTPUT_DIR] [-c PYCAFFE_FOLDER] [-g GPU]
               deploy_prototxt_file model

positional arguments:
  deploy_prototxt_file  path to the deploy prototxt file
  model                 path to the caffemodel containing the trained weights

optional arguments:
  -h, --help            show this help message and exit
  -l IMAGE_LIST, --image_list IMAGE_LIST
                        list of image files to be tested
  -f IMAGE_FILE, --image_file IMAGE_FILE
                        a single image file to be tested
  -d IMAGE_DIR, --image_dir IMAGE_DIR
                        root folder of the image files in the list or the
                        single image file
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        folder to store the test results
  -c PYCAFFE_FOLDER, --pycaffe_folder PYCAFFE_FOLDER
                        pycaffe folder that contains the caffe/_caffe.so file
  -g GPU, --gpu GPU     use which gpu device (default=0)
```

Visualization Script: visualize_multilabel.py
--------------------------

```
usage: visualize_multilabel.py [-h] [-o OUTPUT_FOLDER] [-g GT_NAME]
                               [-f RESULT_FMT] [-t THRESH] [-c DO_EACH_COMP]
                               raw_name

positional arguments:
  raw_name              input rgb filename

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        visualization output folder
  -g GT_NAME, --gt_name GT_NAME
                        full path to the corresponding multi-label ground
                        truth file
  -f RESULT_FMT, --result_fmt RESULT_FMT
                        folders storing testing results for each class
  -t THRESH, --thresh THRESH
                        set any probability<=thresh to 0
  -c DO_EACH_COMP, --do_each_comp DO_EACH_COMP
                        if gt_name is not None, whether to visualize each
                        class component (1) or not (0)

```



**If you use this package, please cite our CVPR 2017 paper:**

```
@inproceedings{yu2017casenet, 
  title={uppercase{CASEN}et: Deep Category-Aware Semantic Edge Detection}, 
  author={Z. Yu and C. Feng and M. Y. Liu and S. Ramalingam}, 
  booktitle={IEEE Conf. on Computer Vision and Pattern Recognition}, 
  year={2017}
}
```


Contact
-------

[Zhiding Yu](yzhiding@andrew.cmu.edu)   
[Chen Feng](simbaforrest@gmail.com)

**Feel free to email any bugs or suggestions to help us improve the code. Thank you!**