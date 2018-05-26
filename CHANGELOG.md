We have modified Caffe, an open-source deep learning tool, to implement our CASENet paper. The modification details are listed below.

1. Newly added (for CASENet):
include\caffe\layers\multichannel_reweighted_sigmoid_cross_entropy_loss_layer.hpp
src\caffe\layers\multichannel_reweighted_sigmoid_cross_entropy_loss_layer.cpp

2. Copied/modified from the official/DeepLab-v2/HED versions of Caffe (for I/O):
include\caffe\layers\image_dim_prefetching_data_layer.hpp
include\caffe\layers\image_seg_data_layer.hpp
include\caffe\layers\base_data_layer.hpp
include\caffe\data_transformer.hpp
include\caffe\layers\reweighted_sigmoid_cross_entropy_loss_layer.hpp
src\caffe\layers\image_dim_prefetching_data_layer.cpp
src\caffe\layers\image_dim_prefetching_data_layer.cu
src\caffe\layers\image_seg_data_layer.cpp
src\caffe\layers\reweighted_sigmoid_cross_entropy_loss_layer.cpp
src\caffe\data_transformer.cpp
src\caffe\proto\caffe.proto