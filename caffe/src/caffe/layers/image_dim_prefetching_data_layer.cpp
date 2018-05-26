#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/image_dim_prefetching_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
void ImageDimPrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  if (top.size() == 3) {
    output_data_dim_ = true;
  } else {
    output_data_dim_ = false;
  }
  for (int i = 0; i < BasePrefetchingDataLayer<Dtype>::prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.mutable_cpu_data();
    if (this->output_labels_) {
      this->prefetch_[i]->label_.mutable_cpu_data();
    }
    if (output_data_dim_) {
      this->prefetch_[i]->dim_.mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < BasePrefetchingDataLayer<Dtype>::prefetch_.size(); ++i) {
      this->prefetch_[i]->data_.mutable_gpu_data();
      if (this->output_labels_) {
        this->prefetch_[i]->label_.mutable_gpu_data();
      }
      if (output_data_dim_) {
	this->prefetch_[i]->dim_.mutable_gpu_data();
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  BasePrefetchingDataLayer<Dtype>::StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void ImageDimPrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = 
    this->prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.cpu_data(),
        top[1]->mutable_cpu_data());
  }
  if (output_data_dim_) {
    top[2]->ReshapeLike(batch->dim_);
    caffe_copy(batch->dim_.count(), batch->dim_.cpu_data(),
	       top[2]->mutable_cpu_data());
  }

  this->prefetch_free_.push(batch);
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(ImageDimPrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(ImageDimPrefetchingDataLayer);

}  // namespace caffe
