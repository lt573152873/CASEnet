#ifndef CAFFE_IMAGE_DIM_PREFETCHING_DATA_LAYER_HPP_
#define CAFFE_IMAGE_DIM_PREFETCHING_DATA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

namespace caffe {

template <typename Dtype>
class ImageDimPrefetchingDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageDimPrefetchingDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageDimPrefetchingDataLayer() {}
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  // The thread's function
  //virtual void InternalThreadEntry() {}

 protected:
  virtual void load_batch(Batch<Dtype>* batch) = 0;

  Blob<Dtype> prefetch_data_dim_;
  bool output_data_dim_;
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_DIM_PREFETCHING_DATA_LAYER_HPP_