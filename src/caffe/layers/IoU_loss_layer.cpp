// --------------------------------------------------------
// IoU loss layer
// Written by Yuguang Liu, 2016.
// An implementation of the IoU loss layer in paper
// 'UnitBox: An Advanced Object Detection Network'
// --------------------------------------------------------

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
//#include "caffe/loss_layers.hpp"
#include "caffe/fast_rcnn_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
void IouLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_weights_ = (bottom.size() == 3); // pred + gt + weigths
}

template <typename Dtype>
void IouLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  if (has_weights_) {
    //CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
    CHECK_EQ(1, bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[2]->height());
    CHECK_EQ(bottom[0]->width(), bottom[2]->width());
  }
  //[min(pred_t, gt_t), min(pred_b, gt_b), min(pred_l, gt_l), min(pred_r, gt_r)]
  min_coords_.Reshape(bottom[0]->num(), bottom[0]->channels(),bottom[0]->height(), bottom[0]->width());
  pred_tb_sum_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  pred_lr_sum_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  pred_area_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());

  gt_tb_sum_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  gt_lr_sum_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  gt_area_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());

  intersect_height_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  intersect_width_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  intersect_area_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  
  union_area_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  errors_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());

  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void IouLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void IouLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(IouLossLayer);
#endif

INSTANTIATE_CLASS(IouLossLayer);
REGISTER_LAYER_CLASS(IouLoss);

}  // namespace caffe




