// ------------------------------------------------------------------
// Written by Yuguang Liu
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/rfcn_layers.hpp"  // here declare RpnClsOHEM class 
#include "caffe/proto/caffe.pb.h" // here declare layer params

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

  template <typename Dtype>
  void RpnClsOHEMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    RpnClsOHEMParameter rpn_cls_param = this->layer_param_.rpn_cls_ohem_param();
    bg_per_img_ = rpn_cls_param.bg_per_img();  //256
    CHECK_GT(bg_per_img_, 0);
    ignore_label_ = rpn_cls_param.ignore_label(); // -1
  }

  template <typename Dtype>
  void RpnClsOHEMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    num_ = bottom[0]->num();   // 1
    CHECK_EQ(1, bottom[0]->channels()); // 1
    height_ = bottom[0]->height(); // 7*hei
    width_ = bottom[0]->width(); // wid
    spatial_dim_ = height_*width_; //
    
    CHECK_EQ(bottom[1]->num(), num_);
    CHECK_EQ(bottom[1]->channels(), 1);
    CHECK_EQ(bottom[1]->height(), height_);
    CHECK_EQ(bottom[1]->width(), width_);

    // Labels for scoring
    top[0]->Reshape(num_, 1, height_, width_);  // 1(num) X 1(ch) X (7*hei) X wid
  }

  template <typename Dtype>
  void RpnClsOHEMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void RpnClsOHEMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }


#ifdef CPU_ONLY
  STUB_GPU(RpnClsOHEMLayer);
#endif

  INSTANTIATE_CLASS(RpnClsOHEMLayer);
  REGISTER_LAYER_CLASS(RpnClsOHEM);

}  // namespace caffe
