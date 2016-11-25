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
	anchor_num_ = rpn_cls_param.anchor_num();
	CHECK_GT(anchor_num_, 0);
	random_shuffle_percent_ = rpn_cls_param.random_shuffle_percent();  //0.3
	CHECK_GT(random_shuffle_percent_, 0);
    ignore_label_ = rpn_cls_param.ignore_label(); // -1
  }

  template <typename Dtype>
  void RpnClsOHEMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    num_ = bottom[0]->num();   // 1
    CHECK_EQ(1, bottom[0]->channels()); // 1
    height_ = bottom[0]->height(); // 7*hei
    width_ = bottom[0]->width(); // wid
    spatial_dim_ = height_*width_; // 7*hei*wid
    
    CHECK_EQ(bottom[1]->num(), num_);
    CHECK_EQ(bottom[1]->channels(), 1);
    CHECK_EQ(bottom[1]->height(), height_);
    CHECK_EQ(bottom[1]->width(), width_);

    CHECK_EQ(bottom[2]->num(), num_);          // 1
    bbox_channels_ = bottom[2]->channels();   // 4*7
    CHECK_EQ(bottom[2]->height(), height_/anchor_num_); // hei
    CHECK_EQ(bottom[2]->width(), width_);     // width

    CHECK_EQ(bottom[3]->num(), num_);
    CHECK_EQ(bottom[3]->channels(), 1);
    CHECK_EQ(bottom[3]->height(), height_);
    CHECK_EQ(bottom[3]->width(), width_);

    // Labels for scoring
    top[0]->Reshape(num_, 1, height_, width_);  // 1(num) X 1(ch) X (7*hei) X wid
    // Loss weights for bbox regression
    top[1]->Reshape(num_, bbox_channels_, height_ / anchor_num_, width_);// 1(num) X 4*7(ch) X hei X wid
    // Label weights for scoring
    top[2]->Reshape(num_, 1, height_, width_);  // 1(num) X 1(ch) X (7*hei) X wid

	bottom_loss_mirror_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      				 bottom[0]->height(), bottom[0]->width());
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
