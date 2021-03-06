// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#ifndef CAFFE_FAST_RCNN_LAYERS_HPP_
#define CAFFE_FAST_RCNN_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

  /* ROIPoolingLayer - Region of Interest Pooling Layer
  */
  template <typename Dtype>
  class ROIPoolingLayer : public Layer<Dtype> {
  public:
    explicit ROIPoolingLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "ROIPooling"; }

    virtual inline int MinBottomBlobs() const { return 2; }
    virtual inline int MaxBottomBlobs() const { return 2; }
    virtual inline int MinTopBlobs() const { return 1; }
    virtual inline int MaxTopBlobs() const { return 1; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    int channels_;
    int height_;
    int width_;
    int pooled_height_;
    int pooled_width_;
    Dtype spatial_scale_;
    Blob<int> max_idx_;
  };

  template <typename Dtype>
  class SmoothL1LossLayer : public LossLayer<Dtype> {
  public:
    explicit SmoothL1LossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "SmoothL1Loss"; }

    virtual inline int ExactNumBottomBlobs() const { return -1; }
    virtual inline int MinBottomBlobs() const { return 2; }
    virtual inline int MaxBottomBlobs() const { return 3; }
    virtual inline int ExactNumTopBlobs() const { return -1; }
    virtual inline int MinTopBlobs() const { return 1; }
    virtual inline int MaxTopBlobs() const { return 2; }

    /**
    * Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
    * to both inputs -- override to return true and always allow force_backward.
    */
    virtual inline bool AllowForceBackward(const int bottom_index) const {
      return true;
    }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    /// Read the normalization mode parameter and compute the normalizer based
    /// on the blob size. 
    virtual Dtype get_normalizer(
      LossParameter_NormalizationMode normalization_mode, Dtype pre_fixed_normalizer);

    Blob<Dtype> diff_;
    Blob<Dtype> errors_;
    bool has_weights_;

    int outer_num_, inner_num_;

    /// How to normalize the output loss.
    LossParameter_NormalizationMode normalization_;
  };

// 1030 added IoU_loss_layer
    template <typename Dtype>
    class IouLossLayer : public LossLayer<Dtype> {
    public:
	explicit IouLossLayer(const LayerParameter& param)
		: LossLayer<Dtype>(param){}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "IouLoss"; }

	virtual inline int ExactNumBottomBlobs() const { return -1; }
	virtual inline int MinBottomBlobs() const { return 2; }
	virtual inline int MaxBottomBlobs() const { return 3; }

	/**
	* Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
	* to both inputs -- override to return true and always allow force_backward.
	*/
	virtual inline bool AllowForceBackward(const int bottom_index) const {
		return true;
	}

    protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	Blob<Dtype> min_coords_; //x_t + x_b;
	Blob<Dtype> pred_tb_sum_; //x_t + x_b;
	Blob<Dtype> pred_lr_sum_; //x_l + x_r;
	Blob<Dtype> gt_tb_sum_; //x_t + x_b;
	Blob<Dtype> gt_lr_sum_; //x_l + x_r;
	Blob<Dtype> intersect_height_; //min(x_t_pred, x_t_gt) + min(x_b_pred, x_b_gt);
	Blob<Dtype> intersect_width_; //min(x_l_pred, x_l_gt) + min(x_r_pred, x_r_gt);
	Blob<Dtype> intersect_area_; //intersect_height_ .* intersect_width_
	Blob<Dtype> pred_area_; //
	Blob<Dtype> gt_area_; //
	Blob<Dtype> union_area_; //pred_area_ + gt_area_ - intersect_area_
	Blob<Dtype> errors_; // element-wise loss value: -log(intersect_area_./union_area_)
	Blob<Dtype> diff_; // for backpropogation
	bool has_weights_;
    };

}  // namespace caffe

#endif  // CAFFE_FAST_RCNN_LAYERS_HPP_

