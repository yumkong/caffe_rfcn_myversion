#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/normalize_layer.hpp"

namespace caffe {

template <typename Dtype>
void NormalizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  // same with the bottom blob size
  buffer_.Reshape(1, bottom[0]->channels(),
                   bottom[0]->height(), bottom[0]->width());
  // same with channel size
  buffer_channel_.Reshape(1, bottom[0]->channels(), 1, 1);
  // same with spatial size
  buffer_spatial_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
  NormalizeParameter norm_param = this->layer_param().norm_param();
  across_spatial_ = norm_param.across_spatial();
  // use one norm across the spatial position
  if (across_spatial_) {
    norm_.Reshape(bottom[0]->num(), 1, 1, 1);
  } else {
    // use different norm for each spatial position
    norm_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  }
  // avoid dividing zero
  eps_ = norm_param.eps();
  // channel num
  int channels = bottom[0]->channels();
  int spatial_dim = bottom[0]->width() * bottom[0]->height();
  sum_channel_multiplier_.Reshape(1, channels, 1, 1);
  caffe_set(channels, Dtype(1), sum_channel_multiplier_.mutable_cpu_data());
  sum_spatial_multiplier_.Reshape(
      1, 1, bottom[0]->height(), bottom[0]->width());
  caffe_set(spatial_dim, Dtype(1), sum_spatial_multiplier_.mutable_cpu_data());
  channel_shared_ = norm_param.channel_shared();
  // If this layer's param blob has been set up in prototxt file, skip;  
  // otherwise, this LayerSetUp() is used to initialize this layer's param. 
  // Here, blobs_ is used to save normalization parameter
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    if (channel_shared_) {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));
    } else {
      this->blobs_[0].reset(new Blob<Dtype>(vector<int>(1, channels)));
    }
    shared_ptr<Filler<Dtype> > scale_filler;
    if (norm_param.has_scale_filler()) {
      scale_filler.reset(GetFiller<Dtype>(norm_param.scale_filler()));
    } else {
      FillerParameter filler_param;
      filler_param.set_type("constant");
      filler_param.set_value(1.0);
      scale_filler.reset(GetFiller<Dtype>(filler_param));
    }
    scale_filler->Fill(this->blobs_[0].get());
  }
  if (channel_shared_) {
  	// use the same scaling factor for all channels
    CHECK_EQ(this->blobs_[0]->count(), 1)
        << "Scale size is inconsistent with prototxt config";
  } else {
    // use distinct scaling factor for each channel
    CHECK_EQ(this->blobs_[0]->count(), channels)
        << "Scale size is inconsistent with prototxt config";
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  buffer_.Reshape(1, bottom[0]->channels(),
                   bottom[0]->height(), bottom[0]->width());
  if (!across_spatial_) {
    norm_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  }
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  if (spatial_dim != sum_spatial_multiplier_.count()) {
    sum_spatial_multiplier_.Reshape(
        1, 1, bottom[0]->height(), bottom[0]->width());
    caffe_set(spatial_dim, Dtype(1),
              sum_spatial_multiplier_.mutable_cpu_data());
    buffer_spatial_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* scale = this->blobs_[0]->cpu_data();
  Dtype* buffer_data = buffer_.mutable_cpu_data();
  Dtype* norm_data = norm_.mutable_cpu_data();
  // add eps to avoid overflow
  caffe_set<Dtype>(norm_.count(), Dtype(eps_), norm_data);
  const Dtype* sum_channel_multiplier = sum_channel_multiplier_.cpu_data();
  const Dtype* sum_spatial_multiplier = sum_spatial_multiplier_.cpu_data();
  int num = bottom[0]->num();
  // each image's number of elements
  int dim = bottom[0]->count() / num;
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  int channels = bottom[0]->channels();
  // process each image separately
  for (int n = 0; n < num; ++n) {
  	//i = 1:dim, buffer_data[i] = bottom_data[i]^2 
    caffe_sqr<Dtype>(dim, bottom_data, buffer_data);
	// calculate the l2-norm of the whole image
    if (across_spatial_) {
      // add eps to avoid overflow
      // norm = sqrt( |buffer_data[0]| + ...+ |buffer_data[dim-1]| )
      norm_data[n] = pow(caffe_cpu_asum<Dtype>(dim, buffer_data)+eps_,
                         Dtype(0.5));
	  // top_data[i] = bottom_data[i] / norm
      caffe_cpu_scale<Dtype>(dim, Dtype(1.0 / norm_data[n]), bottom_data,
                             top_data);
    } else {
    //CblasTrans == CUBLAS_OP_T: do matrix transpose on buffer_data
    //sum_channel_multiplier: [channels x 1] vector
    //buffer_data --> [channels x spatial_dim] matrix -transpose-> [spatial_dim x channels] matrix
    // norm_data: [spatial_dim x 1] vector
    // norm_data = buffer_data * sum_channel_multiplier
    // sum channels of each spatial position's squared value ==> compute (l2-norm)^2 separately for each position
      caffe_cpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, Dtype(1),
                            buffer_data, sum_channel_multiplier, Dtype(1),
                            norm_data);
      // compute l2-norm norm
      caffe_powx<Dtype>(spatial_dim, norm_data, Dtype(0.5), norm_data);
      // scale the layer
      // buffer_data previously stores the square of each element, here it is overwritten
      // buffer_data = 1 * sum_channel_multiplier x norm_data + 0 * buffer_data
      // ==> duplicates norm_data from 1 x spatial_dim to channels x spatial_dim
      // ==> channels of the same spatial position have same value
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                            1, Dtype(1), sum_channel_multiplier, norm_data,
                            Dtype(0), buffer_data);
	  //i = 1:dim,  top_data[i] = bottom_data[i] / norm[i]
      caffe_div<Dtype>(dim, bottom_data, buffer_data, top_data);
      norm_data += spatial_dim; // shift the data ptr to the next image
    }
    // scale the output
    if (channel_shared_) {
	  // scale all data with the same scaling factor
      caffe_scal<Dtype>(dim, scale[0], top_data);
    } else {
      // overwrite buffer_data to store scaling factor for each channel
      // in each channel, all spatial positions have the same scaling value
      // scale: channels x 1, sum_spatial_multiplier: 1 x spatial_dim
      // buffer_data = scale x sum_spatial_multiplier + 0 * buffer_data
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                            1, Dtype(1), scale, sum_spatial_multiplier,
                            Dtype(0),
                            buffer_data);
	  // normalized data x scaling factor
      caffe_mul<Dtype>(dim, top_data, buffer_data, top_data);
    }
    bottom_data += dim;// shift the data ptr to the next image
    top_data += dim;// shift the data ptr to the next image
  }
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* scale = this->blobs_[0]->cpu_data();
  const Dtype* norm_data = norm_.cpu_data();
  Dtype* buffer_data = buffer_.mutable_cpu_data();
  Dtype* buffer_channel = buffer_channel_.mutable_cpu_data();
  Dtype* buffer_spatial = buffer_spatial_.mutable_cpu_data();
  const Dtype* sum_channel_multiplier = sum_channel_multiplier_.cpu_data();
  const Dtype* sum_spatial_multiplier = sum_spatial_multiplier_.cpu_data();
  int count = top[0]->count();
  int num = top[0]->num();
  int dim = count / num;
  int spatial_dim = top[0]->height() * top[0]->width();
  int channels = top[0]->channels();

  // Propagate to param (param BP)
  if (this->param_propagate_down_[0]) {
    Dtype* scale_diff = this->blobs_[0]->mutable_cpu_diff();
    if (channel_shared_) {
	  //d(loss) / d(scale_factor)
      scale_diff[0] +=
          caffe_cpu_dot<Dtype>(count, top_data, top_diff) / scale[0];
    } else {
      for (int n = 0; n < num; ++n) {
        caffe_mul<Dtype>(dim, top_data+n*dim, top_diff+n*dim, buffer_data);
		//buffer_channel: channels x 1
		//buffer_data: channels x spatial_dim
		//sum_spatial_multiplier: spatial_dim x 1 (all 1's)
		//buffer_channel stores the sum of top_diff[i]x top_data[i] of each channel
		
        caffe_cpu_gemv<Dtype>(CblasNoTrans, channels, spatial_dim, Dtype(1),
                              buffer_data, sum_spatial_multiplier, Dtype(0),
                              buffer_channel);
        // store a / scale[i] in buffer_data temporary
        //i = 1:channels  buffer_channel[i] = buffer_channel[i] / scale[i]
        caffe_div<Dtype>(channels, buffer_channel, scale, buffer_channel);
		// scale_diff = scale_diff + buffer_channel (if have multiple images, 
		// add all scale diff together)
        caffe_add<Dtype>(channels, buffer_channel, scale_diff, scale_diff);
      }
    }
  }

  // Propagate to bottom (data blob BP)
  if (propagate_down[0]) {
    for (int n = 0; n < num; ++n) {
      if (across_spatial_) {
	  	// see notebook for detailed explanation
        Dtype a = caffe_cpu_dot<Dtype>(dim, bottom_data, top_diff);
        caffe_cpu_scale<Dtype>(dim, a / norm_data[n] / norm_data[n],
                               bottom_data, bottom_diff);
        caffe_sub<Dtype>(dim, top_diff, bottom_diff, bottom_diff);
        caffe_scal<Dtype>(dim, Dtype(1.0 / norm_data[n]), bottom_diff);
      } else {
        // dot product between bottom_data and top_diff
        caffe_mul<Dtype>(dim, bottom_data, top_diff, buffer_data);
		// buffer_data: channels x spatial_dim --> spatial_dim x channels
		// sum_channel_multiplier: channels x 1
		// buffer_spatial: spatial_dim x 1
        caffe_cpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, Dtype(1),
                              buffer_data, sum_channel_multiplier, Dtype(0),
                              buffer_spatial);
        // scale bottom_diff
        // sum_channel_multiplier: channels x 1
        // buffer_spatial: 1 x spatial_dim
        // duplicates buffer_spatial in channels
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                              1, Dtype(1), sum_channel_multiplier,
                              buffer_spatial, Dtype(0), buffer_data);
        caffe_mul<Dtype>(dim, bottom_data, buffer_data, bottom_diff);
        // divide by square of norm
        // norm_data: l2_norm of each spatial position
        //buffer_spatial = element-wise square of norm_data
        caffe_powx<Dtype>(spatial_dim, norm_data, Dtype(2), buffer_spatial);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                              1, Dtype(1), sum_channel_multiplier,
                              buffer_spatial, Dtype(0), buffer_data);
        caffe_div<Dtype>(dim, bottom_diff, buffer_data, bottom_diff);
        // subtract
        caffe_sub<Dtype>(dim, top_diff, bottom_diff, bottom_diff);
        // divide by norm
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                              1, Dtype(1), sum_channel_multiplier, norm_data,
                              Dtype(0), buffer_data);
        caffe_div<Dtype>(dim, bottom_diff, buffer_data, bottom_diff);
        norm_data += spatial_dim;
      }
      // scale the diff
      if (channel_shared_) {
        caffe_scal<Dtype>(dim, scale[0], bottom_diff);
      } else {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, spatial_dim,
                              1, Dtype(1), scale, sum_spatial_multiplier,
                              Dtype(0), buffer_data);
        caffe_mul<Dtype>(dim, bottom_diff, buffer_data, bottom_diff);
      }
      bottom_data += dim;
      top_diff += dim;
      bottom_diff += dim;
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(NormalizeLayer);
#endif

INSTANTIATE_CLASS(NormalizeLayer);
REGISTER_LAYER_CLASS(Normalize);

}  // namespace caffe
