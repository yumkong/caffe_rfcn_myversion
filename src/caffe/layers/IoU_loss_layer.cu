// --------------------------------------------------------
// IoU loss layer
// Written by Yuguang Liu, 2016.
// An implementation of the IoU loss layer in paper
// 'UnitBox: An Advanced Object Detection Network'
// --------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/fast_rcnn_layers.hpp"
#include "caffe/util/math_functions.hpp"
//#include "caffe/vision_layers.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void kernel_min(const int n, const Dtype* a,
	    const Dtype* b, Dtype* y) {
	  CUDA_KERNEL_LOOP(index, n) {
	    y[index] = min(a[index], b[index]);
	  }
	}

	template <typename Dtype>
	__global__ void kernel_get_height_width(const int num, const int channels,
	    const int spatial_dim, const Dtype* data, Dtype* out_height, Dtype* out_width) 
	{
	  CUDA_KERNEL_LOOP(index, num * spatial_dim) 
	  {
	    int n = index / spatial_dim;
	    int s = index % spatial_dim;
		//height
	    out_height[index] = data[(n * channels + 0) * spatial_dim + s] + data[(n * channels + 1) * spatial_dim + s];
		out_width[index] = data[(n * channels + 2) * spatial_dim + s] + data[(n * channels + 3) * spatial_dim + s];
	  }
	}

	template <typename Dtype>
	__global__ void IouForwardGPU(const int n, const Dtype* i_area,const Dtype* u_area, const Dtype* weight, Dtype* out) 
	{
		// I/U (weight > 0)
		//   0   (otherwise)
		CUDA_KERNEL_LOOP(index, n) {
			Dtype w = weight[index];
			if (w > 0 && u_area[index] > 0) {
				out[index] = -log((i_area[index] + 0.00001) / u_area[index]);
			}
			else {
				out[index] = 0;
			}
		}
	}

	template <typename Dtype>
	void IouLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		int count = bottom[0]->count();
		int spatial_dim = bottom[0]->height() * bottom[0]->width();
		int batch_num = bottom[0]->num();
		int channels = bottom[0]->channels();
		kernel_min<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), min_coords_.mutable_gpu_data());
		CUDA_POST_KERNEL_CHECK;

		// --------compute height and width of each bbox---------------
		// for pred
		kernel_get_height_width<Dtype><<<CAFFE_GET_BLOCKS(batch_num * spatial_dim),CAFFE_CUDA_NUM_THREADS>>>(
			 batch_num, channels, spatial_dim, bottom[0]->gpu_data(), 
	         pred_tb_sum_.mutable_gpu_data(), pred_lr_sum_.mutable_gpu_data());
		// for gt
		kernel_get_height_width<Dtype><<<CAFFE_GET_BLOCKS(batch_num * spatial_dim),CAFFE_CUDA_NUM_THREADS>>>(
	         batch_num, channels, spatial_dim, bottom[1]->gpu_data(), 
	         gt_tb_sum_.mutable_gpu_data(), gt_lr_sum_.mutable_gpu_data());
		// for intersection
		kernel_get_height_width<Dtype><<<CAFFE_GET_BLOCKS(batch_num * spatial_dim),
	      CAFFE_CUDA_NUM_THREADS>>>(batch_num, channels, spatial_dim, min_coords_.gpu_data(), 
	         intersect_height_.mutable_gpu_data(), intersect_width_.mutable_gpu_data());
		CUDA_POST_KERNEL_CHECK;

		// --------compute area of each bbox---------------
		// for intersection
		caffe_gpu_mul(
				batch_num * spatial_dim,
				intersect_height_.gpu_data(),
				intersect_width_.gpu_data(),
				intersect_area_.mutable_gpu_data());
		// for pred
		caffe_gpu_mul(
				batch_num * spatial_dim,
				pred_tb_sum_.gpu_data(),
				pred_lr_sum_.gpu_data(),
				pred_area_.mutable_gpu_data());
		// for gt
		caffe_gpu_mul(
				batch_num * spatial_dim,
				gt_tb_sum_.gpu_data(),
				gt_lr_sum_.gpu_data(),
				gt_area_.mutable_gpu_data());
		// for union (pred + gt - intersection)
		caffe_gpu_add(
				batch_num * spatial_dim,
				pred_area_.gpu_data(),
				gt_area_.gpu_data(),
				union_area_.mutable_gpu_data());    // pred + gt
		caffe_gpu_sub(
				batch_num * spatial_dim,
				union_area_.gpu_data(),
				intersect_area_.gpu_data(),
				union_area_.mutable_gpu_data());    // (pred + gt) - intersection
		
		//  --------compute IoU loss ---------------
		IouForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(batch_num * spatial_dim), CAFFE_CUDA_NUM_THREADS>>>(
			batch_num * spatial_dim, intersect_area_.gpu_data(), union_area_.gpu_data(), 
			bottom[2]->gpu_data(), errors_.mutable_gpu_data());
		CUDA_POST_KERNEL_CHECK;

		Dtype loss;
		caffe_gpu_asum(batch_num * spatial_dim, errors_.gpu_data(), &loss);
		top[0]->mutable_cpu_data()[0] = loss / batch_num / spatial_dim;  // This implementation takes effects for both RPN and Fast R-CNN.

		//top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num(); // This is the original implementation in *Fast* R-CNN
		// For Fast R-CNN, bottom[0]->num() is 128, and spatial_dim is always 1. This implementation is equivalent.
		// For RPN, bottom[0]->num() is 1, and spatial_dim is about 1000x600/16/16 ~ 2400.
		// Also for RPN, in the SoftmaxLoss we use the default normalize version (sum of weights = batch_size = 256 ), so lambda=10 (see paper) will make SoftmaxLoss and SmoothL1Loss roughly balanced.
	}

	template <typename Dtype>
	__global__ void kernel_IpU_div_ImU(const int n,const Dtype* weight, 
										 const Dtype* i_area,const Dtype* u_area, Dtype* out) 
	{
	    CUDA_KERNEL_LOOP(index, n) {
			Dtype w = weight[index];
			if (w > 0) 
			{
				if (i_area[index]> 0 && u_area[index]> 0)
				{
					out[index] = w * (i_area[index] + u_area[index])/ u_area[index] / i_area[index];
				}
				else
				{
					out[index] = 0;
				}
			}
			else 
			{
				out[index] = 0;
			}
		}
	}
	
	template <typename Dtype>
	__global__ void IouBackwardGPU(const int num, const int channels, const int spatial_dim,
									const Dtype* pred_data, const Dtype* gt_data,
									const Dtype* weight, const Dtype* pred_tb, const Dtype* pred_lr,
									const Dtype* intersect_hei, const Dtype* intersect_wid,
									const Dtype* i_area,const Dtype* u_area,	Dtype* out) {

		CUDA_KERNEL_LOOP(index, num * spatial_dim) {
			Dtype w = weight[index];
			if (w <= 0) {
				out[index] = 0;
			}
			else {
				int n = index / spatial_dim;
			    int s = index % spatial_dim;
				//dL/dx_t
				Dtype tmp_dIdx = (pred_data[(n*channels+0)*spatial_dim + s] < gt_data[(n*channels+0)*spatial_dim + s])?intersect_wid[index]:0;
				//tmp_dIdx = tmp_dIdx * (u_area[index] + i_area[index]) / u_area[index] / i_area[index];
				out[(n * channels + 0) * spatial_dim + s] = (u_area[index] > 0)? pred_lr[index]/u_area[index] - i_area[index] * tmp_dIdx:-i_area[index] * tmp_dIdx;
				//dL/dx_b
				tmp_dIdx = (pred_data[(n*channels+1)*spatial_dim + s] < gt_data[(n*channels+1)*spatial_dim + s])?intersect_wid[index]:0;
				//tmp_dIdx = tmp_dIdx * (u_area[index] + i_area[index]) / u_area[index] / i_area[index];
				out[(n * channels + 1) * spatial_dim + s] = (u_area[index] > 0)?pred_lr[index]/u_area[index] - i_area[index] * tmp_dIdx:-i_area[index] * tmp_dIdx;
				//dL/dx_l
				tmp_dIdx = (pred_data[(n*channels+2)*spatial_dim + s] < gt_data[(n*channels+2)*spatial_dim + s])?intersect_hei[index]:0;
				//tmp_dIdx = tmp_dIdx * (u_area[index] + i_area[index]) / u_area[index] / i_area[index];
				out[(n * channels + 2) * spatial_dim + s] = (u_area[index] > 0)?pred_tb[index]/u_area[index] - i_area[index] * tmp_dIdx:-i_area[index] * tmp_dIdx;
				//dL/dx_r
			    tmp_dIdx = (pred_data[(n*channels+3)*spatial_dim + s] < gt_data[(n*channels+3)*spatial_dim + s])?intersect_hei[index]:0;
				//tmp_dIdx = tmp_dIdx * (u_area[index] + i_area[index]) / u_area[index] / i_area[index];
				out[(n * channels + 3) * spatial_dim + s] = (u_area[index] > 0)?pred_tb[index]/u_area[index] - i_area[index] * tmp_dIdx:-i_area[index] * tmp_dIdx;
			}
		}
	}

	template <typename Dtype>
	void IouLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if (propagate_down[1]) {
			LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
		}
		if (propagate_down[0]) {
			int count = bottom[0]->count();
			int spatial_dim = bottom[0]->height() * bottom[0]->width();
			int batch_num = bottom[0]->num();
			int channels = bottom[0]->channels();
			//first do a helper computation: I = (I + U) / (I*U)
			kernel_IpU_div_ImU<Dtype><<<CAFFE_GET_BLOCKS(batch_num * spatial_dim), CAFFE_CUDA_NUM_THREADS>>>(
									batch_num * spatial_dim, bottom[2]->gpu_data(), 
									intersect_area_.gpu_data(), union_area_.gpu_data(), 
									intersect_area_.mutable_gpu_data());
			
			IouBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
				batch_num, channels, spatial_dim, 
				bottom[0]->gpu_data(), bottom[1]->gpu_data(), bottom[2]->gpu_data(), 
				pred_tb_sum_.gpu_data(), pred_lr_sum_.gpu_data(),
				intersect_height_.gpu_data(), intersect_width_.gpu_data(),
				intersect_area_.gpu_data(), union_area_.gpu_data(),
				diff_.mutable_gpu_data());
			CUDA_POST_KERNEL_CHECK;
			
			//const Dtype sign = (i == 0) ? 1 : -1;
			//int spatial_dim = diff_.height() * diff_.width();
			//const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num() / spatial_dim;
			//LOG(INFO) << "top_diff = " << top[0]->cpu_diff()[0] << ", spatial_dim = " << spatial_dim << ", batch_num = " << batch_num;
			const Dtype alpha = top[0]->cpu_diff()[0] / batch_num / spatial_dim;
			//const Dtype alpha = top[0]->cpu_diff()[0];
			caffe_gpu_axpby(
				count,              			 // count
				alpha,                           // alpha
				diff_.gpu_data(),                // x
				Dtype(0),                        // beta
				bottom[0]->mutable_gpu_diff());  // y	
			
		}

		
	}
	INSTANTIATE_LAYER_GPU_FUNCS(IouLossLayer);
}  // namespace caffe

