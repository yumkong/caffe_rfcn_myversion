// ------------------------------------------------------------------
// Written by Yuguang Liu
// ------------------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/rfcn_layers.hpp"

using std::max;
using std::min;

namespace caffe {  
  template <typename Dtype>
  void RpnClsOHEMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    //bottom_rois is a pointer pointing to the 1st element of the matrix bottom[0]
    const Dtype* bottom_loss = bottom[0]->cpu_data(); // 1(num) x 1(ch) x 7*hei x wid 
    const Dtype* bottom_labels = bottom[1]->cpu_data(); // 1(num) x 1(ch) x 7*hei x wid 
    const Dtype* bottom_bbox_loss_weights = bottom[2]->cpu_data();// 1(num) x 4*7(ch) x hei x wid
    const Dtype* bottom_labels_weight = bottom[3]->cpu_data(); // 1(num) x 1(ch) x 7*hei x wid 
    Dtype* top_labels = top[0]->mutable_cpu_data();    // 1(num) x 1(ch) x 7*hei x wid 
    Dtype* top_bbox_loss_weights = top[1]->mutable_cpu_data();
	Dtype* top_labels_weight = top[2]->mutable_cpu_data(); 
	//1118 added
	Dtype* bottom_loss_mirror_ptr = bottom_loss_mirror_.mutable_cpu_data();
	caffe_set(bottom[0]->count(), Dtype(0), bottom_loss_mirror_ptr); // init label_weights to 0
	
    caffe_set(top[0]->count(), Dtype(ignore_label_), top_labels); // init labels_ohem to -1
    caffe_set(top[1]->count(), Dtype(0), top_bbox_loss_weights); // init bbox_weights to 0
    caffe_set(top[2]->count(), Dtype(0), top_labels_weight); // init label_weights to 0

    int num_anchors_ = bottom[0]->count();// num of all elements

    // Find anchors with max cls loss
    vector<int> sorted_idx(num_anchors_); // num of rois
    for (int i = 0; i < num_anchors_; i++){
      sorted_idx[i] = i; // init orderly from 0 to (num_rois -1)
    }
	// below is a lambda function, a simple example is:
	//// sort using a lambda expression 
    ////std::sort(s.begin(), s.end(), [](int a, int b) {
    ////    return a > b;   // 1st element should be larger than the 2nd element
    ////});
    ////===> after sorting, s will be in descending order
	//similarly, the below is also to sort sorted_idx in descending order, the result is 
	// bottom_loss[sorted_idx[0]] > bottom_loss[sorted_idx[1]] > ... > bottom_loss[sorted_idx[end]] 
	// as a result, the beginning of the sorted index contains the largest [loss_cls + loss_bbox] error
	int fg_left = bg_per_img_ / 4; // number_left(2, 128)==> 2 ints with value 128
    int bg_left = bg_per_img_ - fg_left;
	// 1115 added
	int anchor_num = 7;
	int single_hei = height_ / 7;
	int single_area = single_hei * width_;
	//1118 1/3: only keep the max out of 7 anchors
	for(int k = 0; k < single_area; k++)
	{
		int tmp_idx = 0;
		int tmp_max = bottom_loss[k];
		for(int kk = 1; kk < anchor_num; kk++)
		{
			if(bottom_loss[k + kk*single_area] > tmp_max)
			{
				tmp_max = bottom_loss[k + kk*single_area];
				tmp_idx = kk;
			}
		}
		// only save the largest loss value out of 7 anchors of the same position, others set to 0
		bottom_loss_mirror_ptr[k + tmp_idx*single_area] = bottom_loss[k + tmp_idx*single_area];
	}
	//1118 2/3: width-wise nms
	for(int k = 0; k < height_; k++)
	{
		int start_idx = k * width_;
		for (int w_idx = 1; w_idx < width_-1; w_idx ++)
		{
			if(bottom_loss_mirror_ptr[start_idx+w_idx] <= bottom_loss_mirror_ptr[start_idx+w_idx-1] 
				|| bottom_loss_mirror_ptr[start_idx+w_idx] <= bottom_loss_mirror_ptr[start_idx+w_idx+1])
			{	bottom_loss_mirror_ptr[start_idx+w_idx] = 0; }
		}
	}
	//1118 3/3: height-wise nms
	for( int k = 0; k < anchor_num; k++ )
	{
		int start_idx1 = k * single_area;
		for (int h_idx = 1; h_idx < single_hei - 1; h_idx++)
		{
			int start_idx2 = start_idx1 + h_idx * width_;
			for (int w_idx = 0; w_idx < width_; w_idx ++)
			{
				if(bottom_loss_mirror_ptr[start_idx2+w_idx] <= bottom_loss_mirror_ptr[start_idx2+w_idx-width_] 
					|| bottom_loss_mirror_ptr[start_idx2+w_idx] <= bottom_loss_mirror_ptr[start_idx2+w_idx+width_])
				{	bottom_loss_mirror_ptr[start_idx2+w_idx] = 0; }
			}
		}
	}
	//1118 4/3: sort sorted_idx in bottom_loss_mirror's ascending order
	std::sort(sorted_idx.begin(), sorted_idx.end(),
      			[bottom_loss_mirror_ptr](int i1, int i2){return bottom_loss_mirror_ptr[i1] < bottom_loss_mirror_ptr[i2]; });
	//1118 5/3: get the first element's idx larger than 0
	std::vector<int>::iterator up;
    up = std::upper_bound (sorted_idx.begin(), sorted_idx.end(), 0); // 
    int random_shuffle_num = (int)(sorted_idx.end() - up)*random_shuffle_percent_;
	LOG(INFO) << "first non-zero idx = " << (int)(sorted_idx.begin() - up) << ", random_shuffle_num = " << random_shuffle_num;
	//1118 5/3: inverse sorted_idx
	std::reverse(sorted_idx.begin(),sorted_idx.end());
	LOG(INFO) << "Before shuffle: loss[0] = " << bottom_loss_mirror_ptr[sorted_idx[0]] << ", loss[10] = "<< bottom_loss_mirror_ptr[sorted_idx[10]]
			  <<", loss[100] = "<< bottom_loss_mirror_ptr[sorted_idx[100]];  // check whether in descending order
	//1118 6/3: shuffle the first random_shuffle_percent_ of sorted_idx's non-zero elements
	std::random_shuffle ( sorted_idx.begin(), sorted_idx.begin() + random_shuffle_num );
	LOG(INFO) << "After shuffle: loss[0] = " << bottom_loss_mirror_ptr[sorted_idx[0]] << ", loss[10] = "<< bottom_loss_mirror_ptr[sorted_idx[10]]
			  <<", loss[100] = "<< bottom_loss_mirror_ptr[sorted_idx[100]];

	//std::sort(sorted_idx.begin(), sorted_idx.end(),
    //  			[bottom_loss](int i1, int i2){return bottom_loss[i1] > bottom_loss[i2]; });
	//1117 added to get the 
	// necessary???
    // 1116 added For debug purpose: using built-in random generator to randomly shuffle sorted_idx
    //int random_shuffle_num = std::floor(random_shuffle_percent_ * num_anchors_);
    //std::random_shuffle ( sorted_idx.begin(), sorted_idx.end() );
    //std::random_shuffle ( sorted_idx.begin(), sorted_idx.begin() + random_shuffle_num );
    // Generate output labels for cls and loss_weights for bbox regression
    ////std::vector<int> second (4,100);                       // four ints with value 100
    
    for (int i = 0; i < num_anchors_; i++)
	{
		int index = sorted_idx[i];
		//int s = index % (width_*height_); // x % 1 == 0
		//int n = index / (width_*height_); // x / 1 == x
		if (bottom_labels[index] > 0) //if fg, directly copy the label value from bottom to top
		{
		     if (fg_left > 0) // if this image still has quota
		     {
				fg_left--;
				top_labels[index] = bottom_labels[index]; 
				top_labels_weight[index] = bottom_labels_weight[index]; 
				//int anchor_idx = index / width_ % 7;
				//int hei_idx = index / width_ / 7;
				int anchor_idx = index / width_ / single_hei;
				int hei_idx = index / width_ % single_hei;

				int wid_idx = index % width_;
				for (int j = 0; j < 4; j++) //copy bbox weights from bottom to top
				{
					//int bbox_index = index * 4 + j;
					//int bbox_index = j * spatial_dim_ + index;
					int bbox_index = single_hei * width_ * 4 * anchor_idx + single_hei * width_ * j + width_*hei_idx+wid_idx;
					top_bbox_loss_weights[bbox_index] = bottom_bbox_loss_weights[bbox_index];
				}
		     }
		}
		else
		{   //if bg, only copy the first bg_per_img_ label values from bottom to top, ignore others (-1) 
			if (bg_left > 0) // if this image still has quota
			{
				bg_left--; // this image's quota --
				// the corresponding top_label is set to the value of the bottom label in the same position
				//(all other position are init to -1)
				top_labels[index] = bottom_labels[index]; 
				top_labels_weight[index] = bottom_labels_weight[index]; 
			}
		}
    }
  }

  template <typename Dtype>
  void RpnClsOHEMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    return; // do nothing for Backward
  }

  INSTANTIATE_LAYER_GPU_FUNCS(RpnClsOHEMLayer);

}  // namespace caffe
