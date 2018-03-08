int roi_align_forward_cuda(int crop_height, int crop_width, float spatial_scale,
                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output);

int roi_align_backward_cuda(int crop_height, int crop_width, float spatial_scale,
                        THCudaTensor * top_grad, THCudaTensor * rois,
                        THCudaTensor * bottom_grad);
