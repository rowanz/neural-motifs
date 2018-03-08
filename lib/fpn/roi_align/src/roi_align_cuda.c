#include <THC/THC.h>
#include <math.h>
#include "cuda/roi_align_kernel.h"

extern THCState *state;

int roi_align_forward_cuda(int crop_height, int crop_width, float spatial_scale,
                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output)
{
    // Grab the input tensor
    float * image_ptr = THCudaTensor_data(state, features);
    float * boxes_ptr = THCudaTensor_data(state, rois);

    float * crops_ptr = THCudaTensor_data(state, output);

    // Number of ROIs
    int num_boxes = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 5)
    {
        return 0;
    }

    // batch size
    int batch = THCudaTensor_size(state, features, 0);
    // data height
    int image_height = THCudaTensor_size(state, features, 2);
    // data width
    int image_width = THCudaTensor_size(state, features, 3);
    // Number of channels
    int depth = THCudaTensor_size(state, features, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);
    float extrapolation_value = 0.0;

    ROIAlignForwardLaucher(
         image_ptr, boxes_ptr, num_boxes, batch, image_height, image_width,
         crop_height, crop_width, depth, extrapolation_value, crops_ptr,
         stream);

    return 1;
}

int roi_align_backward_cuda(int crop_height, int crop_width, float spatial_scale,
    THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad)
{
    // Grab the input tensor
    float * grads_ptr = THCudaTensor_data(state, top_grad);
    float * boxes_ptr = THCudaTensor_data(state, rois);

    float * grads_image_ptr = THCudaTensor_data(state, bottom_grad);

    // Number of ROIs
    int num_boxes = THCudaTensor_size(state, rois, 0);
    int size_rois = THCudaTensor_size(state, rois, 1);
    if (size_rois != 5)
    {
        return 0;
    }

    // batch size
    int batch = THCudaTensor_size(state, bottom_grad, 0);
    // data height
    int image_height = THCudaTensor_size(state, bottom_grad, 2);
    // data width
    int image_width = THCudaTensor_size(state, bottom_grad, 3);
    // Number of channels
    int depth = THCudaTensor_size(state, bottom_grad, 1);

    cudaStream_t stream = THCState_getCurrentStream(state);

    ROIAlignBackwardLaucher(
        grads_ptr, boxes_ptr, num_boxes, batch, image_height, image_width,
        crop_height, crop_width, depth, grads_image_ptr, stream);
    return 1;
}
