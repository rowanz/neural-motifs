#ifndef _ROI_ALIGN_KERNEL
#define _ROI_ALIGN_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

__global__ void ROIAlignForward(const int nthreads, const float* image_ptr, const float* boxes_ptr, int num_boxes, int batch, int image_height, int image_width, int crop_height,
  int crop_width, int depth, float extrapolation_value, float* crops_ptr);

int ROIAlignForwardLaucher(
    const float* image_ptr, const float* boxes_ptr,
         int num_boxes,  int batch, int image_height, int image_width, int crop_height,
         int crop_width, int depth, float extrapolation_value, float* crops_ptr, cudaStream_t stream);

__global__ void ROIAlignBackward(const int nthreads, const float* grads_ptr,
    const float* boxes_ptr, int num_boxes, int batch, int image_height,
    int image_width, int crop_height, int crop_width, int depth,
    float* grads_image_ptr);

int ROIAlignBackwardLaucher(const float* grads_ptr, const float* boxes_ptr, int num_boxes,
    int batch, int image_height, int image_width, int crop_height,
    int crop_width, int depth, float* grads_image_ptr, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

