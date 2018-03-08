#include <THC/THC.h>
#include <math.h>
#include "cuda/nms_kernel.h"

extern THCState *state;

int nms_apply(THIntTensor* keep, THCudaTensor* boxes_sorted, const float nms_thresh)
{
    int* keep_data = THIntTensor_data(keep);
    const float* boxes_sorted_data = THCudaTensor_data(state, boxes_sorted);

    const int boxes_num = THCudaTensor_size(state, boxes_sorted, 0);

    const int devId = THCudaTensor_getDevice(state, boxes_sorted);

    int numTotalKeep = ApplyNMSGPU(keep_data, boxes_sorted_data, boxes_num, nms_thresh, devId);
    return numTotalKeep;
}


