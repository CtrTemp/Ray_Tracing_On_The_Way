#ifndef CAST_RAY_CUH
#define CAST_RAY_CUH

#include <stdio.h>
// 添加cuda库
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// template <typename T>


// 非模板函数才可以进行C连接
extern "C" float *matAdd(float *a, float *b, int length);
extern "C" float *cast_ray_cu(float frame_width, float frame_height, int spp);

#endif
