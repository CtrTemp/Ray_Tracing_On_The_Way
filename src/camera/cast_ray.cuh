#ifndef CAST_RAY_CUH
#define CAST_RAY_CUH

#include <stdio.h>
// 添加cuda库
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "utils/ray.cuh"
#include "camera.cuh"
#include "iostream"
#include <sys/time.h>
// template <typename T>
#include "../global_init.cuh"

// 非模板函数才可以进行C连接
extern "C" float *matAdd(float *a, float *b, int length);
extern "C" vec3 *cast_ray_cu(float frame_width, float frame_height, int spp);

#endif
