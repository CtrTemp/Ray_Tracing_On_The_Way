#ifndef FOO_CUH
#define FOO_CUH

#include <stdio.h>
//添加cuda库
#include "cuda_runtime.h"
#include "device_launch_parameters.h"





//非模板函数才可以进行C连接
extern "C" float *matAdd(float *a,float *b,int length);

#endif
