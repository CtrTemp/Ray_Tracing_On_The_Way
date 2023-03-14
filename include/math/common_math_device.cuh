#ifndef DEVICE_COMMON_MATH_CUH
#define DEVICE_COMMON_MATH_CUH
#include <stdio.h>

// /**
//  *  后面都应该写成模板函数
//  *  floor 和 ceil 在 CUDA 的 device 端都可以良好运行，没必要
//  * */
// template <typename num_type>
// __device__ inline int floor_device(num_type a)
// {
//     int floor = (int)(a * 10);
//     return floor > 0 ? (floor / 10) : (floor / 10 - 1);
// }

// template <typename num_type>
// __device__ __host__ inline int ceil_device(num_type a)
// {
//     int ceil = (int)(a * 10);
//     return ceil > 0 ? (ceil / 10 + 1) : (ceil / 10);
// }

// __device__ __host__ inline float get_max_float_val(float val1, float val2)
// {
//     return val1 > val2 ? val1 : val2;
// }
// __device__ __host__ inline float get_min_float_val(float val1, float val2)
// {
//     return val1 > val2 ? val2 : val1;
// }

// __device__ __host__ inline int get_max_int_val(int val1, int val2)
// {
//     return val1 > val2 ? val1 : val2;
// }
// __device__ __host__ inline int get_min_int_val(int val1, int val2)
// {
//     return val1 > val2 ? val2 : val1;
// }


#endif