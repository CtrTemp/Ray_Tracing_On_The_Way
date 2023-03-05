#ifndef DEVICE_RAND_CUH
#define DEVICE_RAND_CUH


#include <curand.h>
#include <curand_kernel.h>

#include "utils/vec3.cuh"



__device__ __forceinline__ double random_float_device(curandStateXORWOW *rand_state)
{
    return curand_uniform(rand_state);
}

__device__ __forceinline__ double random_float_device(double min, double max, curandStateXORWOW *rand_state)
{
    return min + (max - min) * curand_uniform(rand_state);
}

__device__ __forceinline__ vec3 random_vec3_device(curandStateXORWOW *rand_state)
{
    return vec3(curand_uniform(rand_state), curand_uniform(rand_state), curand_uniform(rand_state));
}

__device__ __forceinline__ vec3 random_vec3(double min, double max, curandStateXORWOW *rand_state)
{
    return vec3(random_float_device(min, max, rand_state), random_float_device(min, max, rand_state), random_float_device(min, max, rand_state));
}

__device__ __forceinline__ vec3 random_in_unit_disk_device(curandStateXORWOW *rand_state)
{
    vec3 p;
    do
    {
        p = vec3(random_float_device(-1, 1, rand_state), random_float_device(-1, 1, rand_state), 0);
    } while (dot(p, p) <= 1.0);
    // 模拟在方格中撒点，掉入圆圈的点被收录返回
    return p;
}

__device__ __forceinline__ vec3 random_in_unit_sphere_device(curandStateXORWOW *rand_state)
{
    vec3 p;
    do
    {
        p = random_vec3(-1.0f, 1.0f, rand_state);
    } while (dot(p, p) <= 1.0);
    // 模拟在方格中撒点，掉入圆圈的点被收录返回
    return p;
}


#endif