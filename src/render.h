#ifndef RENDER_H
#define RENDER_H



#include "camera/camera.cuh"
#include "object/geometry/sphere.cuh"
#include "object/primitive/triangle.cuh"
#include "object/model/models.cuh"
#include "material/lambertian.cuh"
#include "material/mental.cuh"
#include "material/dielectric.cuh"
#include "material/diffuse_light.cuh"
#include "texture/perlin.cuh"

#include <sys/time.h>


/* ##################################### 随机数初始化 ##################################### */
// __host__ curandStateXORWOW_t *init_rand(int block_size_width, int block_size_height);
__global__ void initialize_device_random(curandStateXORWOW_t *states, unsigned long long seed, size_t size);

/* ##################################### 摄像机初始化 ##################################### */
// __host__ void init_camera(void);
__host__ camera *createCamera(void);

/* ###################################### 场景初始化 ###################################### */
// __host__ hitable **init_world(curandStateXORWOW_t *rand_states);
__global__ void gen_world(curandStateXORWOW_t *rand_state, hitable **world, hitable **list);

/* ################################### 光线投射/全局渲染 ################################### */
// __host__ void render(int block_size_width, int block_size_height, curandStateXORWOW_t *rand_states, hitable **world_device, vec3 *frame_buffer_host);
__device__ ray get_ray_device(float s, float t, curandStateXORWOW *rand_state);
__global__ void cuda_shading_unit(vec3 *frame_buffer, hitable **world, curandStateXORWOW_t *rand_state);
__device__ vec3 shading_pixel(int depth, const ray &r, hitable **world, curandStateXORWOW_t *rand_state);

// 留给 main 函数的接口
extern "C" __host__ void init_and_render(void);

#endif
