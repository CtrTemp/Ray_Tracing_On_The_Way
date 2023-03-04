#ifndef RENDER_H
#define RENDER_H
#include "camera/camera.cuh"

__host__ curandStateXORWOW_t *init_rand(int block_size_width, int block_size_height);

__host__ void init_camera(void);

__host__ hitable **init_world(curandStateXORWOW_t *rand_states);

__host__ void render(int block_size_width, int block_size_height, curandStateXORWOW_t *rand_states, hitable **world_device);

#endif