#pragma once
#ifndef PERLIN_NOISE
#define PERLIN_NOISE

#include "math/device_rand.cuh"
#include "texture/textures.cuh"
#include "math/device_rand.cuh"
#include "math/common_math_device.cuh"
#include <cmath>
#include <random>

__device__ static float *perlin_generate(curandStateXORWOW *rand_state);
__device__ static vec3 *perlin_generate_vec(curandStateXORWOW *rand_state);
__device__ static int *perlin_generate_perm(curandStateXORWOW *rand_state);

__device__ inline float trelinear_interp(float c[2][2][2], float u, float v, float w);
__device__ inline float perlin_interp(vec3 c[2][2][2], float u, float v, float w);

class perlin
{
public:
	__device__ perlin() = default;
	__device__ perlin(curandStateXORWOW *rand_state)
	{
		ranfloat = perlin_generate(rand_state);

		ranvec = perlin_generate_vec(rand_state);

		perm_x = perlin_generate_perm(rand_state);
		perm_y = perlin_generate_perm(rand_state);
		perm_z = perlin_generate_perm(rand_state);
	};

	__device__ inline float noise(const vec3 &p) const
	{

		float u = p.x() - floor(p.x());
		float v = p.y() - floor(p.y());
		float w = p.z() - floor(p.z());

		int i = floor(p.x());
		int j = floor(p.y());
		int k = floor(p.z());

		// 注意这里模拟一个三维数组
		vec3 c[2][2][2];
		// vec3 c[] = {vec3(1,1,1),vec3(1,1,1),vec3(1,1,1)};
		// vec3 *a = new vec3();

		for (int page = 0; page < 2; ++page)
			for (int row = 0; row < 2; ++row)
				for (int col = 0; col < 2; ++col)
				{
					// int global_index = page * 2 * 2 + row * 2 + col;
					vec3 assign_vec = ranvec[perm_x[(i + page) & 255] ^ perm_y[(j + row) & 255] ^ perm_z[(k + col) & 255]];
					// ***(c + global_index) = vec3(0, 0, 0);
					c[page][row][col] = assign_vec;
				}
		// float ano = perlin_interp(c, u, v, w);
		// return a;

		return 1.85;
	}

public:
	float *ranfloat;

	vec3 *ranvec;

	int *perm_x;
	int *perm_y;
	int *perm_z;
};

// 普通白噪声
class noise_texture : public textures
{
public:
	__device__ noise_texture() = default;
	__device__ noise_texture(float sc, curandStateXORWOW *rand_state)
	{
		scale = sc;
		noise = new perlin(rand_state);
	}
	__device__ virtual vec3 value(float u, float v, const vec3 &p) const
	{
		// printf("perlin = %f", noise->noise(scale * p));
		return vec3(1, 1, 1) * noise->noise(scale * p);
		// return vec3(1, 1, 1) * (*noise->ranfloat);
	}

	perlin *noise;
	float scale;
};

__device__ inline float trelinear_interp(float c[2][2][2], float u, float v, float w)
{
	float accum = 0;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j)
			for (int k = 0; k < 2; ++k)
				accum += (i * u + (1 - i) * (1 - u)) *
						 (j * v + (1 - j) * (1 - v)) *
						 (k * w + (1 - k) * (1 - w)) * c[i][j][k];

	return accum;
}

__device__ inline float perlin_interp(vec3 c[2][2][2], float u, float v, float w)
{
	float uu = u * u * (3 - 2 * u);
	float vv = v * v * (3 - 2 * v);
	float ww = w * w * (3 - 2 * w);

	// float accum = 0;
	// for (int i = 0; i < 2; ++i)
	// 	for (int j = 0; j < 2; ++j)
	// 		for (int k = 0; k < 2; ++k)
	// 		{
	// 			// vec3 *weight_v = new vec3();
	// 			// weight_v[0] = 0;
	// 			// vec3 weight_v(u - i, v - j, w - k);
	// 			// accum += (i * uu + (1 - i) * (1 - uu)) *
	// 			// 		 (j * vv + (1 - j) * (1 - vv)) *
	// 			// 		 (k * ww + (1 - k) * (1 - ww)) *
	// 			// 		 dot(c[i][j][k], c[i][j][k]);
	// 		}

	return 0.35;
	// return fabs(accum);
}

__device__ static float *perlin_generate(curandStateXORWOW *rand_state)
{
	float *p = new float[256];
	for (int i = 0; i < 256; ++i)
		p[i] = random_float_device(rand_state);
	return p;
}
__device__ static vec3 *perlin_generate_vec(curandStateXORWOW *rand_state)
{
	vec3 *p = new vec3[256];
	for (int i = 0; i < 256; ++i)
		p[i] = unit_vector(vec3((2 * random_float_device(rand_state) - 1), (2 * random_float_device(rand_state) - 1), (2 * random_float_device(rand_state) - 1)));
	return p;
}

__device__ static void permute(int *p, int n, curandStateXORWOW *rand_state)
{
	for (int i = n - 1; i > 0; --i)
	{
		int target = int(random_float_device(rand_state) * (i + 1));
		int tmp = p[i];
		p[i] = p[target];
		p[target] = tmp;
	}
	return; // 结尾应该会隐式return吧，这个应该没必要
}

__device__ static int *perlin_generate_perm(curandStateXORWOW *rand_state)
{
	int *p = new int[256];
	for (int i = 0; i < 256; ++i)
		p[i] = i;
	permute(p, 256, rand_state);
	return p;
}

// __device__ float perlin::noise(const vec3 &p) const
// {
// 	float u = p.x() - floor(p.x());
// 	float v = p.y() - floor(p.y());
// 	float w = p.z() - floor(p.z());

// 	int i = floor(p.x());
// 	int j = floor(p.y());
// 	int k = floor(p.z());

// 	// float c[2][2][2];
// 	vec3 c[2][2][2];

// 	for (int di = 0; di < 2; ++di)
// 		for (int dj = 0; dj < 2; ++dj)
// 			for (int dk = 0; dk < 2; ++dk)
// 				// c[di][dj][dk] = ranfloat[perm_x[(i + di) & 255] ^ perm_y[(j + dj) & 255] ^ perm_z[(k + dk) & 255]];
// 				c[di][dj][dk] = ranvec[perm_x[(i + di) & 255] ^ perm_y[(j + dj) & 255] ^ perm_z[(k + dk) & 255]];

// 	return perlin_interp(c, u, v, w);
// }

// 普通白噪声
// __device__ vec3 noise_texture::value(float u, float v, const vec3 &p) const
// {
// 	return vec3(1, 1, 1) * noise.noise(scale * p);
// }

// // 通过将这几句话加入，初始化perlin类内的公有变量，使其在构造perlin噪声时可以直接被使用
// float *perlin::ranfloat = perlin_generate();
// vec3 *perlin::ranvec = perlin_generate_vec();

// int *perlin::perm_x = perlin_generate_perm();
// int *perlin::perm_y = perlin_generate_perm();
// int *perlin::perm_z = perlin_generate_perm();

#endif
