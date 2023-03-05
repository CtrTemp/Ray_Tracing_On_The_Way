#pragma once
#ifndef HITABLELIST_H
#define HITABLELIST_H

#include "object/hitable.cuh"

#include <memory>

class sphere;

class hitable_list : public hitable
{
public:
	__device__ hitable_list() = default;

	// __device__ hitable_list(curandStateXORWOW_t *rand_state);

	__device__ hitable_list(hitable **d_list, int size)
	{
		list = d_list;
		list_size = size;
	}

	__device__ virtual bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const
	{
		hit_record temp_rec;
		bool hit_anything = false;
		double closest_so_far = tmax;
		// printf("hitable world size = %d", this->list_size);

		// printf("tmin = %f, closest_so_far=%f   ", tmin, closest_so_far);

		for (int i = 0; i < list_size; ++i)
		{
			if (list[i]->hit(r, tmin, closest_so_far, temp_rec))
			{
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}
		return hit_anything;
	}

	__device__ virtual bool hasEmission(void) const
	{
		return false;
	}

public:
	hitable **list;
	int list_size;
};

// extern "C" __global__ void gen_world(curandStateXORWOW_t *rand_state, hitable **world, hitable **list);

// __device__ bool hitable_list::hit(const ray &r, float tmin, float tmax, hit_record &rec) const

// __device__ bool hitable_list::hasEmission(void) const

// __device__ hitable_list::hitable_list(curandStateXORWOW_t *rand_state)
// {
// 	list_size = 1;
// 	list = new hitable *[list_size];
// 	int index = 0;
// 	material *red_mat_ptr = new lambertian(vec3(0.8, 0.1, 0.1));
// 	list[index++] = new sphere(vec3(0, -1000, 0), 1000, red_mat_ptr);
// }
#endif
