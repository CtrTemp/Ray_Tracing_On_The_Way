#pragma once
#ifndef HITABLELIST_H
#define HITABLELIST_H

#include "object/hitable.cuh"
#include "material/lambertian.cuh"
#include "object/geometry/sphere.cuh"

#include <memory>

class sphere;

class hitable_list : public hitable
{
public:
	__device__ hitable_list() = default;

	__device__ hitable_list(curandStateXORWOW_t *rand_state);

	__device__ hitable_list(hitable **d_list, int size)
	{
		// list = new hitable *[size];
		*list = (hitable *)malloc(size);
		for (int i = 0; i < size; i++)
		{
			list[i] = d_list[i];
		}
		list_size = size;
	}

	__device__ virtual bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const;

	__device__ virtual bool hasEmission(void) const { return false; };

public:
	hitable **list;
	int list_size;
};

extern "C" __global__ void gen_world(curandStateXORWOW_t *rand_state, hitable **world, hitable **list);

#endif
