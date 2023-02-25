#pragma once
#ifndef SPHERE_H
#define SPHERE_H

#include "object/hitable.cuh"
#include "math/device_rand.cuh"

// sphere球体是可打击类hitable的继承
// 需要重写hit函数和boundingbox函数
class sphere : public hitable
{ // 是sphere类，构造函数返回hitable*类型
public:
	__device__ __host__ sphere() = default;
	__device__ __host__ sphere(vec3 cen, float r, material *mat) : center(cen), radius(r), mat_ptr(mat)
	{
		area = 4 * M_PI * radius * radius;
	}; 
	__device__ virtual bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const;
	
	__device__ virtual bool hasEmission(void) const { return mat_ptr->hasEmission(); };

	vec3 center;
	float radius;
	float area;
	material *mat_ptr;
};


#endif