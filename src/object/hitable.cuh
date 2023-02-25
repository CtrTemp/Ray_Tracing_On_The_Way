#pragma once
#ifndef HITABLE_H
#define HITABLE_H

#include "utils/ray.cuh"
#include "material/material.cuh"
#include "math/device_rand.cuh"

__forceinline__ __device__ float ffmin(float a, float b) { return a < b ? a : b; }
__forceinline__ __device__ float ffmax(float a, float b) { return a > b ? a : b; }

//这个结构体用来记录那些与物体相交的射线的一些参数
struct hit_record
{
	float t;	 //用于记录击中点时，此时射线方向向量的乘数
	vec3 p;		 //用于记录击中点坐标
	vec3 normal; //用于记录击中点处的法线（注意是单位向量）

	// 这里是 uv 贴图坐标
	float u;
	float v; 

	material *mat_ptr; // 当前击中点的材质
	bool happened;	   // 是否发生打击
};

class hitable
{
public:

	__device__ hitable() = default;

	__device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const = 0;


	__device__ virtual bool hasEmission(void) const = 0;

};

#endif
