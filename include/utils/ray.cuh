// #pragma once
// #ifndef RAY_CUH
// #define RAY_CUH
// #include "utils/vec3.cuh"

// class ray
// {
// public:
// 	__device__ ray(){};
// 	__device__ ray(const vec3 &a, const vec3 &b)
// 	{
// 		A = a;
// 		B = b;
// 		// _time = ti;
// 		// inv_dir = vec3(1. / B.x(), 1. / B.y(), 1. / B.z());
// 	}

// 	__device__ vec3 origin() const { return A; }	// 射线起始点
// 	__device__ vec3 direction() const { return B; } // 射线方向向量（应该是单位向量）
// 	// __device__ float time() const { return _time; }
// 	__device__ vec3 point_at_parameter(float t) const { return A + t * B; }
// 	// 给定射线方向向量倍数t，得到射线末端指向点

// 	vec3 A;
// 	vec3 B;
// 	// vec3 inv_dir;
// 	// float _time;
// };

// #endif

#ifndef RAYH
#define RAYH
#include "utils/vec3.cuh"

class ray
{
public:
	__device__ ray() {}
	__device__ ray(const vec3 &a, const vec3 &b)
	{
		A = a;
		B = b;
	}
	__device__ vec3 origin() const { return A; }
	__device__ vec3 direction() const { return B; }
	__device__ vec3 point_at_parameter(float t) const { return A + t * B; }

	vec3 A;
	vec3 B;
};

#endif
