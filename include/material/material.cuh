#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H

struct hit_record;

#include "utils/ray.cuh"
#include "math/device_rand.cuh"
// #include "texture/textures.cuh"

// __device__ vec3 reflect(const vec3 &v, const vec3 &n);
// __device__ bool refract(const vec3 &v, const vec3 &n, float ni_over_nt, vec3 &refracted);
// __device__ float schlick(float cosine, float ref_idx);

// 基类material有两个函数
class material
{

public:
	// 散射参数(二次/间接光源参数): 即给出入射光线参数, 打击点参数, 衰减参数, 针对当前的材料, 给出一条出射光
	__device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandStateXORWOW_t *rand_state) const = 0;
	// 发光参数(一次/直接光源参数): 如果是发光体材质, 还应具有发光参数, 传入
	__device__ virtual vec3 emitted(float u, float v, const vec3 &p) const = 0;
	__device__ virtual bool hasEmission(void) const = 0;
};

static __device__ vec3 reflect(const vec3 &v, const vec3 &n)
{
	return v - 2 * dot(v, n) * n; // 获得纯镜面反射光线（方向射线）
}

// 折射光，返回是否可以发生折射（是否不发生全反射）
// 并通过传引用，隐式的计算出折射光线
static __device__ bool refract(const vec3 &v, const vec3 &n, float ni_over_nt, vec3 &refracted)
{
	vec3 uv = v / v.length(); // unit_vector(v);（这个函数为什么用不了？？） //单位入射光线
	float dt = dot(uv, n);	  // 单位入射光线在法线（我们默认是从球心指向外部）上的投影就是入射角的cos值
	float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
	// 全反射判断公式，可以自推，其中ni_over_nt是相对折射率，可知小于1时一定是不会发生全反射的
	// （因为是从折射率较小的物质射入折射率较大的物质一定不会发生全反射，这里或许可以优化加速）
	if (discriminant > 0)
	{
		refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
		return true;
	}
	else
		return false;
}

static __device__ float schlick(float cosine, float ref_idx)
{
	float r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 = r0 * r0;
	return r0 + (1 - r0) * pow((1 - cosine), 5);
}

// 我应该怎么简写这个枚举类型
// typedef enum material::SelfMaterialType::DIELECTRIC DIELECTRIC;

#endif
