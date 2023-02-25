#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H

struct hit_record;

#include "utils/ray.cuh"
#include "math/device_rand.cuh"
// #include "texture/textures.cuh"

__device__ vec3 reflect(const vec3 &v, const vec3 &n);
__device__ bool refract(const vec3 &v, const vec3 &n, float ni_over_nt, vec3 &refracted);
__device__ float schlick(float cosine, float ref_idx);



// 基类material有两个函数
class material
{

public:
	// 散射参数(二次/间接光源参数): 即给出入射光线参数, 打击点参数, 衰减参数, 针对当前的材料, 给出一条出射光
	__device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuated, ray &scattered, curandStateXORWOW_t *rand_state) const = 0;
	// 发光参数(一次/直接光源参数): 如果是发光体材质, 还应具有发光参数, 传入
	__device__ virtual vec3 emitted(float u, float v, const vec3 &p) const = 0;
	__device__ virtual bool hasEmission(void) const = 0;

};



// 我应该怎么简写这个枚举类型
// typedef enum material::SelfMaterialType::DIELECTRIC DIELECTRIC;

#endif
