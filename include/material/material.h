#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H

struct hit_record;

// #include "utils/vec3.h"
// #include "object/hitable.h"
#include "utils/ray.h"
#include "math/random.h"
#include "texture/textures.h"

vec3 reflect(const vec3 &v, const vec3 &n);
bool refract(const vec3 &v, const vec3 &n, float ni_over_nt, vec3 &refracted);
float schlick(float cosine, float ref_idx);

// 基类material有两个函数
class material
{
public:
	enum class SelfMaterialType
	{
		LAMBERTAIN,
		MENTAL,
		DIELECTRIC,
		LIGHT
	};

public:
	// 散射参数(二次/间接光源参数): 即给出入射光线参数, 打击点参数, 衰减参数, 针对当前的材料, 给出一条出射光
	virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuated, ray &scattered) const = 0;
	// 发光参数(一次/直接光源参数): 如果是发光体材质, 还应具有发光参数, 传入
	virtual vec3 emitted(float u, float v, const vec3 &p) const = 0;
	virtual bool hasEmission(void) const = 0;

	// 计算BRDF是重中之重，这是主要的添加改进，每个不同的material根据不同的光照情况计算BRDF
	// BRDF是反照率/反射系数/吸收系数的综合考量
	// 2023/01/13 加入 BRDF 应有的参数
	virtual vec3 computeBRDF(const vec3 light_in_dir_wi, const vec3 light_out_dir_wo, const hit_record p) = 0;

	virtual float pdf(vec3 r_in_dir, vec3 r_out_dir, vec3 normal) = 0;
	virtual SelfMaterialType getMaterialType() = 0;
	// {
	// 	if (dot(r_out_dir, normal) > 0.0f)
	// 	{
	// 		return 0.5f / M_PI;
	// 	}
	// 	else
	// 	{
	// 		// 这里不能是0.0，pdf是要作为分母的，作为0的除数会造成-inf数值错误
	// 		// 再有一个点，pdf采样应该分开考虑，diffuse和mental都只能进行上半球采样
	// 		// 而新加入的玻璃表面dielectric可以进行下半球采样（透射！！！）
	// 		return 1.0f;
	// 		// return 0.0f;
	// 	}
	// }
};

#endif
