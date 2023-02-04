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
	// 散射参数(二次/间接光源参数): 即给出入射光线参数, 打击点参数, 衰减参数, 针对当前的材料, 给出一条出射光
	virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuated, ray &scattered) const = 0;
	// 发光参数(一次/直接光源参数): 如果是发光体材质, 还应具有发光参数, 传入
	virtual vec3 emitted(float u, float v, const vec3 &p) const = 0;
	virtual bool hasEmission(void) const = 0;

	// 计算BRDF是重中之重，这是主要的添加改进，每个不同的material根据不同的光照情况计算BRDF
	// BRDF是反照率/反射系数/吸收系数的综合考量
	// 2023/01/13 加入 BRDF 应有的参数
	virtual vec3 computeBRDF(ray r_in, ray r_out, vec3 normal) = 0;
};

#endif
