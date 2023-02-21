#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H

struct hit_record;

#include "utils/ray.h"
#include "math/random.h"
#include "texture/textures.h"

Vector3f reflect(const Vector3f &v, const Vector3f &n);
bool refract(const Vector3f &v, const Vector3f &n, float ni_over_nt, Vector3f &refracted);
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
	virtual bool scatter(const ray &r_in, const hit_record &rec, Vector3f &attenuated, ray &scattered) const = 0;
	// 发光参数(一次/直接光源参数): 如果是发光体材质, 还应具有发光参数, 传入
	virtual Vector3f emitted(float u, float v, const Vector3f &p) const = 0;
	virtual bool hasEmission(void) const = 0;

	// 计算BRDF是重中之重，这是主要的添加改进，每个不同的material根据不同的光照情况计算BRDF
	// BRDF是反照率/反射系数/吸收系数的综合考量
	// 2023/01/13 加入 BRDF 应有的参数
	virtual Vector3f computeBRDF(const Vector3f light_in_dir_wi, const Vector3f light_out_dir_wo, const hit_record p) = 0;

	virtual float pdf(Vector3f r_in_dir, Vector3f r_out_dir, Vector3f normal) = 0;
	virtual SelfMaterialType getMaterialType() = 0;
};



// 我应该怎么简写这个枚举类型
// typedef enum material::SelfMaterialType::DIELECTRIC DIELECTRIC;

#endif
