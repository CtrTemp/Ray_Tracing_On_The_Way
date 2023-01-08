#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H

struct hit_record;

#include "utils/vec3.h"
#include "object/hitable.h"
#include "math/random.h"
#include "texture/textures.h"


vec3 reflect(const vec3& v, const vec3& n);
bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted);
float schlick(float cosine, float ref_idx);


// 基类material有两个函数
class material {
public:
	// 散射参数(二次/间接光源参数): 即给出入射光线参数, 打击点参数, 衰减参数, 针对当前的材料, 给出一条出射光
	virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuated, ray& scattered) const = 0;
	// 发光参数(一次/直接光源参数): 如果是发光体材质, 还应具有发光参数, 传入
	virtual vec3 emitted(float u, float v, const vec3 &p) const { return vec3(0, 0, 0); }
};





/******************************** 哑光类/漫反射表面 *********************************/

class lambertian :public material {
public:
	lambertian(texture *a) :albedo(a) {}
	virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuated, ray& scattered) const;

	//vec3 albedo;
	texture *albedo;
};

/******************************** 金属类/镜面 *********************************/
class mental :public material {
public:
	mental(const vec3& a, float f) : albedo(a) { if (f < 1)fuzz = f; else fuzz = 1; }

	virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuated, ray& scattered) const;

	vec3 albedo;
	float fuzz;
};



/******************************** 玻璃类/透射表面 *********************************/
class dielectric :public material {
public:
	dielectric(float ri) : ref_idx(ri) {}
	virtual bool scatter(const ray& r_in, const hit_record& rec, vec3&attenuation, ray& scattered) const;

	float ref_idx;
};

#endif


/*


class dielectric :public material {
public:
dielectric(float ri) : ref_idx(ri) {}
virtual bool scatter(const ray& r_in, const hit_record& rec, vec3&attenuation, ray& scattered) const
{
vec3 outward_normal;
vec3 reflected = reflect(r_in.direction(), rec.normal);
float ni_over_nt;
attenuation = vec3(1.0, 1.0, 0.0);
vec3 refracted;
if (dot(r_in.direction(), rec.normal) > 0)
{
outward_normal = -rec.normal;
ni_over_nt = ref_idx;
}
else
{
outward_normal = rec.normal;
ni_over_nt = 1.0 / ref_idx;
}
if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
{
scattered = ray(rec.p, refracted);
}
else
{
scattered = ray(rec.p, reflected);
return false;
}
return true;

}



float ref_idx;
};

*/