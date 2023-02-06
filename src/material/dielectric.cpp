#include "material/dielectric.h"



bool dielectric::scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered) const
{
	vec3 outward_normal;
	vec3 reflected = reflect(r_in.direction(), rec.normal);
	float ni_over_nt;
	attenuation = vec3(1.0, 1.0, 1.0);
	vec3 refracted;

	float reflect_prob;
	float cosine;

	if (dot(r_in.direction(), rec.normal) > 0) //如果是从晶体内部射向空气
	{
		outward_normal = -rec.normal; //我们将法线方向定义为由外部指向内部
		ni_over_nt = ref_idx;		  // ref_idx是大于1的相对折射率，ni_over_nt是入射端折射率除以出射端折射率
		cosine = ref_idx * dot(r_in.direction(), rec.normal / r_in.direction().length());
	}
	else //如果是从空气射向晶体球内部
	{
		outward_normal = rec.normal; //我们将法线方向定义为由内部指向外部
		ni_over_nt = 1.0 / ref_idx;
		cosine = -dot(r_in.direction(), rec.normal / r_in.direction().length());
	}
	if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) //如果发生折射（无全反射）
	{
		reflect_prob = schlick(cosine, ref_idx); //应该是由菲涅尔公式近似计算出的反射光线强度
	}											 //其实是（转化成）反射光占总光线之比，在抗锯齿章节我们将一个像素点由多（100）条射线表示
	else
	{
		reflect_prob = 1.0; //如果全反射，则反射光占比为100%
	}


	// generate only one piece of ray: reflect or refract
	if (drand48() < reflect_prob)
	{
		scattered = ray(rec.p, reflected);
	} //明白
	else
	{
		scattered = ray(rec.p, refracted);
	} //明白

	return true;
}

vec3 dielectric::computeBRDF(const vec3 wi, const vec3 wo, const hit_record p)
{
	return vec3(0,0,0);
}
