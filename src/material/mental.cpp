#include "mental.h"

bool mental::scatter(const ray &r_in, const hit_record &rec, vec3 &attenuated, ray &scattered) const
{
	vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal); // 获得反射方向向量
	scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere());	 // 获得散射方向向量
	// fuzz参数代表金属表面的“哑光”程度，即我们使用scattered散射参数来间接的代表reflect参数，
	// 可见fuzz参数越大，scattered偏离reflect光线的程度，或者说不确定度就越大，也就表现出一种
	// 光线“随机反射”的哑光、漫反射特性
	attenuated = albedo;
	
	return (dot(scattered.direction(), rec.normal) > 0);
	// 返回false说明散射光线与原法线成钝角，直观上理解：此时散射光线射入了object内部，即为光线被吸收
}

float mental::pdf(vec3 r_in_dir, vec3 r_out_dir, vec3 normal)
{
	if (dot(r_out_dir, normal) > 0.0f)
	{
		return 1.0f / (2 * M_PI * 0.2);
	}
	else
	{
		return 0.0f;
	}
}

// wi是射线指向着色点的方向向量，wo是着色点指向采样光源的方向
vec3 mental::computeBRDF(const vec3 light_in_dir_wi, const vec3 light_in_dir_wo, const hit_record p)
{

	float cosalpha = dot(p.normal, -light_in_dir_wi);
	if (cosalpha <= 0.0f)
	{
		return vec3(0, 0, 0);
	}

	vec3 shade_point_coord = p.p;
	vec3 shade_point_normal = p.normal;
	vec3 mirror_wo = reflect(unit_vector(light_in_dir_wo), shade_point_normal);
	// 这里要计算精确反射向量

	// 这个值越大说明入射角和实际出射角之间的距离差越小，可以接受更多光能
	float compute_fuzz = dot(mirror_wo, light_in_dir_wi);

	vec3 ret_color = this->albedo * compute_fuzz;

	// 给到这个值大于1，应该就不会有直接光源返回！
	if (compute_fuzz <= 1 - this->fuzz)
	{
		return vec3(0, 0, 0);
	}

	return ret_color;
};
