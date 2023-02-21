#include "mental.h"

bool mental::scatter(const ray &r_in, const hit_record &rec, Vector3f &attenuated, ray &scattered) const
{
	Vector3f reflected = reflect(r_in.direction().normalized(), rec.normal); // 获得反射方向向量
	scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere());		 // 获得散射方向向量
	// fuzz参数代表金属表面的“哑光”程度，即我们使用scattered散射参数来间接的代表reflect参数，
	// 可见fuzz参数越大，scattered偏离reflect光线的程度，或者说不确定度就越大，也就表现出一种
	// 光线“随机反射”的哑光、漫反射特性
	attenuated = albedo;

	return (scattered.direction().dot(rec.normal) > 0);
	// 返回false说明散射光线与原法线成钝角，直观上理解：此时散射光线射入了object内部，即为光线被吸收
}

float mental::pdf(Vector3f r_in_dir, Vector3f r_out_dir, Vector3f normal)
{
	if (r_out_dir.dot(normal) > 0.0f)
	{
		return 1.0f / (2 * M_PI * 0.2);
	}
	else
	{
		return 0.0f;
	}
}

// wi是射线指向着色点的方向向量，wo是着色点指向采样光源的方向
Vector3f mental::computeBRDF(const Vector3f light_in_dir_wi, const Vector3f light_in_dir_wo, const hit_record p)
{

	float cosalpha = p.normal.dot(-light_in_dir_wi);
	if (cosalpha <= 0.0f)
	{
		return Vector3f(0, 0, 0);
	}

	Vector3f shade_point_coord = p.p;
	Vector3f shade_point_normal = p.normal;
	Vector3f mirror_wo = reflect(light_in_dir_wo.normalized(), shade_point_normal);
	// 这里要计算精确反射向量

	// 这个值越大说明入射角和实际出射角之间的距离差越小，可以接受更多光能
	float compute_fuzz = mirror_wo.dot(light_in_dir_wi);

	Vector3f ret_color = this->albedo * compute_fuzz;

	// 给到这个值大于1，应该就不会有直接光源返回！
	if (compute_fuzz <= 1 - this->fuzz)
	{
		return Vector3f(0, 0, 0);
	}

	return ret_color;
};
