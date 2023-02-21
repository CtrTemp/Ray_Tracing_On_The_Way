#include "lambertian.h"

//  材质中的scatter函数用于决定二次射线如何生成，这是根据材料属性变化而变化的
bool lambertian::scatter(const ray &r_in, const hit_record &rec, Vector3f &attenuated, ray &scattered) const
{
	Vector3f target = rec.p + rec.normal + random_in_unit_sphere(); // 获得本次打击后得到的下一个目标点

	scattered = ray(rec.p, target - rec.p, r_in.time()); // 本次击中一个目标点后的下一个射线（获得散射光线）
	attenuated = albedo->value(rec.u, rec.v, rec.p);

	return (scattered.direction().dot(rec.normal) > 0);
}

float lambertian::pdf(Vector3f r_in_dir, Vector3f r_out_dir, Vector3f normal)
{
	if (r_out_dir.dot(normal) > 0.0f)
	{
		return 0.5f / M_PI;
	}
	else
	{
		return 0.0f;
	}
}


// wi是射线指向着色点的方向向量，wo是着色点指向采样光源的方向
Vector3f lambertian::computeBRDF(const Vector3f light_in_dir_wi, const Vector3f light_in_dir_wo, const hit_record p)
{
	float cosalpha = p.normal.dot(-light_in_dir_wi);
	if (cosalpha > 0.0f)
	{
		Vector3f diffuse = this->albedo->value(p.u, p.v, p.p) / M_PI;
		return diffuse;
	}
	return Vector3f(0, 0, 0);
}
