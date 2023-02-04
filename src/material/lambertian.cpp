#include "material/lambertian.h"

//  材质中的scatter函数用于决定二次射线如何生成，这是根据材料属性变化而变化的
bool lambertian::scatter(const ray &r_in, const hit_record &rec, vec3 &attenuated, ray &scattered) const
{
	// std::cout << "rec.u = " << rec.u << "; "
	// 		  << "rec.v = " << rec.v << "; "
	// 		  << std::endl;
	vec3 target = rec.p + rec.normal + random_in_unit_sphere(); // 获得本次打击后得到的下一个目标点
	scattered = ray(rec.p, target - rec.p, r_in.time());		// 本次击中一个目标点后的下一个射线（获得散射光线）
	// std::cout << "this" << std::endl;
	attenuated = albedo->value(rec.u, rec.v, rec.p);

	// return true;
	return (dot(scattered.direction(), rec.normal) > 0);
}

vec3 lambertian::computeBRDF(ray r_in, ray r_out, vec3 normal)
{
	return vec3(0,0,0);
}
