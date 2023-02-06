#include "material/mental.h"

bool mental::scatter(const ray &r_in, const hit_record &rec, vec3 &attenuated, ray &scattered) const
{
	vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal); // 获得反射方向向量
	scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere());	 // 获得散射方向向量
	// fuzz参数代表金属表面的“哑光”程度，即我们使用scattered散射参数来间接的代表reflect参数，
	// 可见fuzz参数越大，scattered偏离reflect光线的程度，或者说不确定度就越大，也就表现出一种
	// 光线“随机反射”的哑光、漫反射特性
	// 注意这里scattered的计算方法与lambertian的不同，这里使用reflect为基，加上一个随机方向的向量得到散射方向
	// 而之前lambertian的scattered的计算是使用normal为基进行计算
	// 故在函数末端返回时，这个scattered与normal所成角度是有一定几率为钝角的，此时我们返回false
	attenuated = albedo;
	// return true;
	return (dot(scattered.direction(), rec.normal) > 0);
	// 返回false说明散射光线与原法线成钝角，直观上理解：此时散射光线射入了object内部，即为光线被吸收
}

vec3 mental::computeBRDF(const vec3 wi, const vec3 wo, const hit_record p)
{
	return vec3(0, 0, 0);
};
