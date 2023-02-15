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

	// // 个人写法
	// float k_val = -dot(unit_vector(wi), shade_point_normal);
	// vec3 mirror_wi_own = k_val * shade_point_normal * 2 + unit_vector(wi);
	// // 参考写法
	// // 这里有一个bug，为什么必须使用unit_vector??
	// vec3 mirror_wi_ref = unit_vector(wi) - 2 * dot(unit_vector(wi), shade_point_normal) * shade_point_normal;

	// 这个值越大说明入射角和实际出射角之间的距离差越小，可以接受更多光能
	float compute_fuzz = dot(mirror_wo, light_in_dir_wi);

	// fuzz 应该用单位向量的终端圆半径来定义？？
	vec3 ret_color = this->albedo * compute_fuzz;

	// vec3 test_vec = mirror_wi + wi;
	// float test_val = dot(test_vec, shade_point_normal);

	// float wi_vec_len = wi.length();
	// float wo_vec_len = wo.length();
	// float mirror_wi_vec_len = mirror_wi.length();

	// 给到这个值大于1，应该就不会有直接光源返回！
	if (compute_fuzz <= 1 - this->fuzz)
	{
		return vec3(0, 0, 0);
	}

	return ret_color;
};
