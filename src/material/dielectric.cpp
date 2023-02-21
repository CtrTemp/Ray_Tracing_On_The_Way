#include "dielectric.h"

bool dielectric::scatter(const ray &r_in, const hit_record &rec, Vector3f &attenuation, ray &scattered) const
{
	Vector3f outward_normal;
	Vector3f reflected = reflect(r_in.direction(), rec.normal);
	float ni_over_nt;
	// 纯透明玻璃体，只传递/分配光能，不会附加衰减（染色）
	attenuation = Vector3f(1.0, 1.0, 1.0);
	Vector3f refracted;

	float reflect_prob;
	float cosine;

	// 以下判断成立说明是从晶体内部射向空气，需要考虑是否发生了全反射
	if (r_in.direction().dot(rec.normal) > 0)
	{
		// 我们将法线方向定义为由外部指向内部
		outward_normal = -rec.normal;
		// ref_idx是大于1的相对折射率，ni_over_nt是入射端折射率除以出射端折射率
		// 当前情况也可以将这个值理解为 出射角正弦值（较大）除以入射角正弦值（较小）
		ni_over_nt = ref_idx;
		cosine = ref_idx * r_in.direction().dot(rec.normal / r_in.direction().norm());
		//
	}
	// 否则为从空气射向晶体球内部，这个时候不可能发生全反射情况
	else
	{
		outward_normal = rec.normal; // 我们将法线方向定义为由内部指向外部
		ni_over_nt = 1.0 / ref_idx;
		cosine = -r_in.direction().dot(rec.normal / r_in.direction().norm());
	}
	// 如果不发生全反射现象
	if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
	{
		reflect_prob = schlick(cosine, ref_idx); // 应该是由菲涅尔公式近似计算出的反射光线强度
	}											 // 其实是（转化成）反射光占总光线之比，在抗锯齿章节我们将一个像素点由多（100）条射线表示
	else
	{
		reflect_prob = 1.0; // 如果全反射，则反射光占比为100%
	}

	// 在发生折射的情况下，每次也只生成一条光线，要么折射光，要么反射光，二者占比满足折射/反射的光能分配
	// reflect_prob 的值在 0-1 之间，表示反射光的光能占比
	if (drand48() < reflect_prob)
	{
		scattered = ray(rec.p, reflected);
	} // 明白
	else
	{
		scattered = ray(rec.p, refracted);
	} // 明白


	return true;
}

// 因为对于这种可以透射的表面上下半球都可以采样，所以这里直接返回单位半球面积，不用考虑上下半球
float dielectric::pdf(Vector3f r_in_dir, Vector3f r_out_dir, Vector3f normal)
{
	return 1.0f / (2 * M_PI * 0.2);
}



Vector3f dielectric::computeBRDF(const Vector3f light_in_dir_wi, const Vector3f light_in_dir_wo, const hit_record p)
{



	Vector3f shade_point_coord = p.p;		// 着色点空间坐标
	Vector3f shade_point_normal = p.normal; // 着色点表面法向量

	Vector3f mirror_reflect_wi(0, 0, 0); // 理想状况下光源光线反射出的方向
	Vector3f mirror_refract_wi(0, 0, 0); // 理想状况下光源光线折射出的方向

	float compute_fuzz_refract = 0; // 折射光衰减
	float compute_fuzz_reflect = 0; // 反射光衰减

	Vector3f outward_normal; // 与入射光呈钝角的法向量方向（折射光计算使用）
	float ni_over_nt;	 // 入射光所在材质相对折射出射光材质的相对折射率（折射光计算使用）
	float reflect_prob;	 // 反射光能占比
	float cosine;		 // 计算系数（计算反射光占比使用）

	Vector3f ret_color_refract(0, 0, 0); // 返回的折射光
	Vector3f ret_color_reflect(0, 0, 0); // 返回的反射光

	// 大于零：光源在晶体表面内侧； 小于零：光源在晶体表面外侧
	const float surface_lightSource_direction = light_in_dir_wi.dot(shade_point_normal);
	// 大于零：观测点在晶体表面外侧； 小于零：观测点在晶体表面内侧
	const float surface_viewPoint_direction = light_in_dir_wo.dot(shade_point_normal);

	// 第一步先计算反射光，看当前入射方向的光源可否可能是反射光

	// 以上二系数乘积小于零，说明入射与出射光线在晶体表面同侧，可能发生反射情况，不可能是折射情况
	if (surface_lightSource_direction * surface_viewPoint_direction < 0.0f)
	{
		// return Vector3f(0, 0, 0);
		// if (surface_lightSource_direction > 0)
		// {
		// 	outward_normal = -shade_point_normal;
		// }
		mirror_reflect_wi = reflect(light_in_dir_wi, shade_point_normal);
		compute_fuzz_reflect = mirror_reflect_wi.dot(light_in_dir_wo);
		if (compute_fuzz_reflect <= 0.995)
		{
			return Vector3f(0, 0, 0);
		}
		ret_color_reflect = compute_fuzz_reflect * Vector3f(1, 1, 1);
		return ret_color_reflect;
	}
	// 第二步计算折射光，
	// 这种情况下入射光与观测点在晶体表面异侧
	else
	{
		// return Vector3f(0, 0, 0);
		// 光源在内侧，观测点在外侧
		if (surface_lightSource_direction > 0)
		{
			outward_normal = -shade_point_normal;
			ni_over_nt = ref_idx;
			cosine = ref_idx * light_in_dir_wi.dot(shade_point_normal / light_in_dir_wi.norm());
		}
		// 光源在外侧，观测点在内侧
		else
		{
			outward_normal = shade_point_normal;
			ni_over_nt = 1 / ref_idx;
			cosine = -light_in_dir_wi.dot(shade_point_normal / light_in_dir_wi.norm());
		}

		if (refract(light_in_dir_wi.normalized(), outward_normal, ni_over_nt, mirror_refract_wi))
		{
			compute_fuzz_refract = mirror_refract_wi.dot(light_in_dir_wo);

			if (compute_fuzz_refract <= 0.995)
			{
				return Vector3f(0, 0, 0);
			}
			if (compute_fuzz_refract > 1)
			{
				// std::cout << "sss" << std::endl;
				compute_fuzz_refract = 0.999;
			}
			Vector3f ret_color = Vector3f(1, 1, 1) * compute_fuzz_refract;
			// reflect_prob = schlick(cosine, ref_idx);
			// std::cout << "reflect_prob = " << reflect_prob << std::endl;
			return ret_color;
		}
		else
		{
			return Vector3f(0, 0, 0);
		}
	}
	return Vector3f(0, 0, 0);
}
