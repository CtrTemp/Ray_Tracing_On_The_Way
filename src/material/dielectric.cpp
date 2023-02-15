#include "material/dielectric.h"

bool dielectric::scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered) const
{
	vec3 outward_normal;
	vec3 reflected = reflect(r_in.direction(), rec.normal);
	float ni_over_nt;
	// 纯透明玻璃体，只传递/分配光能，不会附加衰减（染色）
	attenuation = vec3(1.0, 1.0, 1.0);
	vec3 refracted;

	float reflect_prob;
	float cosine;

	// 以下判断成立说明是从晶体内部射向空气，需要考虑是否发生了全反射
	if (dot(r_in.direction(), rec.normal) > 0)
	{
		// 我们将法线方向定义为由外部指向内部
		outward_normal = -rec.normal;
		// ref_idx是大于1的相对折射率，ni_over_nt是入射端折射率除以出射端折射率
		// 当前情况也可以将这个值理解为 出射角正弦值（较大）除以入射角正弦值（较小）
		ni_over_nt = ref_idx;
		cosine = ref_idx * dot(r_in.direction(), rec.normal / r_in.direction().length());
		//
	}
	// 否则为从空气射向晶体球内部，这个时候不可能发生全反射情况
	else
	{
		outward_normal = rec.normal; // 我们将法线方向定义为由内部指向外部
		ni_over_nt = 1.0 / ref_idx;
		cosine = -dot(r_in.direction(), rec.normal / r_in.direction().length());
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

	// 我们现在先只模仿反射行为
	// scattered = ray(rec.p, reflected);

	// 而后我们只模仿折射行为
	// scattered = ray(rec.p, refracted);

	return true;
}

// 因为对于这种可以透射的表面上下半球都可以采样，所以这里直接返回单位半球面积，不用考虑上下半球
float dielectric::pdf(vec3 r_in_dir, vec3 r_out_dir, vec3 normal)
{
	return 1.0f / (2 * M_PI * 0.2);
}

// wi是射线指向着色点的方向向量，wo是着色点指向采样光源的方向
// 对于可以透射的玻璃表面我们应该怎么考虑呢？
/**
 * 	首先我们应该再次明确一下BRDF是什么？BRDF是光线传播过程中的一个传递系数，用于描述：从某个wi方向来的光，
 * 经过当前这个表面的这个点p，向wo方向传递的能量（衰减）系数。
 *
 * 	再次强调我们在这里需要计算的重点，需要考虑的重点只有：“在当前表面的这个点，光能是如何传递的”
 *
 * 	所以对于玻璃表面这种材质也不应该有例外，我们考虑的重点同样应该是“光能如何传递”。另外对于玻璃材质还需要额外
 * 考虑的一点就是，光能在透射与反射之间的分配。
 *
 * 	这里有一个关键点你需要明确：
 *  1.如果光线是从外部射向内部，与物体交点在p点，此时我们计算的是光源向探测射线发出点的能量传递，那么这种情况
 * 下，很自然的，我们只能考虑反射光的影响
 *  2.同样，如果光线是从内部射向外部，此时的光源wo方向来的能量源只可能是二次光源，并且这个二次光源必然是之前
 * 从外部射向晶体内部的光线
 *
 * */

vec3 dielectric::computeBRDF(const vec3 light_in_dir_wi, const vec3 light_in_dir_wo, const hit_record p)
{

	// // // 如果下式成立说明采样到的光源方向根本就不在当前表面的上半球，传递能量直接归零
	// // 这个式子对于透射物体不适用
	// float cosalpha = dot(p.normal, -light_in_dir_wi);
	// if (cosalpha <= 0.0f)
	// {
	// 	return vec3(0, 0, 0);
	// }

	// vec3 shade_point_coord = p.p;
	// vec3 shade_point_normal = p.normal;
	// vec3 mirror_wo = reflect(unit_vector(light_in_dir_wo), shade_point_normal);
	// float compute_fuzz = dot(mirror_wo, light_in_dir_wi);
	// vec3 ret_color = vec3(1, 1, 1) * compute_fuzz;

	// if (compute_fuzz <= 0.95)
	// {
	// 	return vec3(0, 0, 0);
	// }
	// return ret_color;

	vec3 shade_point_coord = p.p;		// 着色点空间坐标
	vec3 shade_point_normal = p.normal; // 着色点表面法向量

	vec3 mirror_reflect_wi(0, 0, 0); // 理想状况下光源光线反射出的方向
	vec3 mirror_refract_wi(0, 0, 0); // 理想状况下光源光线折射出的方向

	float compute_fuzz_refract = 0; // 折射光衰减
	float compute_fuzz_reflect = 0; // 反射光衰减

	vec3 outward_normal; // 与入射光呈钝角的法向量方向（折射光计算使用）
	float ni_over_nt;	 // 入射光所在材质相对折射出射光材质的相对折射率（折射光计算使用）
	float reflect_prob;	 // 反射光能占比
	float cosine;		 // 计算系数（计算反射光占比使用）

	vec3 ret_color_refract(0, 0, 0); // 返回的折射光
	vec3 ret_color_reflect(0, 0, 0); // 返回的反射光

	// 大于零：光源在晶体表面内侧； 小于零：光源在晶体表面外侧
	const float surface_lightSource_direction = dot(light_in_dir_wi, shade_point_normal);
	// 大于零：观测点在晶体表面外侧； 小于零：观测点在晶体表面内侧
	const float surface_viewPoint_direction = dot(light_in_dir_wo, shade_point_normal);

	// 第一步先计算反射光，看当前入射方向的光源可否可能是反射光

	// 以上二系数乘积小于零，说明入射与出射光线在晶体表面同侧，可能发生反射情况，不可能是折射情况
	if (surface_lightSource_direction * surface_viewPoint_direction < 0.0f)
	{
		// return vec3(0, 0, 0);
		// if (surface_lightSource_direction > 0)
		// {
		// 	outward_normal = -shade_point_normal;
		// }
		mirror_reflect_wi = reflect(light_in_dir_wi, shade_point_normal);
		compute_fuzz_reflect = dot(mirror_reflect_wi, light_in_dir_wo);
		if (compute_fuzz_reflect <= 0.995)
		{
			return vec3(0, 0, 0);
		}
		ret_color_reflect = compute_fuzz_reflect * vec3(1, 1, 1);
		return ret_color_reflect;
	}
	// 第二步计算折射光，
	// 这种情况下入射光与观测点在晶体表面异侧
	else
	{
		// return vec3(0, 0, 0);
		// 光源在内侧，观测点在外侧
		if (surface_lightSource_direction > 0)
		{
			outward_normal = -shade_point_normal;
			ni_over_nt = ref_idx;
			cosine = ref_idx * dot(light_in_dir_wi, shade_point_normal / light_in_dir_wi.length());
		}
		// 光源在外侧，观测点在内侧
		else
		{
			outward_normal = shade_point_normal;
			ni_over_nt = 1 / ref_idx;
			cosine = -dot(light_in_dir_wi, shade_point_normal / light_in_dir_wi.length());
		}

		if (refract(unit_vector(light_in_dir_wi), outward_normal, ni_over_nt, mirror_refract_wi))
		{
			compute_fuzz_refract = dot(mirror_refract_wi, light_in_dir_wo);

			if (compute_fuzz_refract <= 0.995)
			{
				return vec3(0, 0, 0);
			}
			if (compute_fuzz_refract > 1)
			{
				// std::cout << "sss" << std::endl;
				compute_fuzz_refract = 0.999;
			}
			vec3 ret_color = vec3(1, 1, 1) * compute_fuzz_refract;
			// reflect_prob = schlick(cosine, ref_idx);
			// std::cout << "reflect_prob = " << reflect_prob << std::endl;
			return ret_color;
		}
		else
		{
			return vec3(0, 0, 0);
		}
	}
	return vec3(0, 0, 0);
}

// vec3 computeBRDF(int void_val, const vec3 light_in_dir_wi, const vec3 light_in_dir_wo, const hit_record p)
// {
// 	// // // 如果下式成立说明采样到的光源方向根本就不在当前表面的上半球，传递能量直接归零
// 	// // 这个式子对于透射物体不适用
// 	// float cosalpha = dot(p.normal, -light_in_dir_wi);
// 	// if (cosalpha <= 0.0f)
// 	// {
// 	// 	return vec3(0, 0, 0);
// 	// }

// 	// vec3 shade_point_coord = p.p;
// 	// vec3 shade_point_normal = p.normal;
// 	// vec3 mirror_wo = reflect(unit_vector(light_in_dir_wo), shade_point_normal);
// 	// float compute_fuzz = dot(mirror_wo, light_in_dir_wi);
// 	// vec3 ret_color = vec3(0.9, 0.9, 0.9) * compute_fuzz;

// 	// // 给到这个值大于1，应该就不会有直接光源返回！
// 	// if (compute_fuzz <= 0.90)
// 	// {
// 	// 	return vec3(0, 0, 0);
// 	// }
// 	// return ret_color;

// 	// 以上是对反射行为的仿真，下面对折射行为进行仿真 ##########################################################

// 	// 只考虑折射情况，则出射和入射光线必然分居晶体表面两侧
// 	// 如果下式成立说明采样到的光源方向根本就不在当前表面的下半球，传递能量直接归零
// 	// float cosalpha = dot(p.normal, -light_in_dir_wi);
// 	// if (cosalpha >= 0.0f)
// 	// {
// 	// 	return vec3(0, 0, 0);
// 	// }

// 	// vec3 shade_point_coord = p.p;
// 	// vec3 shade_point_normal = p.normal;
// 	// float ni_over_nt;
// 	// vec3 outward_normal;
// 	// vec3 mirror_wo;
// 	// float compute_fuzz;

// 	// // 光源在晶体内部
// 	// if (dot(light_in_dir_wi, shade_point_normal) > 0)
// 	// {
// 	// 	// if (dot(light_in_dir_wo, shade_point_normal) > 0)
// 	// 	// {
// 	// 	// 	return vec3(0, 0, 0);
// 	// 	// }
// 	// 	outward_normal = -shade_point_normal;
// 	// 	ni_over_nt = ref_idx;
// 	// }
// 	// else
// 	// {
// 	// 	// if (dot(light_in_dir_wo, shade_point_normal) < 0)
// 	// 	// {
// 	// 	// 	return vec3(0, 0, 0);
// 	// 	// }
// 	// 	outward_normal = shade_point_normal;
// 	// 	ni_over_nt = 1 / ref_idx;
// 	// }

// 	// if (refract(unit_vector(light_in_dir_wo), outward_normal, ni_over_nt, mirror_wo))
// 	// {
// 	// 	compute_fuzz = dot(mirror_wo, light_in_dir_wo);
// 	// 	// compute_fuzz = compute_fuzz * compute_fuzz * compute_fuzz * compute_fuzz * compute_fuzz;

// 	// 	if (compute_fuzz <= 0)
// 	// 	{
// 	// 		return vec3(0, 0, 0);
// 	// 		// compute_fuzz = -compute_fuzz;
// 	// 	}
// 	// 	if (compute_fuzz > 1)
// 	// 	{
// 	// 		std::cout << "sss" << std::endl;
// 	// 	}
// 	// 	vec3 ret_color = vec3(1, 1, 1) * compute_fuzz / 5;
// 	// 	return ret_color;
// 	// }
// 	// else
// 	// {
// 	// 	return vec3(1, 1, 1) / 5;
// 	// }

// 	vec3 shade_point_coord = p.p;
// 	vec3 shade_point_normal = p.normal;

// 	vec3 mirror_reflect_wi(0, 0, 0); // 理想状况下光线反射出的方向
// 	vec3 mirror_refract_wi(0, 0, 0); // 理想状况下光线折射出的方向

// 	float compute_fuzz_refract = 0;
// 	float compute_fuzz_reflect = 0;

// 	vec3 outward_normal;
// 	vec3 refracted;
// 	float ni_over_nt;
// 	float reflect_prob;
// 	float cosine;

// 	vec3 ret_color_refract(0, 0, 0);
// 	vec3 ret_color_reflect(0, 0, 0);

// 	// 大于零：光源在晶体表面内侧； 小于零：光源在晶体表面外侧
// 	const float surface_lightSource_direction = dot(light_in_dir_wi, shade_point_normal);
// 	// 大于零：观测点在晶体表面外侧； 小于零：观测点在晶体表面内侧
// 	const float surface_viewPoint_direction = dot(light_in_dir_wo, shade_point_normal);

// 	// 光源在晶体内侧
// 	if (surface_lightSource_direction > 0)
// 	{
// 		// 我们将法线方向定义为由外部指向内部方便计算
// 		outward_normal = -shade_point_normal;
// 		// ref_idx是大于1的相对折射率，ni_over_nt是入射端折射率除以出射端折射率
// 		ni_over_nt = ref_idx;
// 		cosine = dot(light_in_dir_wi, -outward_normal / light_in_dir_wi.length());
// 		reflect_prob = schlick(cosine, ref_idx);

// 		// 以下的分支都有两点需要计算
// 		// 1/由斯涅尔折射定律计算得到的反射与折射的光能分配配比
// 		// 2/由玻璃表面粗糙程度fuzz所决定的反射/折射光的能量衰减
// 		// 这是你之后需要做的 2023-02-11

// 		// // #1 观测点在晶体外侧，与光源分居晶体表面两侧，这时候是折射情况
// 		// if (surface_viewPoint_direction > 0)
// 		// {
// 		// 	refract(light_in_dir_wo, outward_normal, ni_over_nt, mirror_refract_wi);
// 		// 	mirror_refract_wi.make_unit_vector();
// 		// 	compute_fuzz = dot(light_in_dir_wo, mirror_refract_wi);
// 		// 	ret_color = vec3(1, 1, 1) * compute_fuzz * (1 - reflect_prob);
// 		// }
// 		// // #2 观测点在晶体内侧，与光源均在晶体内侧，这时候是反射情况
// 		// else
// 		// {
// 		// 	mirror_reflect_wi = reflect(light_in_dir_wo, outward_normal);
// 		// 	mirror_reflect_wi.make_unit_vector();
// 		// 	compute_fuzz = dot(light_in_dir_wo, mirror_reflect_wi);
// 		// 	ret_color = vec3(1, 1, 1) * compute_fuzz * reflect_prob;
// 		// }
// 	}
// 	// 光源在晶体外侧
// 	else
// 	{
// 		outward_normal = shade_point_normal; // 我们将法线方向定义为由内部指向外部
// 		ni_over_nt = 1.0 / ref_idx;
// 		// cosine = -dot(wi, shade_point_normal / wi.length());

// 		cosine = dot(light_in_dir_wi, -outward_normal / light_in_dir_wi.length());
// 		reflect_prob = schlick(cosine, ref_idx);

// 		// // #3 观测点在晶体外侧，与光源均在晶体外侧，这时候是反射情况
// 		// if (surface_viewPoint_direction > 0)
// 		// {
// 		// 	mirror_reflect_wi = reflect(light_in_dir_wo, outward_normal);
// 		// 	mirror_reflect_wi.make_unit_vector();
// 		// 	compute_fuzz = dot(light_in_dir_wo, mirror_reflect_wi);
// 		// 	ret_color = vec3(1, 1, 1) * compute_fuzz * reflect_prob;
// 		// }
// 		// // #4 观测点在晶体内侧，与光源分居晶体表面两侧，这时候是折射情况
// 		// else
// 		// {
// 		// 	refract(light_in_dir_wo, outward_normal, ni_over_nt, mirror_refract_wi);
// 		// 	mirror_refract_wi.make_unit_vector();
// 		// 	compute_fuzz = dot(light_in_dir_wo, mirror_refract_wi);
// 		// 	ret_color = vec3(1, 1, 1) * compute_fuzz * (1 - reflect_prob);
// 		// }
// 	}

// 	mirror_reflect_wi = reflect(light_in_dir_wo, outward_normal);
// 	mirror_reflect_wi.make_unit_vector();
// 	if (refract(light_in_dir_wo, outward_normal, ni_over_nt, mirror_refract_wi))
// 	{
// 		mirror_refract_wi.make_unit_vector();
// 	}

// 	compute_fuzz_refract = dot(mirror_refract_wi, light_in_dir_wo);
// 	if (compute_fuzz_refract <= 0)
// 	{
// 		ret_color_refract = vec3(0, 0, 0);
// 	}
// 	compute_fuzz_reflect = dot(mirror_reflect_wi, light_in_dir_wo);
// 	if (compute_fuzz_reflect <= 0)
// 	{
// 		ret_color_reflect = vec3(0, 0, 0);
// 	}
// 	ret_color_refract = vec3(1, 1, 1) * compute_fuzz_refract * (1 - reflect_prob);
// 	ret_color_reflect = vec3(1, 1, 1) * compute_fuzz_reflect * (reflect_prob);

// 	return ret_color_refract + ret_color_reflect;

// 	// 目前问题：
// 	// 1. compute_fuzz可能是负值或大于1？！

// 	// // 如果不发生全反射现象
// 	// if (refract(wi, outward_normal, ni_over_nt, refracted))
// 	// {
// 	// 	reflect_prob = schlick(cosine, ref_idx); // 应该是由菲涅尔公式近似计算出的反射光线强度
// 	// }											 // 其实是（转化成）反射光占总光线之比，在抗锯齿章节我们将一个像素点由多（100）条射线表示
// 	// else
// 	// {
// 	// 	reflect_prob = 1.0; // 如果全反射，则反射光占比为100%
// 	// }

// 	// // 这个值越大说明入射角和实际出射角之间的距离差越小，可以接受更多光能
// 	// float compute_fuzz = dot(mirror_wi, wo);
// 	// compute_fuzz *= compute_fuzz;

// 	// if (compute_fuzz <= 0.5)
// 	// {
// 	// 	return vec3(0, 0, 0);
// 	// }

// 	// vec3 ret_color = vec3(1, 1, 1) * compute_fuzz * reflect_prob;

// 	// return ret_color;
// 	// return vec3(0, 0, 0);
// }
