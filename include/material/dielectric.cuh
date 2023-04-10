#pragma once
#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include "material.cuh"
// #include "texture/textures.h"
#include "object/hitable.cuh"

/******************************** 玻璃类/透射表面 *********************************/
class dielectric : public material
{
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandStateXORWOW_t *rand_state) const
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
            // 我现在就让他全反射 OKK 没问题测试通过
            // reflect_prob = 1.0;
            // 现在测试全折射
            // reflect_prob = 0.0;
            reflect_prob = schlick(cosine, ref_idx); // 应该是由菲涅尔公式近似计算出的反射光线强度
        }                                            // 其实是（转化成）反射光占总光线之比，在抗锯齿章节我们将一个像素点由多（100）条射线表示
        else
        {
            reflect_prob = 1.0; // 如果全反射，则反射光占比为100%
            // 现在测试全折射，出现了之前发现的问题，中间有一个亮圈！其余部分黑的
            // reflect_prob = 0.0;
        }

        // 在发生折射的情况下，每次也只生成一条光线，要么折射光，要么反射光，二者占比满足折射/反射的光能分配
        // reflect_prob 的值在 0-1 之间，表示反射光的光能占比
        if (random_float_device(rand_state) < reflect_prob)
        {
            scattered = ray(rec.p, reflected);
        } // 明白
        else
        {
            scattered = ray(rec.p, refracted);
        } // 明白

        return true;
    }
    __device__ virtual vec3 emitted(float u, float v, const vec3 &p) const { return vec3(0, 0, 0); };
    __device__ virtual bool hasEmission(int void_input) { return false; };

    // BRDF 计算函数
    __device__ virtual vec3 computeBRDF(const vec3 light_in_dir_wi, const vec3 light_in_dir_wo, const hit_record p)
    {

        vec3 shade_point_coord = p.p;       // 着色点空间坐标
        vec3 shade_point_normal = p.normal; // 着色点表面法向量

        vec3 mirror_reflect_wi(0, 0, 0); // 理想状况下光源光线反射出的方向
        vec3 mirror_refract_wi(0, 0, 0); // 理想状况下光源光线折射出的方向

        float compute_fuzz_refract = 0; // 折射光衰减
        float compute_fuzz_reflect = 0; // 反射光衰减

        vec3 outward_normal; // 与入射光呈钝角的法向量方向（折射光计算使用）
        float ni_over_nt;    // 入射光所在材质相对折射出射光材质的相对折射率（折射光计算使用）
        float reflect_prob;  // 反射光能占比
        float cosine;        // 计算系数（计算反射光占比使用）

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
    // 表面采样计算函数
    __device__ virtual float pdf(vec3 r_in_dir, vec3 r_out_dir, vec3 normal)
    {
        return 1.0f / (2 * M_PI * 0.2);
    }
    // 返回表面材质
    __device__ virtual SelfMaterialType getMaterialType() { return self_type; }

    float ref_idx;
    vec3 BRDF;
    SelfMaterialType self_type = SelfMaterialType::DIELECTRIC;
};

#endif