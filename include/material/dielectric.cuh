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
            reflect_prob = schlick(cosine, ref_idx); // 应该是由菲涅尔公式近似计算出的反射光线强度
        }                                            // 其实是（转化成）反射光占总光线之比，在抗锯齿章节我们将一个像素点由多（100）条射线表示
        else
        {
            reflect_prob = 1.0; // 如果全反射，则反射光占比为100%
        }

        // 在发生折射的情况下，每次也只生成一条光线，要么折射光，要么反射光，二者占比满足折射/反射的光能分配
        // reflect_prob 的值在 0-1 之间，表示反射光的光能占比
        if (random_double_device(rand_state) < reflect_prob)
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
    __device__ virtual bool hasEmission(void) const { return false; };
    // virtual vec3 computeBRDF(const vec3 light_in_dir_wi, const vec3 light_in_dir_wo, const hit_record p);
    // float pdf(vec3 r_in_dir, vec3 r_out_dir, vec3 normal);
    // SelfMaterialType getMaterialType() { return self_type; }

    float ref_idx;
    // vec3 BRDF;
    // SelfMaterialType self_type = SelfMaterialType::DIELECTRIC;
};

#endif