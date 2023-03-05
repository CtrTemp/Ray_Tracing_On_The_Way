#pragma once
#ifndef MENTAL_H
#define MENTAL_H

#include "material.cuh"
// #include "texture/textures.h"
#include "object/hitable.cuh"

/******************************** 金属类/镜面 *********************************/
class mental : public material
{
public:
    __device__ mental(const vec3 &a, float f) : albedo(a)
    {
        if (f < 1)
            fuzz = f;
        else
            fuzz = 1;
    }

    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandStateXORWOW_t *rand_state) const
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal); // 获得反射方向向量
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere_device(rand_state));  // 获得散射方向向量
        // fuzz参数代表金属表面的“哑光”程度，即我们使用scattered散射参数来间接的代表reflect参数，
        // 可见fuzz参数越大，scattered偏离reflect光线的程度，或者说不确定度就越大，也就表现出一种
        // 光线“随机反射”的哑光、漫反射特性
        attenuation = albedo;

        return (dot(scattered.direction(), rec.normal) > 0);
        // 返回false说明散射光线与原法线成钝角，直观上理解：此时散射光线射入了object内部，即为光线被吸收
    }
    __device__ virtual vec3 emitted(float u, float v, const vec3 &p) const { return vec3(0, 0, 0); };
    __device__ virtual bool hasEmission(void) const { return false; };
    // virtual vec3 computeBRDF(const vec3 light_in_dir_wi, const vec3 light_in_dir_wo, const hit_record p);
    // float pdf(vec3 r_in_dir, vec3 r_out_dir, vec3 normal);
    // SelfMaterialType getMaterialType() { return self_type; }

    vec3 albedo;
    float fuzz;
    // vec3 BRDF;
    // SelfMaterialType self_type = SelfMaterialType::MENTAL;
};

#endif