#pragma once
#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "material.cuh"
#include "texture/textures.cuh"
#include "object/hitable.cuh"
#include "math/device_rand.cuh"

/******************************** 哑光类/漫反射表面 *********************************/

class lambertian : public material
{
public:
    __host__ __device__ lambertian(textures *a) : albedo(a) {}
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandStateXORWOW_t *rand_state) const
    {
        vec3 target = rec.p + rec.normal + random_in_unit_sphere_device(rand_state); // 获得本次打击后得到的下一个目标点

        scattered = ray(rec.p, target - rec.p); // 本次击中一个目标点后的下一个射线（获得散射光线）
        // attenuated = albedo->value(rec.u, rec.v, rec.p);
        attenuation = albedo->value(rec.u, rec.v, rec.p);

        return (dot(scattered.direction(), rec.normal) > 0);
        // return true;
    }
    __device__ virtual vec3 emitted(float u, float v, const vec3 &p) const { return vec3(0, 0, 0); };
    __device__ virtual bool hasEmission(void) const { return false; };

    // BRDF 计算函数
    __device__ virtual vec3 computeBRDF(const vec3 light_in_dir_wi, const vec3 light_in_dir_wo, const hit_record p)
    {
        float cosalpha = dot(p.normal, -light_in_dir_wi);
        if (cosalpha > 0.0f)
        {
            vec3 diffuse = this->albedo->value(p.u, p.v, p.p) / M_PI;
            return diffuse;
        }
        return vec3(0, 0, 0);
    }
    // 表面采样计算函数
    __device__ virtual float pdf(vec3 r_in_dir, vec3 r_out_dir, vec3 normal)
    {
        if (dot(r_out_dir, normal) > 0.0f)
        {
            return 0.5f / M_PI;
        }
        else
        {
            return 0.0f;
        }
    }
    // 返回表面材质
    __device__ virtual SelfMaterialType getMaterialType() { return self_type; }

    textures *albedo;
    vec3 BRDF;
    SelfMaterialType self_type = SelfMaterialType::LAMBERTAIN;
};

#endif