#pragma once
#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "material.cuh"
// #include "texture/textures.cuh"
#include "object/hitable.cuh"
#include "math/device_rand.cuh"

/******************************** 哑光类/漫反射表面 *********************************/

class lambertian : public material
{
public:
    __device__ lambertian(vec3 a) : albedo(a) {}
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandStateXORWOW_t *rand_state) const
    {
        vec3 target = rec.p + rec.normal + random_in_unit_sphere_device(rand_state); // 获得本次打击后得到的下一个目标点

        scattered = ray(rec.p, target - rec.p); // 本次击中一个目标点后的下一个射线（获得散射光线）
        // attenuated = albedo->value(rec.u, rec.v, rec.p);
        attenuation = albedo;

        return (dot(scattered.direction(), rec.normal) > 0);
        // return true;
    }
    __device__ virtual vec3 emitted(float u, float v, const vec3 &p) const { return vec3(0, 0, 0); };
    __device__ virtual bool hasEmission(void) const { return false; };

    vec3 albedo;
};

#endif