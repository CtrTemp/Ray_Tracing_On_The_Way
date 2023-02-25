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
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuated, ray &scattered, curandStateXORWOW_t *rand_state) const;
    __device__ virtual vec3 emitted(float u, float v, const vec3 &p) const { return vec3(0, 0, 0); };
    __device__ virtual bool hasEmission(void) const { return false; };

    vec3 albedo;
};

#endif