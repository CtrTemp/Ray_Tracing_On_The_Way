#ifndef DIFFUSE_LIGHT_H
#define DIFFUSE_LIGHT_H

#include "material.cuh"
#include "texture/textures.cuh"
#include "object/hitable.cuh"

class diffuse_light : public material
{
public:
    __device__ diffuse_light() = default;
    __device__ diffuse_light(textures *a) : emit(a) {}
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuated, ray &scattered, curandStateXORWOW_t *rand_state) const
    {
        return false;
    }
    __device__ virtual vec3 emitted(float u, float v, const vec3 &p) const
    {
        return emit->value(u, v, p);
    }

    __device__ virtual bool hasEmission(void) const { return true; };

    __device__ virtual vec3 computeBRDF(const vec3 wi, const vec3 wo, const hit_record p) { return vec3(0, 0, 0); };

    __device__ float pdf(vec3 r_in_dir, vec3 r_out_dir, vec3 normal) { return 1.0f; }
    __device__ SelfMaterialType getMaterialType() { return self_type; }

    textures *emit;
    vec3 BRDF;
    SelfMaterialType self_type = SelfMaterialType::LIGHT;
};

#endif