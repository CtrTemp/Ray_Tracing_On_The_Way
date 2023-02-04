#pragma once
#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H

#include "material/material.h"
#include "texture/textures.h"
#include "object/hitable.h"

/******************************** 哑光类/漫反射表面 *********************************/

class lambertian : public material
{
public:
    lambertian(texture *a) : albedo(a) {}
    virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuated, ray &scattered) const;
    virtual vec3 emitted(float u, float v, const vec3 &p) const { return vec3(0, 0, 0); };
    virtual bool hasEmission(void) const { return false; };
    virtual vec3 computeBRDF(ray r_in, ray r_out, vec3 normal);


    texture *albedo;
    vec3 BRDF;
};

#endif