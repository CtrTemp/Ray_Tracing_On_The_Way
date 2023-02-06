#pragma once
#ifndef MENTAL_H
#define MENTAL_H

#include "material/material.h"
#include "texture/textures.h"
#include "object/hitable.h"

/******************************** 金属类/镜面 *********************************/
class mental : public material
{
public:
	mental(const vec3 &a, float f) : albedo(a)
	{
		if (f < 1)
			fuzz = f;
		else
			fuzz = 1;
	}

	virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuated, ray &scattered) const;
	virtual vec3 emitted(float u, float v, const vec3 &p) const { return vec3(0, 0, 0); };
	virtual bool hasEmission(void) const { return false; };
    virtual vec3 computeBRDF(const vec3 wi, const vec3 wo, const hit_record p);

	
	vec3 albedo;
	float fuzz;
    vec3 BRDF;
};

#endif