#pragma once
#ifndef DIFFUSE_LIGHT
#define DIFFUSE_LIGHT

#include "material/material.h"
#include "texture/textures.h"
#include "object/hitable.h"

class diffuse_light : public material
{
public:
	diffuse_light(texture *a) : emit(a) {}
	virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuated, ray &scattered) const;
	virtual vec3 emitted(float u, float v, const vec3 &p) const;
	virtual bool hasEmission(void) const { return true; };

	virtual vec3 computeBRDF(const vec3 wi, const vec3 wo, const hit_record p) { return vec3(0, 0, 0); };

	float pdf(vec3 r_in_dir, vec3 r_out_dir, vec3 normal) { return 1.0f; }
	SelfMaterialType getMaterialType() { return self_type; }

	texture *emit;
	vec3 BRDF;
	SelfMaterialType self_type = SelfMaterialType::LIGHT;
};

#endif