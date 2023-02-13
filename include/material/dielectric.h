#pragma once
#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include "material/material.h"
#include "texture/textures.h"
#include "object/hitable.h"

/******************************** 玻璃类/透射表面 *********************************/
class dielectric : public material
{
public:
	dielectric(float ri) : ref_idx(ri) {}
	virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered) const;
	virtual vec3 emitted(float u, float v, const vec3 &p) const { return vec3(0, 0, 0); };
	virtual bool hasEmission(void) const { return false; };
	virtual vec3 computeBRDF(const vec3 light_in_dir_wi, const vec3 light_in_dir_wo, const hit_record p);
	float pdf(vec3 r_in_dir, vec3 r_out_dir, vec3 normal);
	SelfMaterialType getMaterialType() { return self_type; }

	float ref_idx;
	vec3 BRDF;
	SelfMaterialType self_type = SelfMaterialType::DIELECTRIC;
};

#endif