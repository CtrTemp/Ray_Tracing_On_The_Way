#pragma once
#ifndef DIELECTRIC_H
#define DIELECTRIC_H

#include "material.h"
#include "texture/textures.h"
#include "object/hitable.h"

/******************************** 玻璃类/透射表面 *********************************/
class dielectric : public material
{
public:
	dielectric(float ri) : ref_idx(ri) {}
	virtual bool scatter(const ray &r_in, const hit_record &rec, Vector3f &attenuation, ray &scattered) const;
	virtual Vector3f emitted(float u, float v, const Vector3f &p) const { return Vector3f(0, 0, 0); };
	virtual bool hasEmission(void) const { return false; };
	virtual Vector3f computeBRDF(const Vector3f light_in_dir_wi, const Vector3f light_in_dir_wo, const hit_record p);
	float pdf(Vector3f r_in_dir, Vector3f r_out_dir, Vector3f normal);
	SelfMaterialType getMaterialType() { return self_type; }

	float ref_idx;
	Vector3f BRDF;
	SelfMaterialType self_type = SelfMaterialType::DIELECTRIC;
};

#endif