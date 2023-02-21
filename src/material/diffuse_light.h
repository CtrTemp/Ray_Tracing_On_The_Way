#pragma once
#ifndef DIFFUSE_LIGHT
#define DIFFUSE_LIGHT

#include "material.h"
#include "texture/textures.h"
#include "object/hitable.h"

class diffuse_light : public material
{
public:
	diffuse_light(texture *a) : emit(a) {}
	virtual bool scatter(const ray &r_in, const hit_record &rec, Vector3f &attenuated, ray &scattered) const;
	virtual Vector3f emitted(float u, float v, const Vector3f &p) const;
	virtual bool hasEmission(void) const { return true; };

	virtual Vector3f computeBRDF(const Vector3f wi, const Vector3f wo, const hit_record p) { return Vector3f(0, 0, 0); };

	float pdf(Vector3f r_in_dir, Vector3f r_out_dir, Vector3f normal) { return 1.0f; }
	SelfMaterialType getMaterialType() { return self_type; }

	texture *emit;
	Vector3f BRDF;
	SelfMaterialType self_type = SelfMaterialType::LIGHT;
};

#endif