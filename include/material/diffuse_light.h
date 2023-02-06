#pragma once
#ifndef DIFFUSE_LIGHT
#define DIFFUSE_LIGHT

#include "material/material.h"
#include "texture/textures.h"
#include "object/hitable.h"

class diffuse_light : public material {
public:
	diffuse_light(texture *a) :emit(a) {}
	virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuated, ray& scattered) const;
	virtual vec3 emitted(float u, float v, const vec3 &p)const;
	virtual bool hasEmission(void) const { return true; };
	
    virtual vec3 computeBRDF(const vec3 wi, const vec3 wo, const hit_record p) { return vec3(0, 0, 0); };

	

	texture *emit;
    vec3 BRDF;
};



// class xy_rect : public hitable {
// public:
// 	xy_rect() = default;
// 	xy_rect(float _x0, float _x1, float _y0, float _y1, float _k, material *mat) :
// 		x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {};

// 	virtual bool hit(const ray& r, float t0, float t1, hit_record &rec)const;
// 	virtual bool bounding_box(float t0, float t1, aabb &box) const;
// 	virtual aabb getBound(void) const;
// 	virtual bool hasEmission(void) const { return mp->hasEmission(); };

// 	material *mp;
// 	float x0, y0, x1, y1, k;
// };


// class xz_rect : public hitable {
// public:
// 	xz_rect() = default;
// 	xz_rect(float _x0, float _x1, float _z0, float _z1, float _k, material *mat) :
// 		x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat) {};

// 	virtual bool hit(const ray& r, float t0, float t1, hit_record &rec)const;
// 	virtual bool bounding_box(float t0, float t1, aabb &box) const;
// 	virtual aabb getBound(void) const;
// 	virtual bool hasEmission(void) const { return mp->hasEmission(); };

// 	material *mp;
// 	float x0, z0, x1, z1, k;
// };


// class yz_rect : public hitable {
// public:
// 	yz_rect() = default;
// 	yz_rect(float _y0, float _y1, float _z0, float _z1, float _k, material *mat) :
// 		y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat) {};

// 	virtual bool hit(const ray& r, float t0, float t1, hit_record &rec)const;
// 	virtual bool bounding_box(float t0, float t1, aabb &box) const;
// 	virtual aabb getBound(void) const;
// 	virtual bool hasEmission(void) const { return mp->hasEmission(); };

// 	material *mp;
// 	float y0, z0, y1, z1, k;
// };




// class  flip_normals :public hitable {
// public:
// 	flip_normals() = default;
// 	flip_normals(hitable *p) :ptr(p) {}
// 	virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const;
// 	virtual bool bounding_box(float t0, float t1, aabb &box) const;
// 	virtual aabb getBound(void) const;
// 	virtual bool hasEmission(void) const { return ptr->hasEmission(); };

// 	hitable *ptr;
// };


#endif