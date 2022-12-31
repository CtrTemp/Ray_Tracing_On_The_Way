#pragma once
#ifndef BOX_H
#define BOX_H
//当前文件在GlobalInclude/Chapter/二级目录下
#include "../../GlobalInclude/hitable.h"
#include "../../GlobalInclude/hitableList.h"
#include "../../GlobalInclude/Chapter/diffuse_light.h"

class box :public hitable {
public:
	box() = default;
	box(const vec3 &p0, const vec3 &pl, material *ptr);

	virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec)const;
	virtual bool bounding_box(float t0, float t1, aabb &box)const;

	vec3 pmin, pmax;
	hitable *list_ptr;
};



class translate : public hitable {
public:
	translate() = default;
	translate(hitable *p, const vec3 &displacement) :ptr(p), offset(displacement) {}

	virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec)const;
	virtual bool bounding_box(float t0, float t1, aabb &box)const;

	hitable *ptr;
	vec3 offset;
};




class rotate_y : public hitable {
public:
	rotate_y() = default;
	rotate_y(hitable *p, float angle);

	virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec)const;
	virtual bool bounding_box(float t0, float t1, aabb &box)const;

	hitable *ptr;
	float sin_theta;
	float cos_theta;
	bool hasbox;
	aabb bbox;

};



#endif // !BOX_H
