#pragma once
#ifndef HITABLELIST_H
#define HITABLELIST_H

#include "hitable.h"
#include <memory>
#include <vector>



aabb surronding_box(aabb box0, aabb box1);



class hitable_list :public hitable {//是hitable_list类，构造函数返回hitable*类型？
public:
	hitable_list() = default;
	hitable_list(hitable **l, int n) { list = l; list_size = n; }

	virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	virtual bool bounding_box(float t0, float t1, aabb &box) const;

	hitable **list;
	int list_size;
};



class bvh_node :public hitable {
public:
	bvh_node() = default;
	bvh_node(hitable **l, int n, float time0, float time1);

	virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
	virtual bool bounding_box(float t0, float t1, aabb& box) const;


	hitable *left;
	hitable *right;
	aabb box;
};



int box_x_compare(const void *a, const void *b);

int box_y_compare(const void *a, const void *b);

int box_z_compare(const void *a, const void *b);


#endif
