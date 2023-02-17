#pragma once
#ifndef HITABLELIST_H
#define HITABLELIST_H

#include "object/hitable.h"
#include "accel/bvh.h"

#include <memory>
#include <vector>

aabb surronding_box(aabb box0, aabb box1);

class hitable_list : public hitable
{ //是hitable_list类，构造函数返回hitable*类型？
public:
	enum class HitMethod
	{
		NAIVE,
		BVH_TREE
	};
	hitable_list() = default;
	hitable_list(std::vector<hitable *> l)
	{
		list = l;
		list_size = l.size();
		// default construct tree in sence
		tree = new bvh_tree_scene(l);
		method = HitMethod::NAIVE;
		// method = HitMethod::BVH_TREE;
		bounding_box(0, 0, bounds);
	}

	hitable_list(std::vector<hitable *> l, HitMethod m)
	{
		list = l;
		list_size = l.size();
		tree = new bvh_tree_scene(l);
		method = m;
		bounding_box(0, 0, bounds);
	}

	virtual bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const;
	virtual bool bounding_box(float t0, float t1, aabb &box) const;
	virtual aabb getBound(void) const;
	// 暂定这种类型没有 光线 emission
	virtual bool hasEmission(void) const { return false; };

    void Sample(hit_record &pos, float &probability);
    float getArea();

	std::vector<hitable *> list;
	int list_size;

	aabb bounds;
	bvh_tree_scene *tree;
	HitMethod method;
};


#endif
