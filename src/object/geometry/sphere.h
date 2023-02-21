#pragma once
#ifndef SPHERE_H
#define SPHERE_H

#include "object/hitable.h"

// sphere球体是可打击类hitable的继承
// 需要重写hit函数和boundingbox函数
class sphere : public hitable
{ // 是sphere类，构造函数返回hitable*类型
public:
	sphere() = default;
	sphere(Vector3f cen, float r, material *mat) : center(cen), radius(r), mat_ptr(mat)
	{
		bounding_box(0, 0, bound);
		area = 4 * M_PI * radius * radius;
	}; // 创造一个球体，包括球心和半径参数

	// 重写的类中的 函数名/参数/返回类型都必须相同
	virtual bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const;
	// 判断参数：给定一条射线
	virtual bool bounding_box(float t0, float t1, aabb &box) const;
	virtual aabb getBound(void) const;
	virtual bool hasEmission(void) const { return mat_ptr->hasEmission(); };

	void Sample(hit_record &pos, float &probability);
	float getArea();

	aabb bound;
	// 球体作为可打击继承类的其他三个额外参数
	Vector3f center;
	float radius;
	float area;
	material *mat_ptr;
};


#endif