#pragma once
#ifndef RAY_H
#define RAY_H
#include "utils/vec3.h"

class ray
{
public:
	ray() = default;
	ray(const Vector3f &a, const Vector3f &b, float ti = 0.0)
	{
		A = a;
		B = b;
		_time = ti;
		inv_dir = Vector3f(1. / B.x(), 1. / B.y(), 1. / B.z());
	}


	Vector3f origin() const { return A; }	 //射线起始点
	Vector3f direction() const { return B; } //射线方向向量（应该是单位向量）
	float time() const { return _time; }
	Vector3f point_at_parameter(float t) const { return A + t * B; }
	//给定射线方向向量倍数t，得到射线末端指向点

	Vector3f A;
	Vector3f B;
	Vector3f inv_dir;
	float _time;
};

#endif