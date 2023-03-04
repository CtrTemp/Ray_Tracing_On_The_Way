#pragma once
#ifndef SPHERE_H
#define SPHERE_H

#include "object/hitable.cuh"
#include "math/device_rand.cuh"

// sphere球体是可打击类hitable的继承
// 需要重写hit函数和boundingbox函数
class sphere : public hitable
{ // 是sphere类，构造函数返回hitable*类型
public:
	__device__ sphere() = default;
	__device__ sphere(vec3 cen, float r, material *mat) : center(cen), radius(r), mat_ptr(mat)
	{
		area = 4 * M_PI * radius * radius;
	};
	__device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const
	{
		vec3 oc = r.origin() - center;
		// std::cout << "why" << std::endl;
		float a = dot(r.direction(), r.direction());
		float b = 2.0 * dot(oc, r.direction());
		float c = dot(oc, oc) - radius * radius;
		float discriminant = b * b - 4 * a * c;
		// 这里是不是写错了？不应该是b*b-4ac么
		// 不错，已经进行了修改，就应该是b*b - 4*a*c

		// 以下有一个优先返回原则：优先返回双解中离观察点（射线发射点）最近的击中点
		// 注意！我们是传引用，在函数中就直接可以改变 rec结构体变量 的各类值
		if (discriminant > 0)
		{
			float temp = (-b - sqrt(discriminant)) / a / 2;
			if (temp < t_max && temp > t_min)
			{
				rec.t = temp;							// 得到击中点的 t值 并储存入record
				rec.p = r.point_at_parameter(rec.t);	// 得到击中点的坐标 并储存人record
				rec.normal = (rec.p - center) / radius; // 得到击中点的单位法向向量
				rec.mat_ptr = this->mat_ptr;
				rec.happened = true;
				return true;
			}
			temp = (-b + sqrt(discriminant)) / a / 2;
			if (temp < t_max && temp > t_min)
			{
				rec.t = temp;
				rec.p = r.point_at_parameter(rec.t);
				rec.normal = (rec.p - center) / radius;
				rec.mat_ptr = this->mat_ptr;
				rec.happened = true;
				return true;
			}
		}
		rec.happened = false;
		return false;
	}

	__device__ virtual bool hasEmission(void) const
	{
		return mat_ptr->hasEmission();
	}

	vec3 center;
	float radius;
	float area;
	material *mat_ptr;
};

// __device__ bool sphere::hit(const ray &r, float t_min, float t_max, hit_record &rec) const

// __device__ bool sphere::hasEmission(void) const

#endif