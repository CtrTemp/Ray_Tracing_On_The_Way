#pragma once
#ifndef HITABLE_H
#define HITABLE_H

#include "utils/ray.h"
#include "accel/bounds.h"
#include "material/material.h"

inline float ffmin(float a, float b) { return a < b ? a : b; }
inline float ffmax(float a, float b) { return a > b ? a : b; }

//这个结构体用来记录那些与物体相交的射线的一些参数
struct hit_record
{
	float t;	 //用于记录击中点时，此时射线方向向量的乘数
	Vector3f p;		 //用于记录击中点坐标
	Vector3f normal; //用于记录击中点处的法线（注意是单位向量）

	// 这里是 uv 贴图坐标
	float u;
	float v; //最新引入，我还不清楚这个是干啥的

	material *mat_ptr; // 当前击中点的材质
	bool happened;	   // 是否发生打击
};

class hitable
{
public:
	/**
	 * 默认构造函数，不需要实例，因为这是一个抽象类，抽象类只需要指明其派生类应该具有的基本操作
	 * 而不需要为其创建实例，所以构造函数也无需实体，而是全部会被映射到具体的派生类中
	 **/
	hitable() = default;

	// hitable 为基类, 以下以 virtual 关键字定义虚函数, 后由其继承的子类来重写虚函数进行覆盖
	// 故在此基本只是声明,但不进行函数定义  const = 0 代表其为纯虚函数
	virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const = 0;

	// 返回一个物体的包围盒, 显然这里是通过传入的 aabb 的指针来进行隐式的返回
	// 同样,在基本算法中不适用BVH(层次包围盒)加速,不需要这个组件
	virtual bool bounding_box(float t0, float t1, aabb &box) const = 0;

	virtual aabb getBound(void) const = 0;

	virtual bool hasEmission(void) const = 0;

	// 采样函数，对某个可求交物体，给出它表面上的一个特定坐标，并且给定取样到这个坐标的概率
	virtual void Sample(hit_record &pos, float &probability) = 0;
	// 得到目标物体的总面积
	virtual float getArea() = 0;

};

#endif
