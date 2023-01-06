#pragma once
#ifndef HITABLE_H
#define HITABLE_H
//当前文件在GlobalInclude/一级目录下
#include "../GlobalInclude/basic/ray.h"
#include "bounds.h"

inline float ffmin(float a, float b) { return a < b ? a : b; }
inline float ffmax(float a, float b) { return a > b ? a : b; }

// 应该是超前引用相关
class material; //这类为什么没有这个“声明”不可以

//这个结构体用来记录那些与物体相交的射线的一些参数
struct hit_record
{
	float t;	 //用于记录击中点时，此时射线方向向量的乘数
	vec3 p;		 //用于记录击中点坐标
	vec3 normal; //用于记录击中点处的法线（注意是单位向量）

	// 这里是 uv 贴图坐标
	float u;
	float v; //最新引入，我还不清楚这个是干啥的

	material *mat_ptr; // 当前击中点的材质
    bool happened;      // 是否发生打击
};

class hitable
{
public:
	// 默认构造函数
	hitable() = default;

	// hitable 为基类, 以下以 virtual 关键字定义虚函数, 后由其继承的子类来重写虚函数进行覆盖
	// 故在此基本只是声明,但不进行函数定义
	virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const = 0;

	// 返回一个物体的包围盒, 显然这里是通过传入的 aabb 的指针来进行隐式的返回
	// 同样,在基本算法中不适用BVH(层次包围盒)加速,不需要这个组件
	virtual bool bounding_box(float t0, float t1, aabb &box) const = 0;

	// 应该要在这里
	aabb bounds;
};

#endif
