#pragma once
#ifndef HITABLE_H
#define HITABLE_H
//当前文件在GlobalInclude/一级目录下
#include "../GlobalInclude/basic/ray.h"

inline float ffmin(float a, float b) { return a < b ? a : b; }
inline float ffmax(float a, float b) { return a > b ? a : b; }

// aabb是包围盒基类, 这里的包围盒用于之后的 BVH_node 加速算法
// 基本算法不需要包围盒
class aabb
{
public:
	aabb() = default;

	// 基本的构造函数是:通过包围盒对角线的两个顶点来确定一个立方体
	aabb(const vec3 &a, const vec3 &b)
	{
		_min = a;
		_max = b;
	}

	vec3 min() const { return _min; }
	vec3 max() const { return _max; }

	// 用于返回是否击中?
	bool hit(const ray &r, float tmin, float tmax) const;
	/*

	{
	for (int a = 0; a < 3; ++a)
	{
	float t0 = ffmin((_min[a] - r.origin()[a]) / r.direction()[a], (_max[a] - r.origin()[a]) / r.direction()[a]);
	float t1 = ffmax((_min[a] - r.origin()[a]) / r.direction()[a], (_max[a] - r.origin()[a]) / r.direction()[a]);
	tmin = ffmax(t0, tmin);
	tmax = ffmin(t1, tmax);
	if (tmax <= tmin)
	return false;
	}
	return true;
	}
	*/

	// 包围盒的两个顶点
	vec3 _min;
	vec3 _max;
};

//问题来了：如何通过调用区分这两个重载函数？！

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
};

#endif
