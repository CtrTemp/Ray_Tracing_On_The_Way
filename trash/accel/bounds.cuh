#pragma once
#ifndef BOUNDS_H
#define BOUNDS_H

#include "utils/ray.cuh"

#include <array>
// aabb是包围盒基类, 这里的包围盒用于之后的 BVH_node 加速算法
// 基本算法不需要包围盒
class aabb
{
public:
    // 这里默认构建的是一个无穷大的包围盒
    aabb()
    {
        double minNum = std::numeric_limits<double>::lowest();
        double maxNum = std::numeric_limits<double>::max();
        _min = vec3(minNum, minNum, minNum);
        _max = vec3(maxNum, maxNum, maxNum);
    };

    // 基本的构造函数是:通过包围盒对角线的两个顶点来确定一个立方体
    aabb(const vec3 &a, const vec3 &b)
    {
        _min = vec3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
        _max = vec3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));
    }
    vec3 min() const { return _min; }
    vec3 max() const { return _max; }

    // 用于返回是否击中?
    bool hit(const ray &r, float tmin, float tmax) const;

    // 用于返回对角线
    inline vec3 Diagonal() const { return _max - _min; }

    // 找最大轴跨度的函数,返回当前BoundingBox跨度最大的轴
    // 0: x 轴跨度最大
    // 1: y 轴跨度最大
    // 2: z 轴跨度最大
    int maxExtent() const
    {
        vec3 d = Diagonal();
        if (d.x() > d.y() && d.x() > d.z())
            return 0;
        else if (d.y() > d.z())
            return 1;
        else
            return 2;
    }

    // 返回质心坐标
    vec3 center() { return 0.5 * _min + 0.5 * _max; }

    // 求两个box相重叠的部分,并返回这个重叠部分的box
    aabb Intersect(const aabb &b)
    {
        return aabb(vec3(fmax(_min.x(), b._min.x()), fmax(_min.y(), b._min.y()),
                         fmax(_min.z(), b._min.z())),
                    vec3(fmin(_max.x(), b._max.x()), fmin(_max.y(), b._max.y()),
                         fmin(_max.z(), b._max.z())));
    }

    // 判断b2是否在b1内部(完全包裹)
    bool Overlaps(const aabb &b1, const aabb &b2)
    {
        bool x = (b1._max.x() >= b2._min.x()) && (b1._min.x() <= b2._max.x());
        bool y = (b1._max.y() >= b2._min.y()) && (b1._min.y() <= b2._max.y());
        bool z = (b1._max.z() >= b2._min.z()) && (b1._min.z() <= b2._max.z());
        return (x && y && z);
    }

    // 判断点是否在包围盒内
    bool Inside(const vec3 &p, const aabb &b)
    {
        return (p.x() >= b._min.x() && p.x() <= b._max.x() && p.y() >= b._min.y() &&
                p.y() <= b._max.y() && p.z() >= b._min.z() && p.z() <= b._max.z());
    }

    // 声明一个射线与包围盒的相交判断函数
    bool IntersectP(const ray &ray, const vec3 &invDir, const std::array<int, 3> dirIsNeg) const;

    // 包围盒的两个顶点
    vec3 _min;
    vec3 _max;
};

// 两个包围盒做 merge, 扩成一个更大的box
inline aabb Union(const aabb &b1, const aabb &b2)
{
    aabb ret;
    ret._min = vec3::Min(b1.min(), b2.min());
    ret._max = vec3::Max(b1.max(), b2.max());

    return ret;
}

// 对一个包围盒做扩充, 如果点p在包围盒外, 则将包围盒扩充到刚好可以包裹住点p的位置
inline aabb Union(const aabb &b, const vec3 &p)
{
    aabb ret;
    ret.min() = vec3::Min(b.min(), p);
    ret.max() = vec3::Max(b.max(), p);
    return ret;
}

#endif