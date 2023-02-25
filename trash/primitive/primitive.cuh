#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "object/hitable.cuh"
#include "utils/vertex.cuh"
#include <vector>

/*
    primitive（基元类）是模型表面元素 triangle/quadrangle 的顶层抽象，同时是hitable的派生类
*/

class primitive : public hitable
{

public:
    enum class PrimType
    {
        TRIANGLE,
        QUADRANGLE
    };
    /**
     * @brief 对于hitable的派生类primitive，是一个中间的抽象层，也非具体实现，你可以为其中添加下属子类的方法
     * 但同样不需要为其指明具体的构造函数以及具体的成员变量
     * 
     * 保证继承其基类的所有方法，如果没有具体实现，保持其最初的（纯）虚函数，以被派生类覆盖
     * 在此基础上进行进一步扩充添加
     */
    primitive() = default;
	virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const = 0;
	virtual bool bounding_box(float t0, float t1, aabb &box) const = 0;
    // 这里是一个bug区，暂时这样写，之后要修改
	virtual aabb getBound(void) const {return aabb();};

    virtual void Sample(hit_record &pos, float &probability) = 0;
    virtual float getArea() = 0;

    // 以上进一步写作纯虚函数，由其派生类 triangle 和 quadrangle 进行具体实现
};

#endif
