#pragma once
#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "object/hitable.h"
#include "primitive.h"
#include "utils/vertex.h"
#include "material/material.h"
#include "accel/bounds.h"

// 对于 triangle 类型，重新定义关于 hitable 的派生类
class triangle : public primitive
{
public:
    triangle() = default;

    /*
        三角形面元应该支持两种构造函数重载：
        1/直接传入三个顶点信息
        2/传入顶点列表和索引缓冲区
    */
    // 第一种：传入三个顶点进行构造
    triangle(vertex v0, vertex v1, vertex v2, material *mat)
    {
        index[0] = 0;
        index[1] = 1;
        index[2] = 2;

        vertices[0] = v0;
        vertices[1] = v1;
        vertices[2] = v2;

        edges[0] = vertices[1].position - vertices[0].position;
        edges[1] = vertices[2].position - vertices[1].position;
        edges[2] = vertices[0].position - vertices[2].position;

        normal = normalized_vec(cross(edges[0], edges[1]));

        mat_ptr = mat;
        // 获取当前三角形的包围盒，并将其传入成员变量
        bounding_box(0, 0, bounds);



        // 在这里计算一下三角形面积
        float a = edges[0].length();
        float b = edges[1].length();
        float c = edges[2].length();

        float p = (a + b + c) / 2;

        area = sqrt(p * (p - a) * (p - b) * (p - c));
        
    };
    // 第二种：传入顶点列表以及索引值

    triangle(uint32_t i0, uint32_t i1, uint32_t i2, vertex *vertexList, material *mat)
    {
        index[0] = i0;
        index[1] = i1;
        index[2] = i2;

        vertices[0] = vertexList[i0];
        vertices[1] = vertexList[i1];
        vertices[2] = vertexList[i2];

        edges[0] = vertices[1].position - vertices[0].position;
        edges[1] = vertices[2].position - vertices[1].position;
        edges[2] = vertices[0].position - vertices[2].position;

        normal = normalized_vec(cross(edges[0], edges[1]));

        mat_ptr = mat;
        // 获取当前三角形的包围盒，并将其传入成员变量
        bounding_box(0, 0, bounds);


        // 在这里计算一下三角形面积
        float a = edges[0].length();
        float b = edges[1].length();
        float c = edges[2].length();

        float p = (a + b + c) / 2;

        area = sqrt(p * (p - a) * (p - b) * (p - c));
    };
    /*
        判断三角形与射线是否相交，如果相交则返回true并要更新交点坐标，返回交点信息
    */
    bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const;
    /*
        返回包围盒
    */
    bool bounding_box(float t0, float t1, aabb &box) const;
	virtual aabb getBound(void) const;
	virtual bool hasEmission(void) const { return mat_ptr->hasEmission(); };

    void Sample(hit_record &pos, float &probability);
    float getArea();


    // 三角形索引缓冲区
    uint8_t index[3];
    // 三角形顶点缓冲区
    vertex vertices[3];

    // 三角形面积，用于光源采样，初始化为0，每当有光线击中时，再根据顶点位置进行计算
    float area;

    // 规定第一条边是第0个顶点指向第1个顶点
    // 规定第二条边是第1个顶点指向第2个顶点
    // 规定第三条边是第2个顶点指向第0个顶点
    vec3 edges[3];
    // 面法向量，我们规定，面符合右手螺旋定则，按照索引找出其“正面”（逆时针为正面）
    vec3 normal;

    material *mat_ptr;

    aabb bounds;
};

#endif