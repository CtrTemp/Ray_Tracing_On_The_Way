#pragma once
#ifndef TRIANGLE_LIST_H
#define TRIANGLE_LIST_H

#include "triangle.h"
#include "material.h"
#include "string"

class triangleList : public hitable
{
public:
    triangleList() = default;
    // 第一种方式通过传入一个三角形列表来构建
    triangleList(triangle **tri, int n) : tri_list(tri), list_size(n){};
    // 第二种方式通过传入顶点数组以及索引数组来构建
    triangleList(vertex *vertList, uint32_t *indList, uint32_t ind_len, material *mat);
    // 第三种方式直接从模型地址导入构建
    triangleList(const std::string module_path, material *mat);

    virtual bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const;
    virtual bool bounding_box(float t0, float t1, aabb &box) const;

    triangle **tri_list;
    int list_size;
};

#endif
