#pragma once
#ifndef TRIANGLE_LIST_H
#define TRIANGLE_LIST_H

#include "triangle.h"
#include "material.h"
#include "string"
#include "vector"
#include "bvh.h"

class triangleList : public hitable
{
public:
    enum class HitMethod
    {
        NAIVE,
        BVH_TREE
    };

    triangleList() = default;
    // 第一种方式通过传入一个三角形列表来构建
    triangleList(std::vector<triangle *> tri, int n, HitMethod m) : tri_list(tri), list_size(n), method(m)
    {
        // 以下测试使用，正常情况下不会为这种构造函数构建的三角形列表建立加速结构
        tree = new bvh_tree(tri_list);
        // throw std::runtime_error("break point construct BVH_TREE");
    };
    // 第二种方式通过传入顶点数组以及索引数组来构建
    triangleList(vertex *vertList, uint32_t *indList, uint32_t ind_len, material *mat, HitMethod m);
    // 第三种方式直接从模型地址导入构建
    // 仅在这种构建方式下，我们为其构建层级包围盒加速结构（BVH_Node_Tree）
    triangleList(const std::string module_path, material *mat, HitMethod m);

    virtual bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const;
    virtual bool bounding_box(float t0, float t1, aabb &box) const;

    std::vector<triangle *> tri_list;
    int list_size;
    aabb bounds;
    bvh_tree *tree;
    HitMethod method;
};

#endif
