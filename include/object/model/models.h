#pragma once
#ifndef TRIANGLE_LIST_H
#define TRIANGLE_LIST_H

#include "object/primitive/triangle.h"
#include "object/primitive/primitive.h"
#include "material/material.h"
#include "accel/bvh.h"

#include <string>
#include <vector>

class models : public hitable
{
public:
    enum class HitMethod
    {
        NAIVE,
        BVH_TREE
    };
    enum class PrimType
    {
        TRIANGLE,
        QUADRANGLE
    };

    /**
     * @brief 随着参数越来越多，我们之后也应该像vulkan那样通过传入一个配置结构体的方式来创建模型
     * 这是一个优化方面，后期进行改变
     */

    models() = default;
    // 第一种方式通过传入一个面元列表来构建
    models(std::vector<primitive *> prims, int n, HitMethod m, PrimType p);
    // 第二种方式通过传入顶点数组以及索引数组来构建
    models(vertex *vertList, uint32_t *indList, uint32_t ind_len, material *mat, HitMethod m, PrimType p);
    // 第三种方式直接从模型地址导入构建
    models(const std::string model_path, material *mat, HitMethod m, PrimType p);

    virtual bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const;
    virtual bool bounding_box(float t0, float t1, aabb &box) const;
	virtual aabb getBound(void) const;
	virtual bool hasEmission(void) const { return false; };

    std::vector<primitive *> prim_list;
    int list_size;
    aabb bounds;
    bvh_tree *tree;
    HitMethod method;
    PrimType type;
};

#endif
