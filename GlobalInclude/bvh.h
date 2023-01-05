#pragma once
#ifndef BVH_TREE_H
#define BVH_TREE_H

#include "bounds.h"
#include "triangle.h"
#include "vector"

class bvh_node
{
public:
    bvh_node()
    {
        bounds = aabb(); // 包围盒初始化为无限大
        left = nullptr;  // 左右子节点均初始化指向空
        right = nullptr;
        object = nullptr;
    }

    // int splitAxis = 0, firstPrimOffset = 0, nPrimitives = 0;
    aabb bounds;
    bvh_node *left;
    bvh_node *right;
    triangle *object;
};

class bvh_tree
{
public:
    bvh_tree() = default;
    bvh_tree(std::vector<triangle *> tri_list, int maxPrimsInNode = 1);
    bvh_node *recursiveConstructTree(std::vector<triangle *> primitives);
    hit_record Intersect(const ray &ray) const;
    hit_record getHitpoint(bvh_node* node, const ray& ray)const;

    const int maxPrimsInNode; // 常量只初始化一次, 定义当前BVH节点所能容纳最大三角形面片数
    std::vector<triangle *> primitives;    // 当前BVH加速结构所囊括的三角形面片组
    bvh_node *root;
};

#endif
