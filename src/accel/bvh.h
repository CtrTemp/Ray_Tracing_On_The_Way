#pragma once
#ifndef BVH_TREE_H
#define BVH_TREE_H

#include "accel/bounds.h"
#include "object/primitive/triangle.h"
#include "object/primitive/primitive.h"
#include <vector>

class bvh_node
{
public:
    bvh_node()
    {
        bound = aabb(); // 包围盒初始化为无限大
        left = nullptr;  // 左右子节点均初始化指向空
        right = nullptr;
        object = nullptr;
    }

    // int splitAxis = 0, firstPrimOffset = 0, nPrimitives = 0;
    aabb bound;
    bvh_node *left;
    bvh_node *right;
    // node 节点的主体可以是多种多样的
    primitive *object;
};

class bvh_tree
{
public:
    bvh_tree() = default;
    bvh_tree(std::vector<primitive *> tri_list, int maxPrimsInNode = 1);
    bvh_node *recursiveConstructTree(std::vector<primitive *> primitives);
    hit_record Intersect(const ray &ray) const;
    hit_record getHitpoint(bvh_node *node, const ray &ray) const;

    const int maxPrimsInNode; // 常量只初始化一次, 定义当前BVH节点所能容纳最大三角形面片数
    // 当前BVH加速结构所囊括的三角形面片组
    // 这里应该作出改变以适应不同的情况，首先应该适应传入 hitableList 的情况，为世界坐标系中的不同物体构建树状结构
    std::vector<primitive *> primitives;
    bvh_node *root;
};


class bvh_node_scene
{
public:
    bvh_node_scene()
    {
        bound = aabb(); // 包围盒初始化为无限大
        left = nullptr;  // 左右子节点均初始化指向空
        right = nullptr;
        object = nullptr;
    }

    // int splitAxis = 0, firstPrimOffset = 0, nPrimitives = 0;
    aabb bound;
    bvh_node_scene *left;
    bvh_node_scene *right;
    // node 节点的主体可以是多种多样的
    hitable *object;
};


class bvh_tree_scene
{
public:
    bvh_tree_scene() = default;
    bvh_tree_scene(std::vector<hitable *> tri_list, int maxPrimsInNode = 1);
    bvh_node_scene *recursiveConstructSceneTree(std::vector<hitable *> primitives);
    hit_record Intersect(const ray &ray) const;
    hit_record getHitpoint(bvh_node_scene *node, const ray &ray) const;

    const int maxPrimsInNode; // 常量只初始化一次, 定义当前BVH节点所能容纳最大三角形面片数
    // 当前BVH加速结构所囊括的三角形面片组
    // 这里应该作出改变以适应不同的情况，首先应该适应传入 hitableList 的情况，为世界坐标系中的不同物体构建树状结构
    std::vector<hitable *> obj_list;
    bvh_node_scene *root;
};

#endif
