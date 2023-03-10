#pragma once
#ifndef TRIANGLE_LIST_H
#define TRIANGLE_LIST_H

#include "object/primitive/triangle.cuh"
#include "object/primitive/primitive.cuh"
#include "material/material.cuh"
// #include "accel/bvh.h"

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

    __device__ models() = default;
    // 第一种方式通过传入一个面元列表来构建
    __device__ models(primitive **prims, int n, HitMethod m, PrimType p)
    {
        model_eimssion = false;
        list_size = n;
        method = m;
        type = p;
        prim_list = prims;
        // for (int i = 0; i < list_size; i++)
        // {
        //     // 如果是自发光基元，则应该将其倒入发光基元列表
        //     // if (prims[i]->hasEmission())
        //     // {
        //     //     emit_prim_list.push_back(prims[i]);
        //     // }
        // }
        // if (emit_prim_list.size() >= 1)
        // {
        //     model_eimssion = true;
        // }
        // if (m == HitMethod::BVH_TREE)
        // {
        //     tree = new bvh_tree(prim_list);
        // }
        // else
        // {
        //     tree = nullptr;
        // }
        // bounding_box(0, 0, bounds);
    }
    // 第二种方式通过传入顶点数组以及索引数组来构建
    __device__ models(vertex *vertList, uint32_t *indList, uint32_t ind_len, material *mat, HitMethod m, PrimType p)
    {
        // model_eimssion = mat->hasEmission(); // 为啥这句执行会报错？！
        model_eimssion = false;
        method = m;
        type = p;
        list_size = ind_len / 3;

        if (p == PrimType::TRIANGLE)
        {
            prim_list = new primitive *[list_size];
            for (int i = 0; i < list_size; i += 1)
            {
                primitive *prim_unit = new triangle(
                    indList[i * 3 + 0], indList[i * 3 + 1], indList[i * 3 + 2],
                    vertList,
                    mat);
                prim_list[i] = prim_unit;
                // if (model_eimssion)
                // {
                //     emit_prim_list.push_back(prim_unit);
                // }
            }
        }
        else if (p == PrimType::QUADRANGLE)
        {
            // for (int i = 0; i < ind_len; i += 4)
            // {
            //     prim_list.push_back(new triangle(
            //         indList[i + 0], indList[i + 1], indList[i + 2],
            //         vertList,
            //         mat));
            // }
            // throw std::runtime_error("still not support QUADRANGLE primitives");
        }

        // list_size = prim_list.size();
        // if (m == HitMethod::BVH_TREE)
        // {
        //     tree = new bvh_tree(prim_list);
        // }
        // else
        // {
        //     tree = nullptr;
        // }
        // bounding_box(0, 0, bounds);
    }
    // 第三种方式直接从模型地址导入构建（暂时不支持这种）
    __device__ models(const std::string model_path, material *mat, HitMethod m, PrimType p)
    {
        // model_eimssion = mat->hasEmission();

        // method = m;

        // tinyobj::attrib_t attrib;
        // std::vector<tinyobj::shape_t> shapes;
        // std::vector<tinyobj::material_t> materials;
        // std::string warn, err;
        // /*
        //     从我们预定义的文件路径中读入，OBJ 文件由顶点位置信息/顶点法线信息/纹理坐标信息/表面组成，分别
        // 由v/vn/vt/f几个字段进行标识。
        //     以下使用 attrib 字段作为v/vn/vt三者的存储器，并使用attrib中的vertices/normals/texcoords
        // 几个字段分别指示；使用 shapes 字段作为f的存储器。
        // */
        // if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, module_path.c_str()))
        // {
        //     // 内置报错信息，如果有错误会自动抛出对应提示信息
        //     throw std::runtime_error(warn + err);
        // }

        // if (p == PrimType::TRIANGLE)
        // {
        //     int primitives_len = shapes[0].mesh.indices.size() / 3;

        //     for (const auto &shape : shapes)
        //     {
        //         std::vector<vertex> vertList;

        //         // 遍历整个三角形列表，为当前列表创建整体的包围盒，能够囊括其中所有的面元
        //         vec3 min_vert = vec3{std::numeric_limits<float>::infinity(),
        //                              std::numeric_limits<float>::infinity(),
        //                              std::numeric_limits<float>::infinity()};
        //         vec3 max_vert = vec3{-std::numeric_limits<float>::infinity(),
        //                              -std::numeric_limits<float>::infinity(),
        //                              -std::numeric_limits<float>::infinity()};
        //         for (const auto &index : shape.mesh.indices)
        //         {
        //             vertex vert{};
        //             vert.position = {
        //                 attrib.vertices[3 * index.vertex_index + 0],
        //                 attrib.vertices[3 * index.vertex_index + 1],
        //                 attrib.vertices[3 * index.vertex_index + 2]};
        //             vertList.push_back(vert);

        //             min_vert = vec3(std::min(min_vert[0], vert.position.x()),
        //                             std::min(min_vert[1], vert.position.y()),
        //                             std::min(min_vert[2], vert.position.z()));

        //             max_vert = vec3(std::max(max_vert[0], vert.position.x()),
        //                             std::max(max_vert[1], vert.position.y()),
        //                             std::max(max_vert[2], vert.position.z()));
        //         }
        //         for (int i = 0; i < vertList.size(); i += 3)
        //         {
        //             primitive *prim_unit = new triangle(vertList[i + 0], vertList[i + 1], vertList[i + 2], mat);
        //             prim_list.push_back(prim_unit);
        //             if (model_eimssion)
        //             {
        //                 emit_prim_list.push_back(prim_unit);
        //             }
        //         }
        //         // // 创建包围盒
        //         // bounds = aabb(min_vert, max_vert);
        //     }
        // }
        // else if (p == PrimType::QUADRANGLE)
        // {
        //     // throw std::runtime_error("still not support QUADRANGLE primitives");
        // }

        // list_size = prim_list.size();

        // if (m == HitMethod::BVH_TREE)
        // {
        //     tree = new bvh_tree(prim_list);
        // }
        // else
        // {
        //     tree = nullptr;
        // }

        // bounding_box(0, 0, bounds);
    }

    __device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const
    {
        hit_record temp_rec;
        bool hit_anything = false;
        double closest_so_far = t_max;

        switch (method)
        {
        // 朴素暴力遍历法求解交点
        case HitMethod::NAIVE:
            for (int i = 0; i < list_size; i++)
            {
                if (prim_list[i]->hit(r, t_min, closest_so_far, temp_rec))
                {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }
            break;

        // 使用树装加速结构求解交点
        case HitMethod::BVH_TREE:
            // std::cout << "tree hit" << std::endl;
            // temp_rec = tree->getHitpoint(tree->root, r);
            // if (temp_rec.happened)
            // {
            //     hit_anything = true;
            //     closest_so_far = temp_rec.t;
            //     rec = temp_rec;
            // }
            // throw std::runtime_error("not support currently");
            break;

        default:
            // throw std::runtime_error("invalid iteration ergodic methods--triangle list");
            break;
        }

        return hit_anything;
    }
    // virtual bool bounding_box(float t0, float t1, aabb &box) const;
    // virtual aabb getBound(void) const;

    // // 这里我们将模型视为一个整体，要么整体发光，要么都不发光，不存在内嵌有部分发光的情况
    // virtual bool hasEmission(void) const { return false; };
    __device__ virtual bool hasEmission(void) const { return model_eimssion; };

    // void Sample(hit_record &pos, float &probability);
    // float getArea();

    primitive **prim_list;
    // primitive ** emit_prim_list;
    bool model_eimssion;
    int list_size;
    // aabb bounds;
    // bvh_tree *tree;
    HitMethod method;
    PrimType type;
};

#endif
