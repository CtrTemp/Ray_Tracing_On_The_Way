#pragma once
#ifndef TRIANGLE_LIST_H
#define TRIANGLE_LIST_H

#include "object/primitive/triangle.cuh"
#include "object/primitive/primitive.cuh"
#include "material/material.cuh"
#include "material/lambertian.cuh"
#include "texture/textures.cuh"
// #include "accel/bvh.h"

#define TINYOBJLOADER_IMPLEMENTATION
// 我对该库进行了一些修改，在其头文件中加入了一些 static/inline 关键字
// 使得这个“header only”库可以在头文件中被包含
#include "tiny_obj_loader.h"

#include <string>
#include <vector>

// struct models_import_info{
//     std::string path;
// };

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

    __host__ __device__ models() = default;
    // 第一种方式通过传入一个面元列表来构建
    __host__ __device__ models(primitive **prims, int n, HitMethod m, PrimType p)
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
    __host__ __device__ models(vertex *vertList, uint32_t *indList, uint32_t ind_len, material *mat, HitMethod m, PrimType p)
    {
        // model_eimssion = mat->hasEmission(); // 为啥这句执行会报错？！
        model_eimssion = false;
        method = m;
        type = p;
        list_size = ind_len / 3;
        // printf("prim size = %d\n", list_size);
        // printf("ind_list = [%d,%d,%d]\n", indList[0], indList[1], indList[2]);
        if (p == PrimType::TRIANGLE)
        {
            prim_list = new primitive *[list_size];
            for (int i = 0; i < list_size; i += 1)
            {
                // printf("print index = [%d,%d,%d]\n", indList[i * 3 + 0], indList[i * 3 + 1], indList[i * 3 + 2]);
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

__host__ static void import_obj_from_file(vertex **vertList_host, int **vertex_offset, uint32_t **indList_host, int **ind_offset, std::vector<std::string> models_paths)
{

    // // // 暂时写死，之后要通过参数传入的
    // std::vector<std::string> models_paths;
    // models_paths.push_back("../Models/bunny/bunny_low_resolution.obj");
    // models_paths.push_back("../Models/bunny/bunny_low_resolution.obj");
    // models_paths.push_back("../Models/basic_geo/cuboid.obj");
    // models_paths.push_back("../Models/basic_geo/cuboid.obj");
    *vertex_offset = new int[models_paths.size() + 1];
    (*vertex_offset)[0] = 0;
    *ind_offset = new int[models_paths.size() + 1];
    (*ind_offset)[0] = 0;

    // /* ##################################### 模型读入初始化参数 ##################################### */
    // /*
    //         从我们预定义的文件路径中读入，OBJ 文件由顶点位置信息/顶点法线信息/纹理坐标信息/表面组成，分别
    //     由v/vn/vt/f几个字段进行标识。
    //         以下使用 attrib 字段作为v/vn/vt三者的存储器，并使用attrib中的vertices/normals/texcoords
    //     几个字段分别指示；使用 shapes 字段作为f的存储器。
    // */
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    // 模型面元索引数量
    int indices_len = 0;
    // 模型顶点数
    int vertices_len = 0;

    // 第一阶段进行预读取，获取整体列表长度，以便之后进行数组空间动态分配内存
    for (int model_ind = 0; model_ind < models_paths.size(); model_ind++)
    {
        std::string model_path_local = models_paths[model_ind];

        // 模型读取，且失败时报错
        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, model_path_local.c_str()))
        {
            // throw std::runtime_error(warn + err);
        }
        indices_len += shapes[0].mesh.indices.size();   // 模型面元数量
        (*ind_offset)[model_ind + 1] = indices_len;     // 模型面元偏移量
        vertices_len += attrib.vertices.size() / 3;     // 模型顶点数
        (*vertex_offset)[model_ind + 1] = vertices_len; // 模型顶点偏移量
    }

    // printf("indices_len = %d\n", indices_len);
    // printf("vertices_len = %d\n\n", vertices_len);

    // 动态内存分配
    *indList_host = new uint32_t[indices_len];
    *vertList_host = new vertex[vertices_len];
    // printf("indices offset = [%d,%d,%d]\n", (*ind_offset)[0], (*ind_offset)[1], (*ind_offset)[2]);
    // printf("vertices offset = [%d,%d,%d]\n", (*vertex_offset)[0], (*vertex_offset)[1], (*vertex_offset)[2]);
    /**
     *  正式导入数据，所有的模型顶点数据/面索引数据都将被统一导入一个相同的列表，
     * 并通过之前记录的offset列表进行模型之间的分隔
     * */
    for (int model_ind = 0; model_ind < models_paths.size(); model_ind++)
    {
        std::string module_path = models_paths[model_ind];
        // 模型读取，且失败时报错
        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, module_path.c_str()))
        {
            throw std::runtime_error(warn + err);
        }

        // std::cout << "primitives_len = " << primitives_len << std::endl;
        // std::cout << "vertices_len = " << vertices_len << std::endl;
        // for (int i = 0; i < 21; i += 3)
        // {
        //     std::cout << shapes[0].mesh.indices[i + 0].vertex_index << ","
        //               << shapes[0].mesh.indices[i + 1].vertex_index << ","
        //               << shapes[0].mesh.indices[i + 2].vertex_index << "," << std::endl;
        // }
        // for (int i = 0; i < 15; i += 3)
        // {
        //     std::cout << attrib.vertices[i + 0] << ","
        //               << attrib.vertices[i + 1] << ","
        //               << attrib.vertices[i + 2] << "," << std::endl;
        // }

        int ind_offset_local = (*ind_offset)[model_ind];
        int vert_offset_local = (*vertex_offset)[model_ind];

        // printf("ind_offset_local = %d\n", ind_offset_local);
        // printf("vert_offset_local = %d\n\n", vert_offset_local);

        for (const auto &shape : shapes)
        {
            int index_count = 0;
            int prims_len_local = shape.mesh.indices.size() / 3;
            int vert_len_local = attrib.vertices.size() / 3;

            for (int i = 0; i < prims_len_local; i++)
            {
                int vert_1_ind = shape.mesh.indices[i * 3 + 0].vertex_index;
                int vert_2_ind = shape.mesh.indices[i * 3 + 1].vertex_index;
                int vert_3_ind = shape.mesh.indices[i * 3 + 2].vertex_index;
                (*indList_host)[ind_offset_local + index_count++] = vert_1_ind;
                (*indList_host)[ind_offset_local + index_count++] = vert_2_ind;
                (*indList_host)[ind_offset_local + index_count++] = vert_3_ind;
            }

            for (int i = 0; i < vert_len_local; i++)
            {
                vertex vert{};
                vert.position = {
                    attrib.vertices[i * 3 + 0],
                    attrib.vertices[i * 3 + 1],
                    attrib.vertices[i * 3 + 2]};
                (*vertList_host)[vert_offset_local + i] = vert;
            }
        }
    }
}

#endif
