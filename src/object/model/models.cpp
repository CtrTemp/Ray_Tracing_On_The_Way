#include "object/model/models.h"
// 为什么这个只能定义在cpp文件中？？是因为它的函数定义和声明都卸载.h文件中了么
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

// 获取单个三角形的包围盒（之后要修改为更为通用的版本）
aabb surronding_box_tri(aabb box0, aabb box1)
{
    vec3 small(
        fmin(box0.min().x(), box1.min().x()),
        fmin(box0.min().y(), box1.min().y()),
        fmin(box0.min().z(), box1.min().z()));

    vec3 big(
        fmax(box0.max().x(), box1.max().x()),
        fmax(box0.max().y(), box1.max().y()),
        fmax(box0.max().z(), box1.max().z()));
    return aabb(small, big);
}

// 最暴力的算法应该是将其写成完全遍历的模式
bool models::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
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
        temp_rec = tree->getHitpoint(tree->root, r);
        if (temp_rec.happened)
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
        break;

    default:
        throw std::runtime_error("invalid iteration ergodic methods--triangle list");
        break;
    }

    return hit_anything;
}

bool models::bounding_box(float t0, float t1, aabb &box) const
{
    if (list_size < 1)
        return false;
    aabb temp_box;
    bool first_true = prim_list[0]->bounding_box(t0, t1, temp_box);
    if (!first_true)
        return false;
    else
        box = temp_box;

    for (int i = 1; i < list_size; ++i)
    {
        if (prim_list[0]->bounding_box(t0, t1, temp_box))
            box = surronding_box_tri(box, temp_box);
        else
            return false;
    }
    return true;
}

aabb models::getBound(void) const
{
    // 列表中没有单元，则返回一个无穷大的包围盒
    if (list_size < 1)
        return aabb();

    aabb bound_temp = prim_list[0]->getBound();

    for (int i = 0; i < list_size; i++)
    {
        bound_temp = Union(bound_temp, prim_list[i]->getBound());
    }

    return bound_temp;
}
// 12月30日截至点

/*
    第一种方式通过传入一个面元列表来构建
*/

models::models(std::vector<primitive *> prims, int n, HitMethod m, PrimType p)
{
    list_size = n;
    method = m;
    type = p;
    for (int i = 0; i < prims.size(); i++)
    {
        prim_list.push_back(prims[i]);
    }
    if (m == HitMethod::BVH_TREE)
    {
        tree = new bvh_tree(prim_list);
    }
    else
    {
        tree = nullptr;
    }
    bounding_box(0, 0, bounds);
}

/*
    第二种方式通过传入顶点数组以及索引数组来构建
*/
models::models(vertex *vertList, uint32_t *indList, uint32_t ind_len, material *mat, HitMethod m, PrimType p)
{
    method = m;
    type = p;
    if (p == PrimType::TRIANGLE)
    {
        for (int i = 0; i < ind_len; i += 3)
        {
            prim_list.push_back(new triangle(
                indList[i + 0], indList[i + 1], indList[i + 2],
                vertList,
                mat));
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
        throw std::runtime_error("still not support QUADRANGLE primitives");
    }

    list_size = prim_list.size();
    if (m == HitMethod::BVH_TREE)
    {
        tree = new bvh_tree(prim_list);
    }
    else
    {
        tree = nullptr;
    }
    bounding_box(0, 0, bounds);
}

/*
    第三种方式直接从模型地址导入构建
*/
models::models(const std::string module_path, material *mat, HitMethod m, PrimType p)
{

    method = m;

    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;
    /*
        从我们预定义的文件路径中读入，OBJ 文件由顶点位置信息/顶点法线信息/纹理坐标信息/表面组成，分别
    由v/vn/vt/f几个字段进行标识。
        以下使用 attrib 字段作为v/vn/vt三者的存储器，并使用attrib中的vertices/normals/texcoords
    几个字段分别指示；使用 shapes 字段作为f的存储器。
    */
    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, module_path.c_str()))
    {
        // 内置报错信息，如果有错误会自动抛出对应提示信息
        throw std::runtime_error(warn + err);
    }

    if (p == PrimType::TRIANGLE)
    {
        int primitives_len = shapes[0].mesh.indices.size() / 3;

        for (const auto &shape : shapes)
        {
            std::vector<vertex> vertList;

            // 遍历整个三角形列表，为当前列表创建整体的包围盒，能够囊括其中所有的面元
            vec3 min_vert = vec3{std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity(),
                                 std::numeric_limits<float>::infinity()};
            vec3 max_vert = vec3{-std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity(),
                                 -std::numeric_limits<float>::infinity()};
            for (const auto &index : shape.mesh.indices)
            {
                vertex vert{};
                vert.position = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]};
                vertList.push_back(vert);

                min_vert = vec3(std::min(min_vert[0], vert.position.x()),
                                std::min(min_vert[1], vert.position.y()),
                                std::min(min_vert[2], vert.position.z()));

                max_vert = vec3(std::max(max_vert[0], vert.position.x()),
                                std::max(max_vert[1], vert.position.y()),
                                std::max(max_vert[2], vert.position.z()));
            }
            for (int i = 0; i < vertList.size(); i += 3)
            {
                prim_list.push_back(new triangle(vertList[i + 0], vertList[i + 1], vertList[i + 2], mat));
            }
            // 创建包围盒
            bounds = aabb(min_vert, max_vert);
        }
    }
    else if (p == PrimType::QUADRANGLE)
    {
        throw std::runtime_error("still not support QUADRANGLE primitives");
    }

    list_size = prim_list.size();

    if (m == HitMethod::BVH_TREE)
    {
        tree = new bvh_tree(prim_list);
    }
    else
    {
        tree = nullptr;
    }

    bounding_box(0, 0, bounds);
}
