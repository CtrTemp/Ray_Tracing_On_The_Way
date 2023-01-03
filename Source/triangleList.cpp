#include "../GlobalInclude/triangleList.h"
// 为什么这个只能定义在cpp文件中？？是因为它的函数定义和声明都卸载.h文件中了么
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"
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
bool triangleList::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    double closest_so_far = t_max;
    for (int i = 0; i < list_size; i++)
    {
        if (tri_list[i]->hit(r, t_min, closest_so_far, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

bool triangleList::bounding_box(float t0, float t1, aabb &box) const
{
    if (list_size < 1)
        return false;
    aabb temp_box;
    bool first_true = tri_list[0]->bounding_box(t0, t1, temp_box);
    if (!first_true)
        return false;
    else
        box = temp_box;

    for (int i = 1; i < list_size; ++i)
    {
        if (tri_list[0]->bounding_box(t0, t1, temp_box))
            box = surronding_box_tri(box, temp_box);
        else
            return false;
    }
    return true;
}
// 12月30日截至点

/*
    通过传入顶点列表以及顶点索引列表来创建三角形列表，适用于之后的模型导入
    （是不是需要预定义一个size？）

*/
triangleList::triangleList(vertex *vertList, uint32_t *indList, uint32_t ind_len, material *mat)
{
    // material *yellow = new lambertian(new constant_texture(vec3(0.85, 0.55, 0.025)));

    triangle **tri_list_temp = new triangle *[ind_len / 3];
    int tri_index = 0;
    // 此时传入的顶点列表一定是3的倍数，按照每3个一组依次取就可以
    for (int i = 0; i < ind_len; i += 3)
    {
        // 这里我们暂时只选择一种材质（暂时选定金属材质）
        tri_list_temp[tri_index++] = new triangle(
            indList[i + 0], indList[i + 1], indList[i + 2],
            vertList,
            // new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48())
            mat);

        // std::cout << "triangle index = "
        //           << indList[i + 0] << "; "
        //           << indList[i + 1] << "; "
        //           << indList[i + 2] << "; "
        //           << std::endl;

        // std::cout << "triangle vertex = "
        //           << vertList[indList[i + 0]].position[0] << "; "
        //           << vertList[indList[i + 1]].position[0] << "; "
        //           << vertList[indList[i + 2]].position[0] << "; "
        //           << std::endl;
    }
    tri_list = tri_list_temp;
    // new triangleList(tri_list, tri_index);
    list_size = tri_index;
}

triangleList::triangleList(const std::string module_path, material *mat)
{

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

    triangle **tri_list_temp = new triangle *[shapes[0].mesh.indices.size() / 3];
    int tri_index = 0;

    // // std::cout << "vertices size = " << attrib.vertices.size() << std::endl;
    // std::cout << "shapes size = " << shapes.size() << std::endl;
    // std::cout << "shapes = "
    //           << shapes[0].mesh.indices[0].vertex_index << "; "
    //           << shapes[0].mesh.indices[1].vertex_index << "; "
    //           << shapes[0].mesh.indices[2].vertex_index << "; "
    //           << std::endl;
    /*
        遍历所有面，创建三角形，并构建列表
    */
    // 一个文件中如果只有一个对象的话，那么其子物体也就只有一个，所以以下的shapes的长度为1是正常的
    for (const auto &shape : shapes)
    {
        std::vector<vertex> vertList;
        for (const auto &index : shape.mesh.indices)
        {
            // std::cout << "tri_index = " << tri_index << std::endl;
            vertex vert{};
            vert.position = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]};
            vertList.push_back(vert);
            // std::cout << vert.position[0] << "; "
            //           << vert.position[1] << "; "
            //           << vert.position[2] << "; " << std::endl;
        }
        for (int i = 0; i < vertList.size(); i += 3)
        {
            // std::cout << "i = " << i << std::endl;
            tri_list_temp[tri_index++] = new triangle(vertList[i + 0], vertList[i + 1], vertList[i + 2], mat);
        }
        // tri_list_temp[tri_index++] = new triangle(vertList[0], vertList[1], vertList[2], mat);
    }
    tri_list = tri_list_temp;
    list_size = tri_index;
}
