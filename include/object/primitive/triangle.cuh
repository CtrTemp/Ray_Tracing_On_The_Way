#pragma once
#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "object/hitable.cuh"
#include "primitive.cuh"
#include "utils/vertex.cuh"
#include "material/material.cuh"
// #include "accel/bounds.cuh"

// 质心计算 这次我们使用莱布尼茨公式进行求解
__device__ static void getBarycentricCoord(vec3 P, vec3 A, vec3 B, vec3 C, float *alpha, float *beta, float *gamma)
{
    const vec3 cP = P;
    const vec3 cA = A;

    // temp02 = -temp01;
    /**
     *  几个问题：
     *  1/ inline 函数意义是什么？
     *  2/ 函数返回值地址何时被销毁，其 return 传回的是什么东西？
     *  3/ 何时调用赋值构造函数，何时调用重载的赋值运算符
     *  4/ C++的深浅拷贝
     */
    // vec3 v0(1, 2, 3);
    // vec3 v1(1, 2, 3);
    // vec3 v2(1, 2, 3);
    vec3 v0(B.e[0] - A.e[0], B.e[1] - A.e[1], B.e[2] - A.e[2]);
    vec3 v1(C.e[0] - A.e[0], C.e[1] - A.e[1], C.e[2] - A.e[2]);
    vec3 v2(P.e[0] - A.e[0], P.e[1] - A.e[1], P.e[2] - A.e[2]);

    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float d00 = dot(v0, v0);
    float d11 = dot(v1, v1);
    float d01 = dot(v0, v1);

    float d = (d00 * d11 - d01 * d01);
    *beta = (d20 * d11 - d21 * d01) / d;
    *gamma = (d21 * d00 - d20 * d01) / d;
    *alpha = 1 - *beta - *gamma;

    // 打印输出验证
    // std::cout << "alpha = " << *alpha << "; "
    //           << "beta = " << *beta << "; "
    //           << "gamma = " << *gamma << "; "
    //           << std::endl;
}

__device__ inline float get_max_float_val(float val1, float val2)
{
    return val1 > val2 ? val1 : val2;
}
__device__ inline float get_min_float_val(float val1, float val2)
{
    return val1 > val2 ? val2 : val1;
}

// 对于 triangle 类型，重新定义关于 hitable 的派生类
class triangle : public primitive
{
public:
    __device__ triangle() = default;

    /*
        三角形面元应该支持两种构造函数重载：
        1/直接传入三个顶点信息
        2/传入顶点列表和索引缓冲区
    */
    // 第一种：传入三个顶点进行构造
    __device__ triangle(vertex v0, vertex v1, vertex v2, material *mat)
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
        // bounding_box(0, 0, bounds);

        // 在这里计算一下三角形面积
        float a = edges[0].length();
        float b = edges[1].length();
        float c = edges[2].length();

        float p = (a + b + c) / 2;

        area = sqrt(p * (p - a) * (p - b) * (p - c));
    };

    // 第二种：传入顶点列表以及索引值
    __device__ triangle(uint32_t i0, uint32_t i1, uint32_t i2, vertex *vertexList, material *mat)
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
        // bounding_box(0, 0, bounds);

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
    __device__ bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const
    {

        /*
            第一步是求解三角形平面与直线是否有交点：
            三角形所在平面当前最适合使用点法式来表示，由一般形式为：
            (x0-x, y0-y, z0-z)表示平面上任意一点与另一确定点的连线，n为法线，可知二者
        点乘结果为0，于是表示为(x0-x, y0-y, z0-z)*n = 0.进一步拆解为：
            n*P_any = n*p0
            以上p0表示平面上一确定点，P_any表示平面上任意一点。
            于是我们选取三角形任意一个顶点作为p0代入，并将直线的参数方程代入P_any，
        直线参数方程为：L(t) = Ori+t*Dir;
        */
        float t = t_max;

        const float temp_num_1 = dot((vertices[0].position - r.origin()), normal);
        const float temp_num_2 = dot(r.direction(), normal);

        t = temp_num_1 / temp_num_2;

        if (t > t_max || t < t_min)
        {
            rec.happened = false;
            return false;
        }

        vec3 current_point = r.point_at_parameter(t);

        /*
            第二步是判断当前点是否在三角形内部，实际上就是看当前点是否可以用三角形的质心
        坐标来表示。
            以下我们先用简单的向量乘法来判断，即，每个三角形顶点与已知交点的连成的向量与
        三角形三条边（按照法则规定顺序）的叉乘必须符号相同，才认为点在三角形平面内。
        */

        vec3 e_temp1 = current_point - vertices[0].position;
        vec3 e_temp2 = current_point - vertices[1].position;
        vec3 e_temp3 = current_point - vertices[2].position;

        vec3 judgeVec1 = normalized_vec(cross(edges[0], e_temp1));
        vec3 judgeVec2 = normalized_vec(cross(edges[1], e_temp2));
        vec3 judgeVec3 = normalized_vec(cross(edges[2], e_temp3));

        float judge1 = dot(judgeVec1, judgeVec2);
        float judge2 = dot(judgeVec2, judgeVec3);
        float judge3 = dot(judgeVec3, judgeVec1);

        if (judge1 > 0 && judge2 > 0 && judge3 > 0)
        {
            // std::cout << "sss" << std::endl;
            rec.t = t;
            rec.p = current_point;
            rec.normal = normal;
            rec.mat_ptr = mat_ptr;
            /*
                这里要补充记录 uv 值，从三角形的三个顶点出发，获取三个顶点的uv值，最终插值得到当前点的uv值
            */
            float alpha, beta, gamma;
            getBarycentricCoord(current_point, vertices[0].position, vertices[1].position, vertices[2].position, &alpha, &beta, &gamma);

            float u_temp = vertices[0].tex_coord[0] * alpha + vertices[1].tex_coord[0] * beta + vertices[2].tex_coord[0] * gamma;
            float v_temp = vertices[0].tex_coord[1] * alpha + vertices[1].tex_coord[1] * beta + vertices[2].tex_coord[1] * gamma;
            // float w = vertices[0].tex_coord[2] * alpha + vertices[1].tex_coord[2] * beta + vertices[2].tex_coord[2] * gamma;

            rec.u = u_temp;
            rec.v = v_temp;

            rec.happened = true;
            return true;
        }

        // std::cout << "any_ray_hit_triangle???" << std::endl;

        rec.happened = false;
        return false;
    }

    /*
        返回包围盒
    */
    // bool bounding_box(float t0, float t1, aabb &box) const;
    // virtual aabb getBound(void) const;
    __device__ virtual bool hasEmission(void) const { return mat_ptr->hasEmission(); };

    // void Sample(hit_record &pos, float &probability);
    // float getArea();

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

    // aabb bounds;
};

#endif