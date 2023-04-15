#pragma once
#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "object/hitable.cuh"
#include "utils/vertex.cuh"
#include "material/material.cuh"
#include "accel/bounds.cuh"
#include "math/common_math_device.cuh"

__device__ __host__ inline float get_max_float_val_triangle(float val1, float val2)
{
    return val1 > val2 ? val1 : val2;
}
__device__ __host__ inline float get_min_float_val_triangle(float val1, float val2)
{
    return val1 > val2 ? val2 : val1;
}

// 质心计算 这次我们使用莱布尼茨公式进行求解
__device__ static void getBarycentricCoord(vec3 P, vec3 A, vec3 B, vec3 C, float *alpha, float *beta, float *gamma)
{
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

// 对于 triangle 类型，重新定义关于 hitable 的派生类
class triangle : public hitable
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
        bounding_box(0, 0, bounds);

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

        // printf("print index = [%d,%d,%d]\n", i0, i1, i2);

        vertices[0] = vertexList[i0];
        vertices[1] = vertexList[i1];
        vertices[2] = vertexList[i2];

        edges[0] = vertices[1].position - vertices[0].position;
        edges[1] = vertices[2].position - vertices[1].position;
        edges[2] = vertices[0].position - vertices[2].position;

        normal = normalized_vec(cross(edges[0], edges[1]));

        mat_ptr = mat;
        // 获取当前三角形的包围盒，并将其传入成员变量
        bounding_box(0, 0, bounds);

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
            // 2023-04-14 深夜 startup
            // 这里的 normal 应该为三个顶点的normal的均值，我们要做一个插值处理
            // 首先一步就是在读取Obj文件的时候初始化三角形的顶点法向量
            // rec.normal = normal;
            rec.mat_ptr = mat_ptr;
            /*
                这里要补充记录 uv 值，从三角形的三个顶点出发，获取三个顶点的uv值，最终插值得到当前点的uv值
            */
            float alpha, beta, gamma;
            // 2023-04-14 深夜 如果没有使用texture mapping 其实这个计算质心坐标也无关紧要，这里也是一个优化点
            getBarycentricCoord(current_point, vertices[0].position, vertices[1].position, vertices[2].position, &alpha, &beta, &gamma);

            // // 暂时写一个，看看效率是不是会有优化？？ 结果是基本上没有优化，，，， 说明渲染的时间大头还是在射线求交，，，2023-04-15 凌晨
            // if (mat_ptr->getMaterialType() == material::SelfMaterialType::LIGHT)
            // {
            //     getBarycentricCoord(current_point, vertices[0].position, vertices[1].position, vertices[2].position, &alpha, &beta, &gamma);
            // }

            // 2023-04-15 将三角形面打击位点的法向量替换为顶点法向量的质心插值
            rec.normal = alpha * vertices[0].normal + beta * vertices[1].normal + gamma * vertices[2].normal;

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
    __device__ bool bounding_box(float t0, float t1, aabb &box) const
    {
        vec3 v0 = vertices[0].position;
        vec3 v1 = vertices[1].position;
        vec3 v2 = vertices[2].position;

        // 这里应该是一个优化点
        float max_x = get_max_float_val_triangle(get_max_float_val_triangle(v0[0], v1[0]), v2[0]);
        float max_y = get_max_float_val_triangle(get_max_float_val_triangle(v0[1], v1[1]), v2[1]);
        float max_z = get_max_float_val_triangle(get_max_float_val_triangle(v0[2], v1[2]), v2[2]);

        float min_x = get_min_float_val_triangle(get_min_float_val_triangle(v0[0], v1[0]), v2[0]);
        float min_y = get_min_float_val_triangle(get_min_float_val_triangle(v0[1], v1[1]), v2[1]);
        float min_z = get_min_float_val_triangle(get_min_float_val_triangle(v0[2], v1[2]), v2[2]);

        vec3 min_point(min_x, min_y, min_z);
        vec3 max_point(max_x, max_y, max_z);
        box = aabb(min_point, max_point);

        return true;
    }
    __device__ aabb getBound(void) const
    {
        vec3 v0 = vertices[0].position;
        vec3 v1 = vertices[1].position;
        vec3 v2 = vertices[2].position;

        float max_x = get_max_float_val_triangle(get_max_float_val_triangle(v0[0], v1[0]), v2[0]);
        float max_y = get_max_float_val_triangle(get_max_float_val_triangle(v0[1], v1[1]), v2[1]);
        float max_z = get_max_float_val_triangle(get_max_float_val_triangle(v0[2], v1[2]), v2[2]);

        float min_x = get_min_float_val_triangle(get_min_float_val_triangle(v0[0], v1[0]), v2[0]);
        float min_y = get_min_float_val_triangle(get_min_float_val_triangle(v0[1], v1[1]), v2[1]);
        float min_z = get_min_float_val_triangle(get_min_float_val_triangle(v0[2], v1[2]), v2[2]);

        vec3 min_point(min_x, min_y, min_z);
        vec3 max_point(max_x, max_y, max_z);

        return aabb(min_point, max_point);
    }
    __device__ virtual bool objHasEmission(void) const { return mat_ptr->hasEmission(0); };

    // 采样函数，对某个可求交物体，给出它表面上的一个特定坐标，并且给定取样到这个坐标的概率
    __device__ virtual void Sample(hit_record &pos, float &probability, curandStateXORWOW *states)
    {
        // 这里的sqrt要进行修改
        // std::sqrt 对应 cuda 中的 sqrt 函数
        float x = sqrt(random_float_device(states));

        float y = random_float_device(states);
        pos.p = vertices[0].position * (1.0f - x) +
                vertices[1].position * (x * (1.0f - y)) +
                vertices[2].position * (x * y);
        pos.normal = this->normal;
        probability = 1.0f / area;

        pos.mat_ptr = this->mat_ptr;

        /*
            这里要补充记录 uv 值，从三角形的三个顶点出发，获取三个顶点的uv值，最终插值得到当前点的uv值
        */
        float alpha, beta, gamma;
        getBarycentricCoord(pos.p, vertices[0].position, vertices[1].position, vertices[2].position, &alpha, &beta, &gamma);

        float u_temp = vertices[0].tex_coord[0] * alpha + vertices[1].tex_coord[0] * beta + vertices[2].tex_coord[0] * gamma;
        float v_temp = vertices[0].tex_coord[1] * alpha + vertices[1].tex_coord[1] * beta + vertices[2].tex_coord[1] * gamma;

        pos.u = u_temp;
        pos.v = v_temp;

        pos.happened = true;
    }
    // 得到目标物体的总面积
    __device__ virtual float getArea() { return area; }

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

    aabb bounds;
};

#endif