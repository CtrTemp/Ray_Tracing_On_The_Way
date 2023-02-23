#pragma once
#ifndef VERTEX_CUH
#define VERTEX_CUH

#include "utils/vec3.cuh"

class vertex
{

public:
    __device__ __host__ vertex() = default;

    // 允许用户只定义顶点坐标来确定
    __device__ __host__ vertex(vec3 p)
    {
        position = p;
        color = vec3(0, 0, 0);
        normal = vec3(0, 0, 0);
        tex_coord = vec3(0, 0, 0);
    }
    
    // 允许用户只定义顶点坐标/顶点颜色/顶点法相量来确定一个顶点
    __device__ __host__ vertex(vec3 p, vec3 c, vec3 n)
    {
        position = p;
        color = c;
        normal = n;
        tex_coord = vec3(0, 0, 0);
    }
    
    __device__ __host__ vertex(vec3 p, vec3 c, vec3 n, vec3 uvw) : position(p), color(c), normal(n), tex_coord(uvw){};

    vec3 position;
    vec3 color;
    vec3 normal;
    vec3 tex_coord;
};

#endif
