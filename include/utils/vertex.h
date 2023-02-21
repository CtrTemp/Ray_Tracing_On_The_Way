#pragma once
#ifndef VERTEX_H
#define VERTEX_H

#include "utils/vec3.h"

class vertex
{

public:
    vertex() = default;

    // 允许用户只定义顶点坐标来确定
    vertex(Vector3f p)
    {
        position = p;
        color = Vector3f(0, 0, 0);
        normal = Vector3f(0, 0, 0);
        tex_coord = Vector3f(0, 0, 0);
    }
    
    // 允许用户只定义顶点坐标/顶点颜色/顶点法相量来确定一个顶点
    vertex(Vector3f p, Vector3f c, Vector3f n)
    {
        position = p;
        color = c;
        normal = n;
        tex_coord = Vector3f(0, 0, 0);
    }
    
    // 
    vertex(Vector3f p, Vector3f c, Vector3f n, Vector3f uvw) : position(p), color(c), normal(n), tex_coord(uvw){};

    Vector3f position;
    Vector3f color;
    Vector3f normal;
    Vector3f tex_coord;
};

#endif
