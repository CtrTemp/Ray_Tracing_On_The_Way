#pragma once
#ifndef VERTEX_H
#define VERTEX_H

#include "./basic/vec3.h"

class vertex {

public:
    vertex() = default;
    vertex(vec3 p, vec3 c, vec3 n):position(p), color(c), normal(n){};

    vec3 position;
    vec3 color;
    vec3 normal;
};


#endif

