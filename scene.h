#pragma once
#ifndef SCENE
#define SCENE

#include "GlobalInclude/camera.h"
#include "GlobalInclude/hitable.h"
#include "GlobalInclude/hitableList.h"
#include "GlobalInclude/material.h"
#include "GlobalInclude/Chapter/Perlin.h"
#include "GlobalInclude/textures.h"
#include "GlobalInclude/random.h"
#include "GlobalInclude/sphere.h"
// 新加入的 triangle 类
#include "GlobalInclude/triangle.h"
#include "GlobalInclude/triangleList.h"

#include <string>

#include <iostream>
#include <fstream>
#include <random>
#include <sys/time.h>

#include "GlobalInclude/Chapter/diffuse_light.h"
#include "GlobalInclude/Chapter/box.h"

hitable *sample_light();
hitable *cornell_box();
hitable *sample_light_RGB();
hitable *test_triangle();
hitable *test_triangleList();

// // 对较大规模的三角形列表做测试
hitable *test_multi_triangleList();

// // 对模型导入做测试
hitable *test_Load_Models();
hitable *test_image_texture();

// // 天空盒测试
hitable *test_sky_box();

std::vector<hitable *> gen_sky_box_heavy(std::vector<hitable *> hit_list, int how_far);
std::vector<hitable *> gen_sky_box_fire(std::vector<hitable *> hit_list, int how_far);
std::vector<hitable *> gen_sky_box_high(std::vector<hitable *> hit_list, int how_far);

// 具有加速结构后，可以尝试复杂模型的导入并渲染
hitable *test_Load_complex_Models();

// 全局加速结构
hitable *test_complex_scene();

// 同时测试全局加速结构与模型内加速结构
hitable *test_complex_scene_with_complex_models();

#endif