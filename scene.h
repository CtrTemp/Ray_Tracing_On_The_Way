#pragma once
#ifndef SCENE_H
#define SCENE_H

#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include <sys/time.h>

#include "camera/camera.h"
#include "object/hitable.h"
#include "object/group/hitableList.h"
#include "object/geometry/sphere.h"
#include "object/geometry/box.h"
#include "object/primitive/triangle.h"
#include "object/primitive/primitive.h"
#include "object/model/models.h"

#include "material/material.h"
#include "material/dielectric.h"
#include "material/mental.h"
#include "material/lambertian.h"
#include "material/diffuse_light.h"

#include "texture/textures.h"
#include "texture/perlin.h"
#include "math/random.h"


extern std::vector<std::string> skybox_textures_heavy;
extern std::vector<std::string> skybox_textures_fire;
extern std::vector<std::string> skybox_textures_high;

// extern hitable_list sample_light_RGB_world;
extern hitable_list test_triangle_world;
extern hitable_list test_triangleList_world;
extern hitable_list test_Load_Models_world;
extern hitable_list test_image_texture_world;
extern hitable_list test_sky_box_world;


extern hitable_list test_multi_triangleList_world;
extern hitable_list test_Load_complex_Models_world;
extern hitable_list test_complex_scene_world;
extern hitable_list test_complex_scene_with_complex_models_world;

// hitable *cornell_box();
hitable_list cornell_box();

// hitable *sample_light_RGB();
hitable_list sample_light_RGB();


// hitable *test_triangle();
hitable_list test_triangle();

hitable_list test_triangleList();

// // 对较大规模的三角形列表做测试
// hitable *test_multi_triangleList();
hitable_list test_multi_triangleList();

// // 对模型导入做测试
// hitable *test_Load_Models();
hitable_list test_Load_Models();


// hitable *test_image_texture();
hitable_list test_image_texture();

// // 天空盒测试
// hitable *test_sky_box();
hitable_list test_sky_box();

std::vector<hitable *> gen_sky_box(std::vector<std::string> textures_path, std::vector<hitable *> hit_list, int how_far);

// 具有加速结构后，可以尝试复杂模型的导入并渲染
// hitable *test_Load_complex_Models();
hitable_list test_Load_complex_Models();

// 全局加速结构
// hitable *test_Load_complex_Models();
hitable_list test_Load_complex_Models();


hitable_list test_complex_scene();

// 同时测试全局加速结构与模型内加速结构
// hitable *test_complex_scene_with_complex_models();
hitable_list test_complex_scene_with_complex_models();

#endif