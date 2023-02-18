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


hitable_list sample_light_RGB();


#endif