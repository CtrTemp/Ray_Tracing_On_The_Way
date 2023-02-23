#ifndef GLOBAL_INIT_H
#define GLOBAL_INIT_H

#include "utils/ray.cuh"
#include "utils/vertex.cuh"
#include "camera/camera.cuh"
// 添加cuda库
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// extern camera PRIMARY_CAMERA;

// extern "C" camera createCamera(void);

extern "C" void global_initialization(void);
extern "C" camera *get_camera_info(void);
camera *createCamera(void);

// 导致这里报错的原因是你无法从这里（设备端）调用camera的构造函数
// 因为camera的构造函数是定义在host上的，对设备端不可见。这又要如何解决呢？？？？？
__constant__ camera PRIMARY_CAMERA;
__constant__ vec3 vec_temp;
// __constant__ vertex ver_temp;
// __constant__ ray ray_temp;

#endif
