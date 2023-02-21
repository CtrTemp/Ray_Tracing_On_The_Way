#ifndef CAMERA_H
#define CAMERA_H

#include "utils/ray.h"
#include "object/hitable.h"
#include "material/material.h"
#include "object/group/hitableList.h"

#include <math.h>
#include <vector>
#include <string>
#include <fstream>

// #define M_PI	acos(-1)

// 用于在圆盘光孔平面中模拟射入光孔的光线
Vector3f random_in_unit_disk();

typedef struct
{
	Vector3f lookfrom;
	Vector3f lookat;
	Vector3f up_dir;
	float fov;
	float aspect;
	float aperture;
	float focus_dist;
	float t0;
	float t1;
	uint16_t frame_width;
	uint16_t frame_height;
	hitable_list world;
	float RussianRoulette;
	int spp;

} cameraCreateInfo;

class camera
{

public:
	enum class RayDistribution
	{
		NAIVE_RANDOM
	};

	enum class PresentMethod
	{
		WRITE_FILE,
		SCREEN_FLOW
	};

public:
	camera(cameraCreateInfo createInfo);
	ray get_ray(float s, float t);

	// 2023-01-11
	// 之后我们将 shade 着色函数写在camera类中
	void cast_ray(
		uint16_t spp,			   // 每个像素投射光线数量
		RayDistribution distribute // 光线投射在像素中的分布函数
	);
	// 单一光线射出，在场景中bounce后返回着色结果
	Vector3f shading(uint16_t depth, // 最大bounce递归深度
								 //  bool RussianRoulette, // 是否采用俄罗斯轮盘赌的方式终止光线bounce，为false时当达到最大递归深度则终止
				 const ray &r);

	void renderFrame(PresentMethod present, std::string file_path);

	void sampleLight(hit_record &pos, float &pdf);

	Vector3f upper_left_conner;
	Vector3f horizontal;
	Vector3f vertical;
	Vector3f origin;
	Vector3f u, v, w;
	float lens_radius;
	float time0, time1;

	// 2023-01-11 新加入成员变量 framebuffer/framesize
	std::vector<Vector3f> frame_buffer;
	uint16_t frame_width;
	uint16_t frame_height;
	hitable_list world;
	float RussianRoulette = 0;
	int spp;
};

#endif // !1

/*

*/
