#ifndef CAMERA_H
#define CAMERA_H

#include "utils/ray.cuh"

#include <math.h>
#include <vector>
#include <string>
#include <fstream>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

typedef struct
{
	vec3 lookfrom;
	vec3 lookat;
	vec3 up_dir;
	float fov;
	float aspect;
	float aperture;
	float focus_dist;
	float t0;
	float t1;
	uint16_t frame_width;
	uint16_t frame_height;
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
	vec3 shading(uint16_t depth, // 最大bounce递归深度
								 //  bool RussianRoulette, // 是否采用俄罗斯轮盘赌的方式终止光线bounce，为false时当达到最大递归深度则终止
				 const ray &r);

	void renderFrame(PresentMethod present, std::string file_path);
	void showFrameFlow(int width, int height, float *frame_buffer_host);

	vec3 upper_left_conner;
	vec3 horizontal;
	vec3 vertical;
	vec3 origin;
	vec3 u, v, w;
	float lens_radius;
	float time0, time1;

	std::vector<vec3> frame_buffer;
	uint16_t frame_width;
	uint16_t frame_height;
	int spp;
};

#endif // !1

/*

*/
