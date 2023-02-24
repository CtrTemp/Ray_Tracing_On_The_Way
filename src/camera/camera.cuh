#ifndef CAMERA_H
#define CAMERA_H

#include "utils/ray.cuh"
#include "utils/vertex.cuh"
#include "math/device_rand.cuh"

#include <math.h>
#include <vector>
#include <string>
#include <fstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define FRAME_WIDTH 512
#define FRAME_HEIGHT 512



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
	__device__ __host__ camera() = default;//
	__device__ __host__ camera(cameraCreateInfo createInfo);//
	__host__ void renderFrame(PresentMethod present, std::string file_path);//
	__host__ void showFrameFlow(int width, int height, vec3 *frame_buffer_host);//

	__host__ vec3 *cast_ray_device(float frame_width, float frame_height, int spp);

	__device__ ray get_ray_device(float s, float t, curandStateXORWOW *rand_state);
	__device__ vec3 shading_device(const ray &r);


	vec3 upper_left_conner;
	vec3 horizontal;
	vec3 vertical;
	vec3 origin;
	vec3 u, v, w;
	float lens_radius;
	float time0, time1;

	// 貌似加了这个vector就不能默认初始化
	// 原因大概是在设备端找不到 vector 的默认初始化函数，所以导致包裹vector的camera也无法初始化
	// 由于我们都为vec3这种类定义了 __device__ 版本，所以允许这种初始化
	// std::vector<vec3> frame_buffer;
	uint16_t frame_width;
	uint16_t frame_height;
	int spp;
};

__constant__ camera PRIMARY_CAMERA;

extern "C" __host__ __device__ camera *createCamera(void);

#endif // !1

/*

*/
