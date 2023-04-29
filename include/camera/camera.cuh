#ifndef CAMERA_H
#define CAMERA_H

#include "utils/ray.cuh"
#include "utils/vertex.cuh"
#include "math/device_rand.cuh"

#include <math.h>
#include <vector>
#include <string>
#include <fstream>

// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/imgproc.hpp>

#include "object/hitable.cuh"
#include "object/group/hitable_list.cuh"

// 1080P 1280 720
// 2K 1920 1080
// 4K 3840 2160

#define FRAME_WIDTH 1280
#define FRAME_HEIGHT 720

#define BOUNCE_DEPTH 8

typedef struct
{
	vec3 lookfrom;
	vec3 lookat;
	vec3 up_dir;
	float fov;
	float aspect;
	float aperture;
	float focus_dist;
	float RussianRoulette;
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

	// 用于确定输出源
	enum class OutputBuffer
	{
		FRAME_BUFFER,
		DEPTH_BUFFER
	};

public:
	__device__ __host__
	camera() = default; //
	__device__ __host__ camera(cameraCreateInfo createInfo)
	{
		frame_width = createInfo.frame_width;
		frame_height = createInfo.frame_height;
		time0 = createInfo.t0;
		time1 = createInfo.t1;
		RussianRoulette = createInfo.RussianRoulette;
		lens_radius = createInfo.aperture / 2;
		float theta = createInfo.fov * M_PI / 180;
		float half_height = tan(theta / 2);
		float half_width = createInfo.aspect * half_height;

		origin = createInfo.lookfrom;

		w = unit_vector(createInfo.lookat - createInfo.lookfrom); // view_ray direction
		u = unit_vector(cross(w, createInfo.up_dir));			  // camera plane horizontal direction vec
		v = cross(w, u);										  // camera plane vertical direction vec

		// lower_left_conner = origin + focus_dist * w - half_width * focus_dist * u - half_height * focus_dist * v;

		// 我们应该定义的是左上角而不是左下角
		upper_left_conner = origin +
							createInfo.focus_dist * w -
							half_width * createInfo.focus_dist * u -
							half_height * createInfo.focus_dist * v;
		horizontal = 2 * half_width * createInfo.focus_dist * u;
		vertical = 2 * half_height * createInfo.focus_dist * v;

		spp = createInfo.spp;
	};
	// __host__ void renderFrame(PresentMethod present, std::string file_path); //
	// __host__ void showFrameFlow(int width, int height, vec3 *frame_buffer_host); //

	// __host__ vec3 *cast_ray_device(float frame_width, float frame_height, int spp);

	// __device__ ray get_ray_device(float s, float t, curandStateXORWOW *rand_state);
	// __device__ vec3 shading_device(int depth, ray &r, hitable **world, curandStateXORWOW_t *rand_state);

	__device__ void sampleLight(hit_record &pos, float &pdf, hitable_list **world, curandStateXORWOW *rand_state)
	{
		float emit_area_sum = 0;
		for (uint32_t k = 0; k < (*world)->list_size; k++)
		{
			if ((*world)->list[k]->objHasEmission())
			{
				emit_area_sum += (*world)->list[k]->getArea();
				// std::cout << "get an emission" << std::endl;
			}
			// std::cout << "k = " << k << std::endl;
		}

		// printf("total emit_area = %f\n", emit_area_sum);

		// 2023/02/06
		// 现在问题已经很明确了：没有光源或者只有一个光源的情况都不适用！！！你的系统无法兼容这种情况
		if (emit_area_sum == 0)
		{
			// printf("there is no light source in this scene! please check your world construction！\n");
		}

		// std::cout << "total area = " << emit_area_sum << std::endl;

		float p = random_float_device(rand_state) * emit_area_sum;
		// std::cout << "p = " << p << std::endl;
		emit_area_sum = 0;
		for (uint32_t k = 0; k < (*world)->list_size; k++)
		{
			if ((*world)->list[k]->objHasEmission())
			{
				emit_area_sum += (*world)->list[k]->getArea();
				if (p <= emit_area_sum)
				{
					// std::cout << "current kk = " << k << std::endl;
					(*world)->list[k]->Sample(pos, pdf, rand_state);
					// printf("pdf = %f\n", pdf);
					break;
				}
			}
		}
	}

	vec3 upper_left_conner;
	vec3 horizontal;
	vec3 vertical;
	vec3 origin;
	vec3 u, v, w;
	float lens_radius;
	float time0, time1;
	float RussianRoulette;

	uint16_t frame_width;
	uint16_t frame_height;
	int spp;
};

__constant__ camera PRIMARY_CAMERA;

// extern "C" __host__ __device__ camera *createCamera(void);
// extern "C" __global__ void initialize_device_random(curandStateXORWOW_t *states, unsigned long long seed, size_t size);
// extern "C" __global__ void cuda_shading_unit(vec3 *frame_buffer, hitable **world, curandStateXORWOW_t *rand_state);
// extern "C" __global__ void gen_world(curandStateXORWOW_t *rand_state, hitable **world, hitable **list);

__host__ static camera *createCamera(cameraCreateInfo camInfo)
{
	// createCamera.lookfrom = vec3(-2, 2, 1);
	// createCamera.lookat = vec3(0, 0, -1);
	// createCamera.lookfrom = vec3(10, 8, 10);
	// createCamera.lookat = vec3(0, 1, 0);
	// // 纹理贴图最佳视点
	// createCamera.lookfrom = vec3(4, 2, 4);
	// createCamera.lookat = vec3(0, 1, 0);
	// // bunny模型导入最佳视点
	// createCamera.lookfrom = vec3(2, 2, 2);
	// createCamera.lookat = vec3(0.25, 0, 0.25);
	// skybox 测试视点

	// 学会像vulkan那样构建
	// return new camera(createCamera);
}

// __host__ static void modifyCamera(camera *cam, cameraCreateInfo camInfo, size_t frame_counts)
__host__ static camera *modifyCamera(cameraCreateInfo camInfo, size_t frame_counts)
{
	int frame_yaw_period = 400;	  // 圆周 偏航角(yaw) 周期
	int frame_pitch_period = 100; // 圆周 俯仰角(pitch) 周期
	float cam_yaw_range = 3;
	float cam_pitch_range = 0.5;
	float x_coord = cam_yaw_range * sin(frame_counts * 2 * M_PI / frame_yaw_period);
	float z_coord = cam_yaw_range * cos(frame_counts * 2 * M_PI / frame_yaw_period);
	float y_coord = camInfo.lookfrom.e[1] + cam_pitch_range * sin(frame_counts * 2 * M_PI / frame_pitch_period);

	camInfo.lookfrom = vec3(x_coord, y_coord, z_coord);

	// camera *new_cam = new camera(camInfo);
	// cam = new_cam;
	return new camera(camInfo);

	// cam->origin = vec3(x_coord, y_coord, z_coord);
	// cam->w = unit_vector(camInfo.lookat - camInfo.lookfrom); // view_ray direction
	// cam->u = unit_vector(cross(cam->w, camInfo.up_dir));	 // camera plane horizontal direction vec
	// cam->v = cross(cam->w, cam->u);
}

#endif // !1

/*

*/
