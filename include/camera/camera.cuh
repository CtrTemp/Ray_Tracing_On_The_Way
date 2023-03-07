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

#define FRAME_WIDTH 1000
#define FRAME_HEIGHT 500

#define BOUNCE_DEPTH 50

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
	__device__ __host__ camera() = default; //
	__device__ __host__ camera(cameraCreateInfo createInfo)
	{
		frame_width = createInfo.frame_width;
		frame_height = createInfo.frame_height;
		time0 = createInfo.t0;
		time1 = createInfo.t1;
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

	vec3 upper_left_conner;
	vec3 horizontal;
	vec3 vertical;
	vec3 origin;
	vec3 u, v, w;
	float lens_radius;
	float time0, time1;

	uint16_t frame_width;
	uint16_t frame_height;
	int spp;
};

__constant__ camera PRIMARY_CAMERA;

// extern "C" __host__ __device__ camera *createCamera(void);
// extern "C" __global__ void initialize_device_random(curandStateXORWOW_t *states, unsigned long long seed, size_t size);
// extern "C" __global__ void cuda_shading_unit(vec3 *frame_buffer, hitable **world, curandStateXORWOW_t *rand_state);
// extern "C" __global__ void gen_world(curandStateXORWOW_t *rand_state, hitable **world, hitable **list);

#endif // !1

/*

*/
