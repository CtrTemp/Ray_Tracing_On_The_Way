
#include "scene.h"

camera createCamera(void);

unsigned int frame_width = 512;
unsigned int frame_height = 512;

vec3 color(const ray &r, hitable *world, int depth)
{

	hit_record rec;

	if (world->hit(r, 0.001, 999999, rec)) // FLT_MAX
	{
		ray scattered;
		vec3 attenuation;
		vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
		// 在判断语句中执行并更新散射射线, 并判断是否还有射线生成
		// 同样根据材质给出衰减系数
		// 射线会在场景中最多bounce50次
		if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered))
		{
			return emitted + attenuation * color(scattered, world, depth + 1);
		}
		else
		{
			return emitted;
		}
	}
	else
	{
		// vec3 unit_direction = unit_vector(r.direction());
		// auto t = 0.5 * (unit_direction.y() + 1.0);
		// return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
		return vec3(0, 0, 0);
	}
}

int main(void)
{

	struct timeval timeStart, timeEnd, timeSystemStart;
	double runTime = 0, systemRunTime;
	gettimeofday(&timeSystemStart, NULL);
	// Linux下计时器

	camera cam = createCamera();

	std::cout << "iNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN" << std::endl;

	gettimeofday(&timeStart, NULL);
	std::string path = "../gen_pic/any.ppm";
	cam.renderFrame(camera::PresentMethod::WRITE_FILE, path);

	gettimeofday(&timeEnd, NULL);
	// 停止计时
	runTime = (timeEnd.tv_sec - timeStart.tv_sec) + (double)(timeEnd.tv_usec - timeStart.tv_usec) / 1000000;
	std::cout << ": The total time of the pirmary loop is: " << runTime << "s" << std::endl;

	std::cout << "ALL DONE" << std::endl;
	return 0;
}

camera createCamera(void)
{
	cameraCreateInfo createCamera{};

	// // general
	// vec3 lookfrom(20, 15, 20);
	// vec3 lookat(0, 0, 0);

	// // only for cornell box
	// vec3 lookfrom(278, 278, -800);
	// vec3 lookat(278, 278, 0);

	// // only for bunny
	// vec3 lookfrom(2, 1, 2);
	// vec3 lookat(0, 0, 0);

	createCamera.lookfrom = vec3(278, 278, -800);
	// createCamera.lookfrom = vec3(2, 1, 2);
	// createCamera.lookfrom = vec3(20, 15, 20);
	// createCamera.lookat = vec3(0, 0, 0);
	createCamera.lookat = vec3(278, 278, 0);

	createCamera.up_dir = vec3(0, 1, 0);
	createCamera.fov = 40;
	createCamera.aspect = float(frame_width) / float(frame_height);
	createCamera.focus_dist = 10.0;
	createCamera.t0 = 0.0;
	createCamera.t1 = 1.0;
	createCamera.frame_width = frame_width;
	createCamera.frame_height = frame_height;

	// createCamera.world = sample_light_RGB_world;
	// createCamera.world = test_triangle_world;
	// createCamera.world = test_triangleList_world;
	createCamera.world = test_Load_Models_world;
	// createCamera.world = test_image_texture_world;
	// createCamera.world = test_sky_box_world;

	// createCamera.world = test_multi_triangleList_world;
	// createCamera.world = test_Load_complex_Models_world;
	// createCamera.world = test_complex_scene_world;
	// createCamera.world = test_complex_scene_with_complex_models_world;
	createCamera.RussianRoulette = 0.8;

	// 考虑frame和这个camera的创建如何结合？
	// 学会像vulkan那样构建！！！
	return camera(createCamera);
}
