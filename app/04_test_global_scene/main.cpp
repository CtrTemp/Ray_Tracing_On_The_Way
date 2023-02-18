
#include "scene.h"

/**
 * 	总结：目前进展来看，虽然渲染方程还存在一定的偏差错误，但已经可以先不进行图像质量的优化了，下一步应该
 * 尽快完成进入下一阶段急需的一些必要工作，如下：
 *
 * 	1.完成一个复杂模型的渲染，并给其赋予贴图。这部分主要是考察复杂模型+贴图引入
 *  2.完成模型的坐标变换函数组。这部分需要你将整个工程中涉及矩阵计算的部分全部替换成第三方库，而非目前
 * 个人书写的库，方便之后的矩阵坐标变换（缩放/旋转/平移）的应用。
 *
 *  以上这些完成后便可以开始进入下一阶段：并行程序加速，真正使用CUDA完成并行计算，使得程序速度得到提升
 * 真正变成可以完成实时渲染的程序。
 *
 *
 * 	2023-02-16
 * 	今天将程序进行“清洗”，抹除不必要的注释，以及多余的测试代码
 *
 * */

camera createCamera(void);

unsigned int frame_width = 512;
unsigned int frame_height = 512;


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

	// createCamera.lookfrom = vec3(278, 278, -800);
	// createCamera.lookfrom = vec3(2, 1, 2);
	createCamera.lookfrom = vec3(20, 15, 20);
	createCamera.lookat = vec3(0, 0, 0);
	// createCamera.lookat = vec3(278, 278, 0);

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
	// createCamera.world = test_Load_Models_world;
	// createCamera.world = test_image_texture_world;
	createCamera.world = test_sky_box_world;

	// createCamera.world = test_multi_triangleList_world;
	// createCamera.world = test_Load_complex_Models_world;
	// createCamera.world = test_complex_scene_world;
	// createCamera.world = test_complex_scene_with_complex_models_world;
	createCamera.RussianRoulette = 0.8;

	// 学会像vulkan那样构建
	return camera(createCamera);
}
