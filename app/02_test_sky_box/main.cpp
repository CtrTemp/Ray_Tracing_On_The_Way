
#include "scene.h"


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
	// 目前的问题是我想知道这个默认路径的相对路径的位置，它到底是相对的是哪个文件夹？！
	std::string path = "skybox.ppm";
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

	createCamera.lookfrom = Vector3f(20, 15, 20);
	createCamera.lookat = Vector3f(0, 0, 0);

	createCamera.up_dir = Vector3f(0, 1, 0);
	createCamera.fov = 40;
	createCamera.aspect = float(frame_width) / float(frame_height);
	createCamera.focus_dist = 10.0;
	createCamera.t0 = 0.0;
	createCamera.t1 = 1.0;
	createCamera.frame_width = frame_width;
	createCamera.frame_height = frame_height;

	createCamera.world = test_sky_box();
	createCamera.RussianRoulette = 0.8;
	createCamera.spp = 5;

	// 学会像vulkan那样构建
	return camera(createCamera);
}
