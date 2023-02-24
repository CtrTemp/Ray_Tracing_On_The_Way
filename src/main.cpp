#include <stdio.h>
#include <iostream>
#include <sys/time.h>

#include <string>

#include "camera/camera.cuh"


using namespace cv;



void use_opencv_window(void);
int use_glfw_window(void);

unsigned int frame_width = 512;
unsigned int frame_height = 512;

extern __constant__ camera PRIMARY_CAMERA;

int main(void)
{

	// // 设置摄像机/场景建模
	// global_initialization();

	struct timeval timeStart, timeEnd, timeSystemStart;
	double runTime = 0, systemRunTime;
	gettimeofday(&timeSystemStart, NULL);
	// Linux下计时器

	// // 这个 camera 应该被送入常量存储区
	camera *cam = createCamera();

	std::cout << "iNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN" << std::endl;

	// std::cout << "PRIMARY_CAMERA.frame_height " << PRIMARY_CAMERA.frame_height << std::endl;

	gettimeofday(&timeStart, NULL);
	// 目前的问题是我想知道这个默认路径的相对路径的位置，它到底是相对的是哪个文件夹？！
	std::string path = "any.ppm";
	cam->renderFrame(camera::PresentMethod::WRITE_FILE, path);
	// cam.renderFrame(camera::PresentMethod::SCREEN_FLOW, path);
	

	gettimeofday(&timeEnd, NULL);
	// 停止计时
	runTime = (timeEnd.tv_sec - timeStart.tv_sec) + (double)(timeEnd.tv_usec - timeStart.tv_usec) / 1000000;
	std::cout << ": The total time of the pirmary loop is: " << runTime << "s" << std::endl;

	std::cout << "ALL DONE" << std::endl;
	return 0;
}


void use_opencv_window(void)
{
	std::string path = "any.ppm";
	cv::Mat img = imread(path);
	imshow("Image", img);
	waitKey(0); // 显示图片不会一闪而过
}
