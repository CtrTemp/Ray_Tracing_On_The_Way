#include <stdio.h>
#include <iostream>
#include <sys/time.h>

#include <string>

#include "foo.cuh"

#include "utils/ray.cuh"
#include "utils/vertex.cuh"
#include "camera/camera.h"

#include "GLFW/glfw3.h"

using namespace cv;

/**
 *  这部分主要讲一下混合编程的问题。
 *
 *  首先，我们用的是C++与CUDA混合编程，体现在编译就需要两套编译标准g++与nvcc。它们将分别执行对.cpp和.cu文件
 * 的编译。
 *
 *  我们希望的是主程序的逻辑部分使用C++代码进行书写，并且在需要进行GPU加速的部分调用CUDA程序，也就是逻辑部分
 * 交给C++其余并行计算部分交给CUDA。那么这就不可避免的造成我们要以C++作为主程序，并调用.cu文件中的CUDA程序。
 * 但是在编译过程中，由于.cu文件中的的编译一般遵循C标准，这使得我们在头文件中对函数进行声明时，要使用“extern
 * 'C'”标志，从而告知g++在对其进行编译时，使用c++标准，从而使得程序在最终的链接阶段可以根据名称找到正确的函数
 * 定义。
 *
 *  这里要提到的一个点就是，C与C++的编译不同，由于C++支持函数重载，使得其编译得到的汇编文件中的函数名称带有附加
 * 的字符来区分重载函数的不同参数，但C编译出来的汇编文件中的函数就只有函数名（因为C不支持函数重载也不需要额外的
 * 字符进行函数区分）。这就导致使用gcc和g++编译出来的文件中的函数不能互相调用，因为虽然函数是同一个函数但编译
 * 出的函数名却不能互相识别。就是以上的问题导致我们在C++与C的混编模式下必须使用extern "C"来声明一个“C++程序
 * 要调用的来自C文件中定义的函数”，使得g++在编译的时候按照C标准生成汇编文件中的函数名。
 *
 *  CUDA的nvcc可能是按照C的标准来进行编译以及函数命名的，所以在与C++混编时，应该也遵从与C混编的原则，就是也需要
 * 使用extern "C"来对头文件中在.cu文件中定义的函数进行声明
 *
 * */

camera createCamera(void);
int use_opencv_window(void);
int use_glfw_window(void);

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
	std::string path = "any.ppm";
	// cam.renderFrame(camera::PresentMethod::WRITE_FILE, path);
	cam.renderFrame(camera::PresentMethod::SCREEN_FLOW, path);

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

	createCamera.lookfrom = vec3(20, 15, 20);
	createCamera.lookat = vec3(0, 0, 0);

	createCamera.up_dir = vec3(0, 1, 0);
	createCamera.fov = 40;
	createCamera.aspect = float(frame_width) / float(frame_height);
	createCamera.focus_dist = 10.0;
	createCamera.t0 = 0.0;
	createCamera.t1 = 1.0;
	createCamera.frame_width = frame_width;
	createCamera.frame_height = frame_height;

	createCamera.spp = 1;

	// 学会像vulkan那样构建
	return camera(createCamera);
}

int use_glfw_window(void)
{

	GLFWwindow *window;

	/* Initialize the library */
	if (!glfwInit())
		return -1;

	/* Create a windowed mode window and its OpenGL context */
	window = glfwCreateWindow(480, 320, "Hello World", NULL, NULL);
	if (!window)
	{
		glfwTerminate();
		return -1;
	}

	/* Make the window's context current */
	glfwMakeContextCurrent(window);

	/* Loop until the user closes the window */
	while (!glfwWindowShouldClose(window))
	{
		/* Draw a triangle */
		glBegin(GL_TRIANGLES);

		glColor3f(1.0, 0.0, 0.0); // Red
		glVertex3f(0.0, 1.0, 0.0);

		glColor3f(0.0, 1.0, 0.0); // Green
		glVertex3f(-1.0, -1.0, 0.0);

		glColor3f(0.0, 0.0, 1.0); // Blue
		glVertex3f(1.0, -1.0, 0.0);

		glEnd();

		/* Swap front and back buffers */
		glfwSwapBuffers(window);

		/* Poll for and process events */
		glfwPollEvents();
	}

	glfwTerminate();
}

int use_opencv_window(void)
{
	std::string path = "any.ppm";
	cv::Mat img = imread(path);
	imshow("Image", img);
	waitKey(0); // 显示图片不会一闪而过

	return 0;
}
