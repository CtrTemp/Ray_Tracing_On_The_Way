

//#include "GlobalInclude/ray.h"
//#include "GlobalInclude/vec3.h"
//当前.cpp文件应该和GlobalInclude文件夹在同一个目录下
#include "GlobalInclude/camera.h"
#include "GlobalInclude/hitable.h"
#include "GlobalInclude/hitableList.h"
#include "GlobalInclude/material.h"
#include "GlobalInclude/textures.h"
#include "GlobalInclude/random.h"
#include "GlobalInclude/sphere.h"

// 以下对新加入的 triangle 类进行测试
#include "GlobalInclude/triangle.h"

#include <string>

#include <iostream>
#include <fstream>
#include <random>
#include <sys/time.h>

#include "GlobalInclude/Chapter/diffuse_light.h"

#include "scene.h"

//关于文件包含关系的总结：
/*
	对于当前文件所在的目录，可以直接include当期目录下的所有头文件，无论当前文件是头文件还是源文件
对于当前文件下属目录下的需要加目录名再引用下属目录下的头文件；如果想包含当前目录上级目录中的文件，则需要、
使用../来引导到上级目录进行引用。

	另外，所有引用文件均可使用绝对路径来指定位置。

	默认情况下，c_cpp_properties.json文件会将根目录引导到当前工程一级目录下。

*/

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

	// vertex **vertList;

	int nx = 500;
	int ny = 500;
	int ns = 2; // Grace Jannik Remix

	int unit_percent = ny / 100;

	struct timeval timeStart, timeEnd, timeSystemStart;
	double runTime = 0, systemRunTime;
	gettimeofday(&timeSystemStart, NULL);
	// Linux下计时器

	// 首先构建世界场景
	// hitable *world = sample_light();
	// hitable *world = test_triangle();
	// hitable *world = test_multi_triangleList();
	hitable *world = test_sky_box();

	// vec3 lookfrom(30, 30, 25);
	vec3 lookfrom(20, 0, -20);
	vec3 lookat(0, 0, 0);

	// only for cornell box

	// vec3 lookfrom(278, 278, -800);
	// vec3 lookat(278, 278, 0);

	float dist_to_focus = 10.0;
	float aperture = 0.0; // MotionBlur
	// float vfov = 40.0;
	float vfov = 40.0;
	// camera cam(lookfrom, lookat, vec3(0, 1, 0), vfov, float(nx) / float(ny), aperture, dist_to_focus, 0.0, 1.0);
	// only for cornell box
	// camera cam(lookfrom, lookat, vec3(0, 1, 0), vfov, float(nx) / float(ny), aperture, dist_to_focus, 0.0, 1.0);

	std::cout << "iNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN" << std::endl;

	gettimeofday(&timeStart, NULL);
	int count = 100;
	for (size_t frame = 0; frame < count; frame++)
	{

		float look_frmo_x = 80 * cos(2 * M_PI / count * frame);
		float look_frmo_z = 80 * sin(2 * M_PI / count * frame);
		float look_frmo_y = 40 * sin(3 * 2 * M_PI / count * frame);

		vec3 lookfrom(look_frmo_x, look_frmo_y, look_frmo_z);
		vec3 lookat(0, 0, 0);
		camera cam(lookfrom, lookat, vec3(0, 1, 0), vfov, float(nx) / float(ny), aperture, dist_to_focus, 0.0, 1.0);

		/* code */
		std::ofstream OutputImage;
		// std::string Path = "/home/ctrtemp/Desktop/ss"+std::to_string(img_index)+".ppm";
		std::string Path = "./pic_flow/skybox_frame" + std::to_string(frame) + ".ppm";
		OutputImage.open(Path);
		OutputImage << "P3\n"
					<< nx << " " << ny << "\n255\n";
		// 开始计时

		for (int j = ny - 1; j >= 0; --j)
		// for (int j = 0; j < ny; ++j)
		{
			for (int i = 0; i < nx; ++i)
			// for (int i = nx-1; i >= 0; --i)
			{
				vec3 col(0, 0, 0);
				for (int s = 0; s < ns; ++s)
				{

					float u = float(i + rand() % 101 / float(101)) / float(nx);
					float v = float(j + rand() % 101 / float(101)) / float(ny);

					ray r = cam.get_ray(u, v);

					vec3 p = r.point_at_parameter(2.0);

					col += color(r, world, 0);
				}

				col /= float(ns);

				col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));

				col = color_unit_normalization(col, 1); //色值

				int ir = int(255.99 * col[0]);
				int ig = int(255.99 * col[1]);
				int ib = int(255.99 * col[2]);

				OutputImage << ir << " " << ig << " " << ib << "\n";
			}
		}

		std::cout << "frame done = " << frame << std::endl;
	}

	gettimeofday(&timeEnd, NULL);
	//停止计时
	runTime = (timeEnd.tv_sec - timeStart.tv_sec) + (double)(timeEnd.tv_usec - timeStart.tv_usec) / 1000000;
	std::cout << ": The total time of the pirmary loop is: " << runTime << "s" << std::endl;

	std::cout << "ALL DONE" << std::endl;
	return 0;
}
