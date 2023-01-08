
#include "scene.h"


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
		vec3 unit_direction = unit_vector(r.direction());
		auto t = 0.5 * (unit_direction.y() + 1.0);
		return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
		// return vec3(0, 0, 0);
	}
}

int main(void)
{

	// vertex **vertList;

	int nx = 200;
	int ny = 200;
	int ns = 1; // Grace Jannik Remix

	int unit_percent = ny / 100;

	struct timeval timeStart, timeEnd, timeSystemStart;
	double runTime = 0, systemRunTime;
	gettimeofday(&timeSystemStart, NULL);
	// Linux下计时器

	// 首先构建世界场景
	// hitable *world = sample_light_RGB();
	// hitable *world = test_triangle();
	// hitable *world = test_triangleList();
	// hitable *world = test_Load_Models();
	// hitable *world = test_image_texture();
	// hitable *world = test_sky_box();
  
	// hitable *world = test_multi_triangleList();
	// hitable *world = test_Load_complex_Models();
	// hitable *world = test_complex_scene();
	hitable *world = test_complex_scene_with_complex_models();

	// vec3 lookfrom(30, 30, 25);
	vec3 lookfrom(20, 4, 6);
	// vec3 lookfrom(50, 30, 50);
	// vec3 lookfrom(2.5, 1.25, 2.5);
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
	int count = 1;
	for (size_t frame = 0; frame < count; frame++)
	{

		// float look_frmo_x = 2.5 * cos(2 * M_PI / count * frame);
		// float look_frmo_z = 2.5 * sin(2 * M_PI / count * frame);
		// float look_frmo_y = 1.5 * sin(3 * 2 * M_PI / count * frame);

		// vec3 lookfrom(look_frmo_x, look_frmo_y, look_frmo_z);
		// vec3 lookat(0, 0, 0);
		camera cam(lookfrom, lookat, vec3(0, 1, 0), vfov, float(nx) / float(ny), aperture, dist_to_focus, 0.0, 1.0);

		/* code */
		std::ofstream OutputImage;
		// std::string Path = "/home/ctrtemp/Desktop/ss"+std::to_string(img_index)+".ppm";
		std::string Path = "./pic_flow/bunny_flow_" + std::to_string(frame) + ".ppm";
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
