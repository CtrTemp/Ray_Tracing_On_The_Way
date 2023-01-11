#include "camera/camera.h"

camera::camera(cameraCreateInfo createInfo)
{
	// vec3 u, v, w;
	world = createInfo.world;
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
}

vec3 random_in_unit_disk()
{
	vec3 p;
	do
	{
		p = 2.0 * vec3(drand48(), drand48(), 0) - vec3(1, 1, 0);
	} while (dot(p, p) >= 1.0);
	//模拟在方格中撒点，掉入圆圈的点被收录返回
	return p;
}

ray camera::get_ray(float s, float t)
{
	vec3 rd = lens_radius * random_in_unit_disk(); //得到设定光孔大小内的任意散点（即origin点——viewpoint）
	//（该乘积的后一项是单位光孔）
	vec3 offset = rd.x() * u + rd.y() * v; // origin视点中心偏移（由xoy平面映射到u、v平面）
	// return ray(origin + offset, lower_left_conner + s*horizontal + t*vertical - origin - offset);
	float time = time0 + drand48() * (time1 - time0);
	return ray(origin + offset, upper_left_conner + s * horizontal + t * vertical - origin - offset, time);
}

// 规定从左上角遍历到右下角，行优先遍历
void camera::cast_ray(uint16_t spp, RayDistribution distribute)
{
	for (int row = 0; row < frame_height; row++)
	{
		for (int col = 0; col < frame_width; col++)
		{

			vec3 pixel(0, 0, 0);
			for (int s = 0; s < spp; s++)
			{
				float u, v;
				switch (distribute)
				{
				case RayDistribution::NAIVE_RANDOM:
					// 这里我们默认使用了一般的射线在亚像素级的分布方式
					u = float(col + rand() % 101 / float(101)) / float(this->frame_width);
					v = float(row + rand() % 101 / float(101)) / float(this->frame_height);
					break;

				default:
					throw std::runtime_error("invaild RayDistribution method!");
					break;
				}

				ray r = get_ray(u, v);
				uint8_t max_bounce_depth = 50;
				pixel += shading(max_bounce_depth, false, r);
			}
			pixel /= spp;
			pixel = vec3(sqrt(pixel[0]), sqrt(pixel[1]), sqrt(pixel[2]));
			pixel = color_unit_normalization(pixel, 1);
			frame_buffer.push_back(pixel);
		}
	}
}

vec3 camera::shading(uint16_t depth, bool RussianRoulette, const ray &r)
{
	// // 暂时考虑只返回单一颜色
	// return vec3(1, 10, 10);
	hit_record rec;

	if (world->hit(r, 0.001, 999999, rec)) // FLT_MAX
	{
		ray scattered;
		vec3 attenuation;
		vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
		// 在判断语句中执行并更新散射射线, 并判断是否还有射线生成
		// 同样根据材质给出衰减系数
		if (depth > 0 && rec.mat_ptr->scatter(r, rec, attenuation, scattered))
		{
			return emitted + attenuation * shading(depth - 1, false, scattered);
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

void camera::renderFrame(PresentMethod present, std::string file_path)
{
	uint8_t spp = 1;
	cast_ray(spp, RayDistribution::NAIVE_RANDOM);

	switch (present)
	{
	case PresentMethod::WRITE_FILE:
	{
		std::ofstream OutputImage;
		OutputImage.open(file_path);
		OutputImage << "P3\n"
					<< frame_width << " " << frame_height << "\n255\n";
		for (auto pixelVal : frame_buffer)
		{

			int ir = int(255.99 * pixelVal[0]);
			int ig = int(255.99 * pixelVal[1]);
			int ib = int(255.99 * pixelVal[2]);

			OutputImage << ir << " " << ig << " " << ib << "\n";
		}
	}
	break;
	case PresentMethod::SCREEN_FLOW:
		throw std::runtime_error("not support SCREEN_FLOW presentation");
		break;
	default:
		throw std::runtime_error("invild presentation method");
		break;
	}
}
