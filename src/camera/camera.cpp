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

	RussianRoulette = createInfo.RussianRoulette;
}

vec3 random_in_unit_disk()
{
	vec3 p;
	do
	{
		p = 2.0 * vec3(drand48(), drand48(), 0) - vec3(1, 1, 0);
	} while (dot(p, p) >= 1.0);
	// 模拟在方格中撒点，掉入圆圈的点被收录返回
	return p;
}

ray camera::get_ray(float s, float t)
{
	vec3 rd = lens_radius * random_in_unit_disk(); // 得到设定光孔大小内的任意散点（即origin点——viewpoint）
	// （该乘积的后一项是单位光孔）
	vec3 offset = rd.x() * u + rd.y() * v; // origin视点中心偏移（由xoy平面映射到u、v平面）
	// return ray(origin + offset, lower_left_conner + s*horizontal + t*vertical - origin - offset);
	float time = time0 + drand48() * (time1 - time0);
	return ray(origin + offset, upper_left_conner + s * horizontal + t * vertical - origin - offset, time);
}

void camera::sampleLight(hit_record &pos, float &pdf)
{
	float emit_area_sum = 0;
	for (uint32_t k = 0; k < world.list_size; k++)
	{
		if (world.list[k]->hasEmission())
		{
			emit_area_sum += world.list[k]->getArea();
			// std::cout << "get an emission" << std::endl;
		}
		// std::cout << "k = " << k << std::endl;
	}

	// 2023/02/06
	// 现在问题已经很明确了：没有光源或者只有一个光源的情况都不适用！！！你的系统无法兼容这种情况
	if (emit_area_sum == 0)
	{
		throw std::runtime_error("there is no light source in this scene! please check your world construction");
	}

	// std::cout << "total area = " << emit_area_sum << std::endl;

	float p = get_random_float() * emit_area_sum;
	// std::cout << "p = " << p << std::endl;
	emit_area_sum = 0;
	for (uint32_t k = 0; k < world.list_size; k++)
	{
		if (world.list[k]->hasEmission())
		{
			emit_area_sum += world.list[k]->getArea();
			if (p <= emit_area_sum)
			{
				// std::cout << "current kk = " << k << std::endl;
				world.list[k]->Sample(pos, pdf);
				break;
			}
		}
	}
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
				// !!@!!changing depth!!
				uint8_t max_bounce_depth = 50;
				pixel += shading(max_bounce_depth, r);
			}
			pixel /= spp;
			pixel = vec3(sqrt(pixel[0]), sqrt(pixel[1]), sqrt(pixel[2]));
			pixel = color_unit_normalization(pixel, 1);
			frame_buffer.push_back(pixel);
		}
	}
}

vec3 camera::shading(uint16_t depth, const ray &r)
{
	// // 暂时考虑只返回单一颜色
	// return vec3(1, 10, 10);
	hit_record rec;

	// if (world.hit(r, 0.001, 999999, rec)) // FLT_MAX
	// {
	// 	ray scattered;
	// 	vec3 attenuation;
	// 	vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
	// 	// 在判断语句中执行并更新散射射线, 并判断是否还有射线生成
	// 	// 同样根据材质给出衰减系数
	// 	if (depth > 0 && rec.mat_ptr->scatter(r, rec, attenuation, scattered))
	// 	{
	// 		return emitted + attenuation * shading(depth - 1, scattered);
	// 	}
	// 	else
	// 	{
	// 		// if (emitted[0] == 0 && emitted[1] == 0 && emitted[2] == 0)
	// 		// {
	// 		// 	// // std::cout << depth << std::endl;
	// 		// 	// std::cout << "origin point = "
	// 		// 	// 		  << r.origin()[0] << ", "
	// 		// 	// 		  << r.origin()[1] << ", "
	// 		// 	// 		  << r.origin()[2] << std::endl;
	// 		// 	// std::cout << "hit point = "
	// 		// 	// 		  << rec.p[0] << ", "
	// 		// 	// 		  << rec.p[1] << ", "
	// 		// 	// 		  << rec.p[2] << std::endl;
	// 		// 	// // std::cout << "surface normal = "
	// 		// 	// // 		  << rec.normal[0] << ", "
	// 		// 	// // 		  << rec.normal[1] << ", "
	// 		// 	// // 		  << rec.normal[2] << std::endl;
	// 		// 	// // std::cout << "scattered = "
	// 		// 	// // 		  << scattered.direction()[0] << ", "
	// 		// 	// // 		  << scattered.direction()[0] << ", "
	// 		// 	// // 		  << scattered.direction()[0] << std::endl;

	// 		// 	// std::cout << "t = " << rec.t << std::endl;

	// 		// 	// std::cout << std::endl;
	// 		// 	// return emitted;
	// 		// 	// return vec3(0.1, 0.1, 0.8);
	// 		// }
	// 		return emitted;
	// 	}
	// }
	// else
	// {
	// 	// vec3 unit_direction = unit_vector(r.direction());
	// 	// auto t = 0.5 * (unit_direction.y() + 1.0);
	// 	// return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
	// 	return vec3(0, 0, 0);
	// }

	// 2023/01/12
	// 这次我们换一种方式，放弃之前漫无目的的随机scatter，转而对光源进行采样
	// 这个scatter的计算感觉应该被集成在material的scatter函数中
	// 通过不同的表面材质对场景中的光源进行采样

	// 2023/02/05 continue

	// 如果击中了场景中的某个物体
	if (world.hit(r, 0.001, 999999, rec))
	{
		// 如果这个物体是发光体
		if (rec.mat_ptr->hasEmission())
		{
			// 则直接返回其发光光强
			return rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
		}
		// 如果不发光则应该向整个场景的光源进行采样
		else
		{

			vec3 shade_point_coord = rec.p;
			vec3 shade_point_normal = rec.normal;
			double shade_point_distance = rec.t;
			// vec3 I;	 // 光源发光强度 Radiant Intensity
			// vec3 E;	 // 着色点辐照度 Irradiance 从可见光源吸收的光强
			// vec3 Fi; // 反照率：当前着色点向特定方向射出的能量 用 Radiance 表示

			// // 第一步应该是获取所有光源辐射到当前着色点的光强
			// for (int i = 0; i < world.list_size; i++)
			// {
			// 	hitable *current_obj = world.list[i];
			// 	// 检测当前物体是不是光源
			// 	if (current_obj->hasEmission())
			// 	{
			// 		// ray
			// 	}
			// }
			vec3 L_dir(0, 0, 0);
			float light_pdf = 0.0;
			hit_record light_point;
			sampleLight(light_point, light_pdf);

			vec3 light_point_coord = light_point.p;
			vec3 light_point_emit = light_point.mat_ptr->emitted(light_point.u, light_point.v, light_point.p);
			vec3 light_point_normal = light_point.normal;

			double light_point_distance = (light_point_coord - shade_point_coord).length();

			vec3 wo = r.direction();
			vec3 wii = (light_point_coord - shade_point_coord);
			wii.make_unit_vector();

			hit_record first_block_point;
			world.hit(ray(shade_point_coord, wii), 0.001, 999999, first_block_point);

			const float cos_theta_shadePoint = dot(shade_point_normal, wii);
			const float cos_theta_lightPoint = dot(light_point_normal, -wii);

			vec3 BRDF = rec.mat_ptr->computeBRDF(wo, wii, rec);

			if (first_block_point.t - light_point_distance > -0.05)
			{
				float parameter = cos_theta_lightPoint * cos_theta_shadePoint / pow(light_point_distance, 2) / light_pdf;
				parameter = parameter < 0 ? -parameter : parameter;
				// L_dir = light_point_emit * BRDF * cos_theta_lightPoint * cos_theta_shadePoint / pow(light_point_distance, 2) / light_pdf;
				L_dir = light_point_emit * BRDF * parameter;

				// std::cout << std::endl;
				// std::cout << "parameter = " << parameter << std::endl;
				// std::cout << "pow = " << pow(light_point_distance, 2) << std::endl;
			}
			// if (L_dir[0] < 0 && light_point_distance < 50)
			// {
			// 	std::cout << "hahhhh" << std::endl;
			// }

			// Then, secondary ray
			vec3 L_indir(0, 0, 0);

			if (this->RussianRoulette < get_random_float())
			{
				return L_dir;
			}
			ray scattered;
			vec3 attenuation;
			rec.mat_ptr->scatter(r, rec, attenuation, scattered);
			vec3 w0 = scattered.direction();
			w0.make_unit_vector();
			ray r_deeper(shade_point_coord, w0);
			hit_record no_emit_obj;
			bool hitted = world.hit(r_deeper, 0.0001, 999999, no_emit_obj);
			if (no_emit_obj.happened && hitted && !no_emit_obj.mat_ptr->hasEmission())
			{
				const float global_pdf = rec.mat_ptr->pdf(wo, w0, shade_point_normal);
				BRDF = rec.mat_ptr->computeBRDF(wo, w0, rec);
				L_indir = shading(depth - 1, r_deeper) * BRDF * dot(w0, shade_point_normal) / RussianRoulette / global_pdf;

				// if ((L_indir)[0] > 1 && light_point_distance < 50)
				// {
				// 	std::cout << "haha" << std::endl;
				// }
			}

			return L_dir + L_indir;
		}
	}
	else
	{
		return vec3(0, 0, 0);
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

			// OutputImage << ir << " " << ig << " " << ib << "\n";
			// OutputImage << (ir < 0 ? 0 : ir) << " " << (ig < 0 ? 255 : ig) << " " << (ib < 0 ? 0 : ib) << "\n";
			// OutputImage << (ir < 0 ? 255 : ir) << " " << (ig < 0 ? 0 : ig) << " " << (ib < 0 ? 0 : ib) << "\n";
			// OutputImage << (ir < 0 ? 0 : ir) << " " << (ig < 0 ? 0 : ig) << " " << (ib < 0 ? 255 : ib) << "\n";
			OutputImage << (ir < 0 ? 0 : ir) << " " << (ig < 0 ? 0 : ig) << " " << (ib < 0 ? 0 : ib) << "\n";
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
