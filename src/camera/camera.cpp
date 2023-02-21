#include "camera.h"
#include <string>

std::string test_file_path = "spark.txt";
std::ofstream spark_ofstream;

camera::camera(cameraCreateInfo createInfo)
{
	// Vector3f u, v, w;
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

	w = (createInfo.lookat - createInfo.lookfrom).normalized(); // view_ray direction
	u = (w.cross(createInfo.up_dir)).normalized();				// camera plane horizontal direction vec
	v = w.cross(u);												// camera plane vertical direction vec

	// lower_left_conner = origin + focus_dist * w - half_width * focus_dist * u - half_height * focus_dist * v;

	// 我们应该定义的是左上角而不是左下角
	upper_left_conner = origin +
						createInfo.focus_dist * w -
						half_width * createInfo.focus_dist * u -
						half_height * createInfo.focus_dist * v;
	horizontal = 2 * half_width * createInfo.focus_dist * u;
	vertical = 2 * half_height * createInfo.focus_dist * v;

	RussianRoulette = createInfo.RussianRoulette;
	spp = createInfo.spp;
}

Vector3f random_in_unit_disk()
{
	Vector3f p;
	do
	{
		p = 2.0 * Vector3f(drand48(), drand48(), 0) - Vector3f(1, 1, 0);
	} while (p.dot(p) >= 1.0);
	// 模拟在方格中撒点，掉入圆圈的点被收录返回
	return p;
}

ray camera::get_ray(float s, float t)
{
	Vector3f rd = lens_radius * random_in_unit_disk(); // 得到设定光孔大小内的任意散点（即origin点——viewpoint）
	// （该乘积的后一项是单位光孔）
	Vector3f offset = rd.x() * u + rd.y() * v; // origin视点中心偏移（由xoy平面映射到u、v平面）
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

			Vector3f pixel(0, 0, 0);
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
				uint8_t max_bounce_depth = 5;
				pixel += shading(max_bounce_depth, r);
			}
			pixel /= spp;
			pixel = Vector3f(sqrt(pixel[0]), sqrt(pixel[1]), sqrt(pixel[2]));
			pixel = color_unit_normalization(pixel, 1);
			frame_buffer.push_back(pixel);
		}
	}
}

Vector3f camera::shading(uint16_t depth, const ray &r)
{
	hit_record rec;

	// if (world.hit(r, 0.001, 999999, rec)) // FLT_MAX
	// {
	// 	ray scattered;
	// 	Vector3f attenuation;
	// 	Vector3f emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
	// 	// 在判断语句中执行并更新散射射线, 并判断是否还有射线生成
	// 	// 同样根据材质给出衰减系数
	// 	if (depth > 0 && rec.mat_ptr->scatter(r, rec, attenuation, scattered))
	// 	{
	// 		return emitted + attenuation * shading(depth - 1, scattered);
	// 	}
	// 	else
	// 	{
	// 		return emitted;
	// 	}
	// }
	// else
	// {
	// 	// Vector3f unit_direction = unit_vector(r.direction());
	// 	// auto t = 0.5 * (unit_direction.y() + 1.0);
	// 	// return (1.0 - t) * Vector3f(1.0, 1.0, 1.0) + t * Vector3f(0.5, 0.7, 1.0);
	// 	return Vector3f(0, 0, 0);
	// }

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

			Vector3f shade_point_coord = rec.p;
			Vector3f shade_point_normal = rec.normal;
			shade_point_normal.normalize();
			double shade_point_distance = rec.t;

			Vector3f L_dir(0, 0, 0);
			float light_pdf = 0.0;
			hit_record light_point;
			sampleLight(light_point, light_pdf);

			Vector3f light_point_coord = light_point.p;
			Vector3f light_point_emit = light_point.mat_ptr->emitted(light_point.u, light_point.v, light_point.p);
			Vector3f light_point_normal = light_point.normal;
			light_point_normal.normalize();

			double light_point_distance = (light_point_coord - shade_point_coord).norm();

			/**
			 * 	从这里开始，我们对命名以及物理量的正方向进行规范化定义
			 * 	基本准则就是，以实际物理意义为准：
			 * 	对于正方向：以实际光线传播方向为准进行定义：从光源发出，最终经过光线在场景中的bounce最终到达
			 * 人眼/观察点，以该方向为正方向。
			 * 	对于命名：以计算过程中体现在公式中的命名为准。
			 * */
			Vector3f shadePoint_to_viewPoint_wo = -r.direction();
			Vector3f directLightSource_to_shadePoint_wi = (shade_point_coord - light_point_coord);
			shadePoint_to_viewPoint_wo.normalize();
			directLightSource_to_shadePoint_wi.normalize();

			hit_record first_block_point;
			world.hit(ray(shade_point_coord, -directLightSource_to_shadePoint_wi), 0.001, 999999, first_block_point);

			const float cos_theta_shadePoint = shade_point_normal.dot(-directLightSource_to_shadePoint_wi);
			const float cos_theta_lightPoint = light_point_normal.dot(directLightSource_to_shadePoint_wi);

			Vector3f BRDF_dir = rec.mat_ptr->computeBRDF(directLightSource_to_shadePoint_wi, shadePoint_to_viewPoint_wo, rec);

			if (BRDF_dir[0] < 0)
			{
				std::cout << "haha" << std::endl;
			}
			Vector3f BRDF_indir;
			float parameter = cos_theta_lightPoint * cos_theta_shadePoint / pow(light_point_distance, 2) / light_pdf;

			if (parameter <= 0)
			{
				/* code */
				parameter = -parameter;
			}

			if (first_block_point.t - light_point_distance > -0.005)
			{

				parameter = parameter < 0 ? -parameter : parameter;
				L_dir = light_point_emit.array() * BRDF_dir.array() * parameter;
			}

			Vector3f L_indir(0, 0, 0);

			if (this->RussianRoulette < get_random_float())
			{
				return L_dir;
			}
			ray scattered;
			Vector3f attenuation;
			rec.mat_ptr->scatter(r, rec, attenuation, scattered);
			Vector3f secondaryLightSource_to_shadePoint_wi = -scattered.direction();
			secondaryLightSource_to_shadePoint_wi.normalize();
			// ray r_deeper(shade_point_coord, secondaryLightSource_to_shadePoint_wi);
			hit_record no_emit_obj;
			bool hitted = world.hit(scattered, 0.0001, 999999, no_emit_obj);
			float cos_para;
			float para_indir;
			// if (no_emit_obj.happened && hitted && !no_emit_obj.mat_ptr->hasEmission())
			if (no_emit_obj.happened && hitted && no_emit_obj.t >= 0.005)
			{
				if (no_emit_obj.mat_ptr->getMaterialType() == material::SelfMaterialType::LAMBERTAIN && no_emit_obj.mat_ptr->hasEmission())
				{
					return L_dir;
				}
				else
				{

					const float global_pdf = rec.mat_ptr->pdf(-shadePoint_to_viewPoint_wo, -secondaryLightSource_to_shadePoint_wi, shade_point_normal);

					BRDF_indir = rec.mat_ptr->computeBRDF(secondaryLightSource_to_shadePoint_wi, shadePoint_to_viewPoint_wo, rec);
					cos_para = -secondaryLightSource_to_shadePoint_wi.dot(shade_point_normal);

					// 对于折射光所必要考虑的一步
					if (cos_para <= 0)
					{
						cos_para = -cos_para;
					}

					para_indir = cos_para / RussianRoulette / global_pdf;

					L_indir = shading(depth - 1, scattered).array() * BRDF_indir .array()* para_indir;
				}
			}

			return L_dir + L_indir;
		}
	}
	else
	{
		return Vector3f(0, 0, 0);
	}
}

void camera::renderFrame(PresentMethod present, std::string file_path)
{
	spark_ofstream.open(test_file_path);
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
			// OutputImage << (ir < 0 ? 255 : ir) << " " << (ig < 0 ? 0 : ig) << " " << (ib < 0 ? 0 : ib) << "\n";
			OutputImage << (ir < 0 ? 0 : ir) << " " << (ig < 0 ? 255 : ig) << " " << (ib < 0 ? 0 : ib) << "\n";
			// OutputImage << (ir < 0 ? 0 : ir) << " " << (ig < 0 ? 0 : ig) << " " << (ib < 0 ? 255 : ib) << "\n";
			// OutputImage << (ir < 0 ? 0 : ir) << " " << (ig < 0 ? 0 : ig) << " " << (ib < 0 ? 0 : ib) << "\n";
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
