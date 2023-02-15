#include "camera/camera.h"
#include "string"

std::string test_file_path = "spark.txt";
std::ofstream spark_ofstream;

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
			shade_point_normal.make_unit_vector();
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

			/**
			 * 	问题发现：2023-02-15晚
			 *
			 * 	对于球状光源的采样我们似乎忽视了一些问题，这导致mental表上的高光似乎并不那么明显：
			 * 这是因为我们在某个球状光源上随机采样时，选择的某个点有可能被自身遮挡从而让我们的着色点错误的
			 * 忽视该光源，这就是为什么一些应该有高光的地方显得比较暗淡。
			 * 	解决这个问题我们需要分开讨论：至少现在看来暂时lambertian表面还算正常，所以我们不对其作太多
			 * 修改。而对于mental表面以及之后的dielectric表面，我们均应允许二次光线打击到光源，从而使得
			 * L_indir中也包含直接光源的影响，尽管这样做在能量守恒上可能是错误的，但至少可以在现在让我们的
			 * 渲染结果变得更加真实。
			 * 	我们来试试看
			 * */

			vec3 light_point_coord = light_point.p;
			vec3 light_point_emit = light_point.mat_ptr->emitted(light_point.u, light_point.v, light_point.p);
			vec3 light_point_normal = light_point.normal;
			light_point_normal.make_unit_vector();

			double light_point_distance = (light_point_coord - shade_point_coord).length();

			/**
			 * 	从这里开始，我们对命名以及物理量的正方向进行规范化定义
			 * 	基本准则就是，以实际物理意义为准：
			 * 	对于正方向：以实际光线传播方向为准进行定义：从光源发出，最终经过光线在场景中的bounce最终到达
			 * 人眼/观察点，以该方向为正方向。
			 * 	对于命名：以计算过程中体现在公式中的命名为准。
			 * */
			vec3 shadePoint_to_viewPoint_wo = -r.direction();
			vec3 directLightSource_to_shadePoint_wi = (shade_point_coord - light_point_coord);
			shadePoint_to_viewPoint_wo.make_unit_vector();
			directLightSource_to_shadePoint_wi.make_unit_vector();

			hit_record first_block_point;
			world.hit(ray(shade_point_coord, -directLightSource_to_shadePoint_wi), 0.001, 999999, first_block_point);

			const float cos_theta_shadePoint = dot(shade_point_normal, -directLightSource_to_shadePoint_wi);
			const float cos_theta_lightPoint = dot(light_point_normal, directLightSource_to_shadePoint_wi);

			// computeBRDF 里面要全部大改
			// 现在先改逻辑
			vec3 BRDF_dir = rec.mat_ptr->computeBRDF(directLightSource_to_shadePoint_wi, shadePoint_to_viewPoint_wo, rec);

			if (BRDF_dir[0] < 0)
			{
				std::cout << "haha" << std::endl;
			}
			vec3 BRDF_indir;
			// std::cout << "BRDF == " << BRDF[0] << ", " << BRDF[1] << ", " << BRDF[2] << " " << std::endl;

			// std::cout << "first_block_point.t = " << first_block_point.t << std::endl;
			// std::cout << "light_point_distance = " << light_point_distance << std::endl;
			// std::cout << "light_point_distance = " << first_block_point.t - light_point_distance << std::endl;
			float parameter = cos_theta_lightPoint * cos_theta_shadePoint / pow(light_point_distance, 2) / light_pdf;

			if (parameter <= 0)
			{
				/* code */
				parameter = -parameter;
			}

			// if (first_block_point.t - light_point_distance > -0.05 || first_block_point.t - light_point_distance < 0.05)
			if (first_block_point.t - light_point_distance > -0.05)
			{

				// if (depth != 50)
				// {
				// 	std::cout << "depth not 50" << std::endl;
				// }

				parameter = parameter < 0 ? -parameter : parameter;
				// std::cout << "parameter = " << parameter << std::endl;
				// L_dir = light_point_emit * BRDF * cos_theta_lightPoint * cos_theta_shadePoint / pow(light_point_distance, 2) / light_pdf;
				L_dir = light_point_emit * BRDF_dir * parameter;

				// 这里改写完了还要进一步检查 2023-02-10
				// std::cout << std::endl;
				// std::cout << "parameter = " << parameter << std::endl;
				// std::cout << "pow = " << pow(light_point_distance, 2) << std::endl;
			}
			// if ((L_dir)[0] >= 1)
			// {
			// 	std::cout << " L_dir[0] = " << L_dir[0] << std::endl;
			// }
			// Then, secondary ray

			// 还是优先检查后面的 L_indir 部分吧 2023-02-10
			vec3 L_indir(0, 0, 0);

			if (this->RussianRoulette < get_random_float())
			{
				return L_dir;
			}
			ray scattered;
			vec3 attenuation;
			rec.mat_ptr->scatter(r, rec, attenuation, scattered);
			vec3 secondaryLightSource_to_shadePoint_wi = -scattered.direction();
			secondaryLightSource_to_shadePoint_wi.make_unit_vector();
			// ray r_deeper(shade_point_coord, secondaryLightSource_to_shadePoint_wi);
			hit_record no_emit_obj;
			bool hitted = world.hit(scattered, 0.0001, 999999, no_emit_obj);
			float cos_para;
			float para_indir;
			// if (no_emit_obj.happened && hitted && !no_emit_obj.mat_ptr->hasEmission())
			if (no_emit_obj.happened && hitted && no_emit_obj.t >= 0.05)
			{

				// 这一大长串应该被优化，写到一个typedef里面进行缩减
				if (rec.mat_ptr->getMaterialType() == material::SelfMaterialType::DIELECTRIC ||
					rec.mat_ptr->getMaterialType() == material::SelfMaterialType::MENTAL ||
					(rec.mat_ptr->getMaterialType() == material::SelfMaterialType::LAMBERTAIN && !no_emit_obj.mat_ptr->hasEmission()))
				{
					// 这里也有问题，这就是为什么你需要特异化你的采样函数
					// 对于lambertian表面以下确实正确，但对于mental这种，以下采样并非半球均匀采样！
					// 同理对于dielectric也不是半球均匀采样，所以你需要进行对提到的这两种表面材质重新
					// 书写pdf采样函数
					const float global_pdf = rec.mat_ptr->pdf(-shadePoint_to_viewPoint_wo, -secondaryLightSource_to_shadePoint_wi, shade_point_normal);

					BRDF_indir = rec.mat_ptr->computeBRDF(secondaryLightSource_to_shadePoint_wi, shadePoint_to_viewPoint_wo, rec);
					cos_para = dot(-secondaryLightSource_to_shadePoint_wi, shade_point_normal);

					// 对于折射光所必要考虑的一步
					if (cos_para <= 0)
					{
						// std::cout << "cou_para = " << cos_para << std::endl;
						cos_para = -cos_para;
					}

					para_indir = cos_para / RussianRoulette / global_pdf;

					L_indir = shading(depth - 1, scattered) * BRDF_indir * para_indir;
				}

				// mental/dielectric的高光部分作为二次光源被lambertian采样到了的情况
				// 我们应该对这个高光项进行限制，暂时还没有想到比较好的对策，我们先不谈这一块

				// if (BRDF_indir[0] < 0)
				// {
				// 	std::cout << "haha" << std::endl;
				// }
			}

			// if ((L_dir + L_indir)[0] <= 0.1 && light_point_distance < 50)
			// {
			// 	std::cout << "haha" << std::endl;
			// }
			// if (rec.mat_ptr->getMaterialType() == material::SelfMaterialType::DIELECTRIC)
			// {
			// 	// std::cout << "material type DIELECTRIC" << std::endl;
			// 	return L_indir;
			// }
			// return L_dir;

			// 这里基本上可以验证确定是因为偶发的极大值无法被平均造成了椒盐噪声！！
			// 但你还无法解释为何大曲率表面不会发生这种现象？！
			// 2023-02-14
			/**
			 * 	猜想以及推导：
			 * 	1.造成这种极大值的项有L_dir和L_indir，后来发现极大值的贡献项主要是L_indir！
			 *  2.然而对于我们现在的测试表面，scatter只可能有两种情况，要么打击不到，要么打击到光源
			 * 而以上两种情况按照逻辑都一定会返回空vec(0,0,0)。
			 *  3.所以可以确定是这一步出问题了，要么是scatter到光源，没有被屏蔽，这是逻辑错误；要么
			 * 就是scatter函数写错，造成了自身scatter到了自身。
			 *  4.我更怀疑是第二种情况的错误：自身scatter到了自身原位置，并且判断通过！先来检查一下。
			 *  5.解决了！！！ 就是二次光线没有判断自相交！！！分析非常合理，应该记录
			 * 	首先想是哪种情况造成了这种极大值？首先是表面对一次光源采样，得到了光源，且正确
			 * 采样，光源的特定采样点与当前着色点之间没有遮挡；其次，
			 * */
			// if (L_dir[0] + L_indir[0] >= 50)
			// {
			// 	vec3 temp = L_indir + L_dir;
			// 	std::cout << "L_dir[0] + L_indir[0] = " << L_dir[0] + L_indir[0] << " ; ";
			// 	std::cout << "L_indir = " << L_indir[0] << std::endl
			// 			  << std::endl;

			// 	spark_ofstream << "[" << temp[0] << " ]"
			// 				   << "\n";
			// }

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
	spark_ofstream.open(test_file_path);
	uint8_t spp = 5;
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
