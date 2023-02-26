#include "camera.cuh"

/**
 * 	相机构造函数
 * */
__device__ __host__ camera::camera(cameraCreateInfo createInfo)
{
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

	spp = createInfo.spp;
}

/**
 * 	相机渲染函数（host端调用，渲染入口）
 * */
__host__ void camera::renderFrame(PresentMethod present, std::string file_path)
{
	// cast_ray(spp, RayDistribution::NAIVE_RANDOM);

	switch (present)
	{
	case PresentMethod::WRITE_FILE:
	{
		// 执行并行射线投射，获得一帧图像
		vec3 *frame_buffer_host = cast_ray_device(frame_width, frame_height, 1);
		// 返回host端画图
		std::ofstream OutputImage;
		OutputImage.open(file_path);
		OutputImage << "P3\n"
					<< frame_width << " " << frame_height << "\n255\n";

		for (int row = 0; row < frame_height; row++)
		{
			for (int col = 0; col < frame_width; col++)
			{
				const int global_index = row * frame_width + col;
				vec3 pixelVal = frame_buffer_host[global_index];
				int ir = int(255.99 * pixelVal[0]);
				int ig = int(255.99 * pixelVal[1]);
				int ib = int(255.99 * pixelVal[2]);
				OutputImage << ir << " " << ig << " " << ib << "\n";
			}
		}
	}
	break;
	case PresentMethod::SCREEN_FLOW:
		// throw std::runtime_error("not support SCREEN_FLOW presentation");
		{
			cv::namedWindow("Image Flow");
			int counter = 0;
			// 一直执行这个循环，并将图像给到OpenCV创建的 window，直到按下 Esc 键推出
			while (true)
			{
				/* code */
				vec3 *frame_buffer_host = cast_ray_device(frame_width, frame_height, counter);
				counter += 1;
				std::cout << counter << std::endl;
				showFrameFlow(frame_width, frame_height, frame_buffer_host);

				if (cv::waitKey(1) == 27)
				{
					break;
				}
			}
		}
		break;
	default:
		throw std::runtime_error("invild presentation method");
		break;
	}
}

/**
 * 	将并行渲染结果使用opencv创建窗口进行展示（图片流形式）
 * */
__host__ void camera::showFrameFlow(int width, int height, vec3 *frame_buffer_host)
{

	cv::Mat img = cv::Mat(cv::Size(width, height), CV_8UC3);

	for (int col = 0; col < height; col++)
	{
		for (int row = 0; row < width; row++)
		{
			const int global_index = col * height + row;
			int color_R = frame_buffer_host[global_index][0] * 255.99;
			int color_G = frame_buffer_host[global_index][1] * 255.99;
			int color_B = frame_buffer_host[global_index][2] * 255.99;
			// std::cout << global_index << "  gray_color = " << gray_color << std::endl;
			img.at<unsigned char>(col, row * 3 + 0) = color_B;
			img.at<unsigned char>(col, row * 3 + 1) = color_G;
			img.at<unsigned char>(col, row * 3 + 2) = color_R;
		}
	}
	// std::cout << "out" << std::endl;
	cv::imshow("Image Flow", img);
	// cv::waitKey(30);
	// while(1){}
}

/**
 * 	渲染函数入口（内部调用 初始化随机数kernel 并行渲染kernel）
 * */
__host__ vec3 *camera::cast_ray_device(float frame_width, float frame_height, int spp)
{
	int device = 0;		   // 设置使用第0块GPU进行运算
	cudaSetDevice(device); // 设置运算显卡
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, device); // 获取对应设备属性

	// 整个frame的大小
	// int size = frame_width * frame_height * sizeof(float);
	int size = frame_width * frame_height * sizeof(vec3);
	// 开辟将要接收计算回传数据的内存空间
	vec3 *frame_buffer_host = (vec3 *)malloc(size);

	unsigned int block_size_width = 32;
	unsigned int block_size_height = 32;
	unsigned int grid_size_width = frame_width / block_size_width;
	unsigned int grid_size_height = frame_height / block_size_height;

	dim3 dimBlock(block_size_height, block_size_width, 1);
	dim3 dimGrid(grid_size_height, grid_size_width, 1);

	/* ############################### 初始化随机数 ############################### */
	curandStateXORWOW_t *states;
	cudaMalloc(&states, sizeof(curandStateXORWOW_t) * FRAME_WIDTH * FRAME_HEIGHT);

	initialize_device_random<<<dimGrid, dimBlock>>>(states, time(nullptr), frame_width * frame_height);

	cudaDeviceSynchronize();

	/* ############################### 初始化摄像机 ############################### */
	int camera_size = sizeof(camera);

	// std::cout << "camera size = " << camera_size << std::endl;
	camera *cpu_camera = createCamera();

	// 将host本地创建初始化好的摄像机，连带参数一同拷贝到device设备端
	cudaMemcpyToSymbol(PRIMARY_CAMERA, cpu_camera, camera_size);

	cudaDeviceSynchronize();

	/* ################################# 场景生成 ################################# */
	// 	其实这里是分配到了显存上的全局内存上，按道理来讲这种经常被访问到的全局场景应该被放在带有
	// 缓存的__constant__内存上，这是之后的修改升级空间（不过现在也是全局可见，不会出错）
	hitable_list **world_device = NULL;
	cudaMalloc((void **)&world_device, sizeof(hitable_list **));
	gen_world<<<1, 1>>>(states, world_device);
	// gen_world();

	/* ############################### Real Render ############################### */

	// 开辟显存空间
	vec3 *frame_buffer_device;
	cudaMalloc((void **)&frame_buffer_device, size);

	// ##################### 这里看一下并行用时 #####################

	cudaEvent_t start, stop;
	float runTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// 不要忘了给模板函数添加模板参数
	// 所有的并行计算应该都在这一个函数中完成，这个函数要调用其他.cu文件中的函数，并且也要在device上执行
	// 关键问题是那些预定义的类怎么办？CUDA中无法直接使用这些类
	cuda_shading_unit<<<dimGrid, dimBlock>>>(frame_buffer_device, world_device, states);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&runTime, start, stop);

	std::cout << ": para time cost: " << runTime << "ms" << std::endl;

	// ##################### End #####################

	// 从显存向内存拷贝（第一个参数是dst，第二个参数是src）
	cudaMemcpy(frame_buffer_host, frame_buffer_device, size, cudaMemcpyDeviceToHost);

	cudaFree(frame_buffer_device);

	return frame_buffer_host;
}

/**
 * 	初始化device端随机数生成kernel
 * */
__global__ void initialize_device_random(curandStateXORWOW_t *states, unsigned long long seed, size_t size)
{
	/*################################ 全局索引 ################################*/
	int row_index = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程所在行索引
	int col_index = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程所在列索引

	int row_len = gridDim.x * blockDim.x; // 行宽（列数）
	// int col_len = gridDim.y * blockDim.y;                 // 列高（行数）
	int global_index = (row_len * row_index + col_index); // 全局索引

	curand_init(seed, global_index, 0, &states[global_index]);
}
/**
 * 	单一像素并行渲染kernel
 * */
__global__ void cuda_shading_unit(vec3 *frame_buffer, hitable_list **world, curandStateXORWOW_t *rand_state)
{

	/*################################ 全局索引 ################################*/
	int row_index = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程所在行索引
	int col_index = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程所在列索引

	int row_len = gridDim.x * blockDim.x; // 行宽（列数）
	// int col_len = gridDim.y * blockDim.y;                 // 列高（行数）
	int global_index = (row_len * row_index + col_index); // 全局索引

	// int global_size = row_len * col_len;

	// /*############################## 获取当前光线 ##############################*/
	float u = float(col_index + random_double_device(&rand_state[global_index])) / float(PRIMARY_CAMERA.frame_width);
	float v = float(row_index + random_double_device(&rand_state[global_index])) / float(PRIMARY_CAMERA.frame_height);
	ray kernal_ray = PRIMARY_CAMERA.get_ray_device(u, v, &rand_state[global_index]);
	vec3 color = PRIMARY_CAMERA.shading_device(3, kernal_ray, *world, rand_state);

	frame_buffer[global_index] = color;
}

/**
 * 	获取当前kernel对应射线
 * */
__device__ ray camera::get_ray_device(float s, float t, curandStateXORWOW *rand_state)
{
	// 全部相机参数
	vec3 u = PRIMARY_CAMERA.u;
	vec3 v = PRIMARY_CAMERA.v;
	float lens_radius = PRIMARY_CAMERA.lens_radius;
	float time0 = PRIMARY_CAMERA.time0, time1 = PRIMARY_CAMERA.time1;
	vec3 origin = PRIMARY_CAMERA.origin;
	vec3 upper_left_conner = PRIMARY_CAMERA.upper_left_conner;
	vec3 horizontal = PRIMARY_CAMERA.horizontal;
	vec3 vertical = PRIMARY_CAMERA.vertical;

	vec3 rd = lens_radius * random_in_unit_disk_device(rand_state); // 得到设定光孔大小内的任意散点（即origin点——viewpoint）
	vec3 offset = rd.x() * u + rd.y() * v;							// origin视点中心偏移（由xoy平面映射到u、v平面）
	float time = time0 + random_double_device(rand_state) * (time1 - time0);
	return ray(origin + offset, upper_left_conner + s * horizontal + t * vertical - origin - offset, time);
}

/**
 * 	着色函数（之后的打击函数/射线相交测试都要写在这里面）
 * */

__device__ vec3 camera::shading_device(int depth, ray &r, hitable_list *world, curandStateXORWOW_t *rand_state)
{

	hit_record rec;
	// printf("depth = %d\n", depth);
	// 现在可以确定的是，执行到这一步的时候出错了，hit函数并没有能良好执行
	// 但是程序运行时也没有报错，说明可能是内部发生了指针错乱
	if (world->hit(r, 0.001, 999999, rec)) // FLT_MAX
	{
		ray scattered;
		vec3 attenuation;
		// vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
		// 在判断语句中执行并更新散射射线, 并判断是否还有射线生成
		// 同样根据材质给出衰减系数
		// if (depth > 0 && rec.mat_ptr->scatter(r, rec, attenuation, scattered, rand_state))
		// {
		// 	// printf("ret depth");
		// 	return emitted + attenuation * shading_device(depth - 1, scattered, world, rand_state);
		// }
		// else
		// {
		// 	// printf("emit return\n");
		// 	return emitted;
		// }
		return vec3(0.1, 0.9, 0.8);
	}
	else
	{
		// printf("not hit return");
		vec3 unit_direction = unit_vector(r.direction());
		auto t = 0.5 * (unit_direction.y() + 1.0);
		return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
		// return vec3(0, 0, 0);
	}
}
// __device__ vec3 camera::shading_device(ray &r, hitable_list *world, curandStateXORWOW_t *rand_state)
// {

// 	hit_record rec;

// 	vec3 mul(0, 0, 0);
// 	int depth = 2;

// 	while ((depth > 0) && world->hit(r, 0, 999999, rec))
// 	{
// 		depth--;
// 		ray scattered;
// 		vec3 attenuation;
// 		if (rec.mat_ptr->scatter(r, rec, attenuation, scattered, rand_state))
// 		{
// 			r = scattered;
// 			mul = cross(mul, attenuation);
// 		}
// 		else
// 		{
// 			return vec3(0, 0, 0);
// 		}
// 	}
// 	if(depth==0)
// 	{
// 		return vec3(0,0,0);
// 	}

// 	vec3 unit_direction = unit_vector(r.direction());
// 	auto t = 0.5 * (unit_direction.y() + 1.0);
// 	// return vec3(0.5, 0, 0);
// 	// return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
// 	return cross(mul, (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0));
// }

/**
 * 	摄像机创建，多用于host端的相机创建，device端的相机实例应从主机创建好后初始化设备上的constant内存
 * 因为相机各项参数是经常被访问到的，且必须对所有的thread全局可见，所以应该存在带有cache的constant内存
 * */
__host__ __device__ camera *createCamera(void)
{
	cameraCreateInfo createCamera{};

	createCamera.lookfrom = vec3(20, 15, 20);
	createCamera.lookat = vec3(0, 0, 0);

	createCamera.up_dir = vec3(0, 1, 0);
	createCamera.fov = 40;
	createCamera.aspect = float(FRAME_WIDTH) / float(FRAME_HEIGHT);
	createCamera.focus_dist = 10.0;
	createCamera.aperture = 1;
	createCamera.t0 = 0.0;
	createCamera.t1 = 1.0;
	createCamera.frame_width = FRAME_WIDTH;
	createCamera.frame_height = FRAME_HEIGHT;

	createCamera.spp = 1;

	// 学会像vulkan那样构建
	return new camera(createCamera);
}
