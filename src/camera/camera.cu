#include "camera.cuh"


__global__ void initialize_device_random(curandStateXORWOW_t *states, unsigned long long seed, size_t size)
{
    /*################################ 全局索引 ################################*/
    int row_index = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程所在行索引
    int col_index = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程所在列索引

    int row_len = gridDim.x * blockDim.x;                 // 行宽（列数）
    int col_len = gridDim.y * blockDim.y;                 // 列高（行数）
    int global_index = (row_len * row_index + col_index); // 全局索引

    curand_init(seed, global_index, 0, &states[global_index]);
}

__global__ void cuda_shading_unit(vec3 *frame_buffer, curandStateXORWOW_t *rand_state)
{

    /*################################ 全局索引 ################################*/
    int row_index = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程所在行索引
    int col_index = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程所在列索引

    int row_len = gridDim.x * blockDim.x;                 // 行宽（列数）
    // int col_len = gridDim.y * blockDim.y;                 // 列高（行数）
    int global_index = (row_len * row_index + col_index); // 全局索引

    // int global_size = row_len * col_len;

    // /*############################## 随机数初始化 ##############################*/
    // curandStateXORWOW_t *rand_state;
    // curand_init(global_index, 0, 0, rand_state);

    // /*############################## 获取当前光线 ##############################*/
    // 原来是这里出了大问题！！最后一项访问不到
    float u = float(col_index + random_double_device(0, 1.0, &rand_state[global_index])) / float(512);
    float v = float(row_index + random_double_device(0, 1.0, &rand_state[global_index])) / float(512);
    ray kernal_ray = PRIMARY_CAMERA.get_ray_device(u, v, &rand_state[global_index]);
    vec3 color = PRIMARY_CAMERA.shading_device(kernal_ray);

    // float single_color = (float)(global_index) / global_size;
    // vec3 color(random_double_device(&rand_state[global_index]), random_double_device(&rand_state[global_index]), random_double_device(&rand_state[global_index]));
    // color[0] = single_color;
    // color[1] = single_color;
    // color[2] = single_color;

    frame_buffer[global_index] = color;
}

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

	// std::cout << "grid size = [" << grid_size_height << ", " << grid_size_width << "]" << std::endl;
	// std::cout << "global size = " << size << std::endl;

	dim3 dimBlock(block_size_height, block_size_width, 1);
	dim3 dimGrid(grid_size_height, grid_size_width, 1);

	/* ############################### 初始化随机数 ############################### */
	curandStateXORWOW_t *states;
	cudaMalloc(&states, sizeof(curandStateXORWOW_t) * FRAME_WIDTH * FRAME_HEIGHT);

	initialize_device_random<<<dimGrid, dimBlock>>>(states, time(nullptr), frame_width * frame_height);

	cudaDeviceSynchronize();

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
	cuda_shading_unit<<<dimGrid, dimBlock>>>(frame_buffer_device, states);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&runTime, start, stop);

	std::cout << ": para time cost: " << runTime << "ms" << std::endl;

	// ##################### End #####################

	// 从显存向内存拷贝（第一个参数是dst，第二个参数是src）
	cudaMemcpy(frame_buffer_host, frame_buffer_device, size, cudaMemcpyDeviceToHost);

	// std::cout << "host[1000] = " << frame_buffer_host[1000] << std::endl;

	// 你不能这样直接访问device的地址！！！
	// std::cout << "device = " << frame_buffer_device[0] << std::endl;

	cudaFree(frame_buffer_device);

	return frame_buffer_host;
}

__device__ ray camera::get_ray_device(float s, float t, curandStateXORWOW *rand_state)
{
	// 全部相机参数
	vec3 u(0.707, 0, 0.707);
	vec3 v(0.3313, -0.8835, 0.3313);
	float lens_radius = 0.5;
	float time0 = 0, time1 = 1.0;
	vec3 origin(20, 15, 20);
	vec3 upper_left_conner(9.97, 13.53, 15.12);
	vec3 horizontal(5.15, 0, 5.15);
	vec3 vertical(2.41, -6.43, 2.41);

	// return ray();

	vec3 rd = lens_radius * random_in_unit_disk_device(rand_state); // 得到设定光孔大小内的任意散点（即origin点——viewpoint）
	// （该乘积的后一项是单位光孔）
	vec3 offset = rd.x() * u + rd.y() * v; // origin视点中心偏移（由xoy平面映射到u、v平面）
	// return ray(origin + offset, lower_left_conner + s*horizontal + t*vertical - origin - offset);
	float time = time0 + random_double_device(rand_state) * (time1 - time0);
	return ray(origin + offset, upper_left_conner + s * horizontal + t * vertical - origin - offset, time);
}

__device__ vec3 camera::shading_device(const ray &r)
{
	vec3 unit_direction = unit_vector(r.direction());
	auto t = 0.5 * (unit_direction.y() + 1.0);
	// return vec3(0.5, 0, 0);
	return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}


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


// camera *get_camera_info(void)
// {

//     int device = 0;        // 设置使用第0块GPU进行运算
//     cudaSetDevice(device); // 设置运算显卡
//     cudaDeviceProp devProp;
//     cudaGetDeviceProperties(&devProp, device); // 获取对应设备属性

//     /* ############################### 初始化摄像机 ############################### */
//     int camera_size = sizeof(camera);

//     camera *cpu_camera = new camera();

//     // 将host本地创建初始化好的摄像机，连带参数一同拷贝到device设备端
//     cudaMemcpyFromSymbol(cpu_camera, PRIMARY_CAMERA, camera_size);

//     // std::cout << "camera height fetch back = " << cpu_camera->frame_height << std::endl;
//     return cpu_camera;
// }


// __host__ ray camera::get_ray(float s, float t)
// {
// 	vec3 rd = lens_radius * random_in_unit_disk(); // 得到设定光孔大小内的任意散点（即origin点——viewpoint）
// 	// （该乘积的后一项是单位光孔）
// 	vec3 offset = rd.x() * u + rd.y() * v; // origin视点中心偏移（由xoy平面映射到u、v平面）
// 	// return ray(origin + offset, lower_left_conner + s*horizontal + t*vertical - origin - offset);
// 	float time = time0 + drand48() * (time1 - time0);
// 	return ray(origin + offset, upper_left_conner + s * horizontal + t * vertical - origin - offset, time);
// }

// 规定从左上角遍历到右下角，行优先遍历
// __device__ __host__ void camera::cast_ray(uint16_t spp, RayDistribution distribute)
// {

// 	for (int row = 0; row < frame_height; row++)
// 	{
// 		for (int col = 0; col < frame_width; col++)
// 		{

// 			vec3 pixel(0, 0, 0);
// 			for (int s = 0; s < spp; s++)
// 			{
// 				float u, v;

// 				u = float(col + rand() % 101 / float(101)) / float(this->frame_width);
// 				v = float(row + rand() % 101 / float(101)) / float(this->frame_height);

// 				ray r = get_ray(u, v);
// 				// !!@!!changing depth!!
// 				uint8_t max_bounce_depth = 50;
// 				pixel += shading(max_bounce_depth, r);
// 			}
// 			pixel /= spp;
// 			pixel = vec3(sqrt(pixel[0]), sqrt(pixel[1]), sqrt(pixel[2]));
// 			pixel = color_unit_normalization(pixel, 1);
// 			// frame_buffer.push_back(pixel);
// 		}
// 	}
// }

// __device__ __host__ vec3 camera::shading(uint16_t depth, const ray &r)
// {
// 	vec3 unit_direction = unit_vector(r.direction());
// 	auto t = 0.5 * (unit_direction.y() + 1.0);
// 	return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
// }
