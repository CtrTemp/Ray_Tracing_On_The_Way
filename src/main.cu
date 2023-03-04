#include <stdio.h>
#include <iostream>
#include <sys/time.h>

#include <string>

#include "camera/camera.cuh"

// using namespace cv;

unsigned int frame_width = 512;
unsigned int frame_height = 512;

__global__ void initialize_device_random(curandStateXORWOW_t *states, unsigned long long seed, size_t size)
{
	int row_index = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程所在行索引
	int col_index = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程所在列索引

	int row_len = gridDim.x * blockDim.x; // 行宽（列数）
	// int col_len = gridDim.y * blockDim.y;                 // 列高（行数）
	int global_index = (row_len * row_index + col_index); // 全局索引

	curand_init(seed, global_index, 0, &states[global_index]);
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

__global__ void gen_world(curandStateXORWOW_t *rand_state, hitable **world, hitable **list)
{
	// 在设备端创建
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		list[0] = new sphere(vec3(0, -100, 0), 100, new lambertian(vec3(0.5, 0.1, 0.1)));
		list[1] = new sphere(vec3(0, 1, 0), 1, new lambertian(vec3(0.1, 0.1, 0.9)));
		*world = new hitable_list(list, 2);
	}
}

__device__ ray get_ray_device(float s, float t, curandStateXORWOW *rand_state)
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
	return ray(origin + offset, upper_left_conner + s * horizontal + t * vertical - origin - offset);
}

__device__ vec3 shading_pixel(int depth, ray &r, hitable **world, curandStateXORWOW_t *rand_state)
{
	hit_record rec;
	ray cur_ray(vec3(0, 0, 0), vec3(0, 0, 0));
	if ((*world)->hit(r, 0.001, 999999, rec)) // FLT_MAX
	{
		// ray scattered;
		// vec3 attenuation;
		// rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, rand_state);
		// return rec.mat_ptr.;
		return vec3(0.9, 0.1, 0.1);
	}
	else
	{
		// printf("not hit return");
		vec3 unit_direction = unit_vector(cur_ray.direction());
		auto t = 0.5 * (unit_direction.y() + 1.0);
		return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
		// return vec3(0, 0, 0);
	}

	// return  cur_ray;

	// ray cur_ray = r;
	// vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	// for (int i = 0; i < 50; i++)
	// {
	// 	hit_record rec;
	// 	if ((*world)->hit(cur_ray, 0.001f, 999999, rec))
	// 	{
	// 		ray scattered;
	// 		vec3 attenuation;
	// 		if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, rand_state))
	// 		{
	// 			cur_attenuation *= attenuation;
	// 			cur_ray = scattered;
	// 		}
	// 		else
	// 		{
	// 			return vec3(0.0, 0.0, 0.0);
	// 		}
	// 	}
	// 	else
	// 	{
	// 		vec3 unit_direction = unit_vector(cur_ray.direction());
	// 		float t = 0.5f * (unit_direction.y() + 1.0f);
	// 		vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
	// 		return cur_attenuation * c;
	// 	}
	// }
	// return vec3(0.0, 0.0, 0.0);
}
__global__ void cuda_shading_unit(vec3 *frame_buffer, hitable **world, curandStateXORWOW_t *rand_state)
{
	// printf("prim camera width = %d", PRIMARY_CAMERA.frame_width);
	int row_index = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程所在行索引
	int col_index = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程所在列索引

	int row_len = gridDim.x * blockDim.x; // 行宽（列数）
	// int col_len = gridDim.y * blockDim.y;                 // 列高（行数）
	int global_index = (row_len * row_index + col_index); // 全局索引

	float u = float(col_index + random_double_device(&rand_state[global_index])) / float(PRIMARY_CAMERA.frame_width);
	float v = float(row_index + random_double_device(&rand_state[global_index])) / float(PRIMARY_CAMERA.frame_height);
	ray kernal_ray = get_ray_device(u, v, &rand_state[global_index]);
	vec3 color = shading_pixel(3, kernal_ray, world, rand_state);

	frame_buffer[global_index] = color;
}

int main(void)
{
	int device = 0;		   // 设置使用第0块GPU进行运算
	cudaSetDevice(device); // 设置运算显卡
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, device); // 获取对应设备属性

	unsigned int block_size_width = 32;
	unsigned int block_size_height = 32;
	unsigned int grid_size_width = FRAME_WIDTH / block_size_width;
	unsigned int grid_size_height = FRAME_HEIGHT / block_size_height;

	dim3 dimBlock(block_size_height, block_size_width, 1);
	dim3 dimGrid(grid_size_height, grid_size_width, 1);
	// curandStateXORWOW_t *states = init_rand(block_size_width, block_size_height);

	/* ##################################### 随机数初始化 ##################################### */
	curandStateXORWOW_t *states;
	cudaMalloc(&states, sizeof(curandStateXORWOW_t) * FRAME_WIDTH * FRAME_HEIGHT);
	initialize_device_random<<<dimGrid, dimBlock>>>(states, time(nullptr), FRAME_WIDTH * FRAME_HEIGHT);
	cudaDeviceSynchronize();

	/* ##################################### 摄像机初始化 ##################################### */
	int camera_size = sizeof(camera);
	camera *cpu_camera = createCamera();
	cudaMemcpyToSymbol(PRIMARY_CAMERA, cpu_camera, camera_size);
	cudaDeviceSynchronize();

	/* ##################################### 场景初始化 ##################################### */
	hitable **world_device;
	hitable **list_device;
	cudaMalloc((void **)&world_device, 2 * sizeof(hitable *));
	cudaMalloc((void **)&list_device, sizeof(hitable *));
	gen_world<<<1, 1>>>(states, world_device, list_device);

	/* ##################################### 全局渲染入口 ##################################### */
	// 初始化帧缓存
	vec3 *frame_buffer_device;
	int size = FRAME_WIDTH * FRAME_HEIGHT * sizeof(vec3);
	cudaMalloc((void **)&frame_buffer_device, size);

	cuda_shading_unit<<<dimGrid, dimBlock>>>(frame_buffer_device, world_device, states);
	// 一个很重要的问题，如果不加以下的阻塞，等待所有进程执行完毕，则cpu不会等待gpu执行完成就会return
	// 因为以上的调用是cpu向gpu提交任务，但不会等GPU执行完毕，直接返回，所以你根本看不到gpu的回传打印数据～
	cudaDeviceSynchronize();
	// 在主机开辟 framebuffer 空间
	vec3 *frame_buffer_host = new vec3[FRAME_WIDTH * FRAME_HEIGHT];

	cudaMemcpy(frame_buffer_host, frame_buffer_device, size, cudaMemcpyDeviceToHost);

	// std::cout << frame_buffer_host[512 * 512 - 1] << std::endl;
	std::string file_path = "./any.ppm";
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

	// init_camera();

	// hitable **world = init_world(states);

	// render(block_size_height, block_size_height, states, world);
	cudaFree(world_device);
	cudaFree(list_device);
	cudaDeviceReset();

	return 0;
}
