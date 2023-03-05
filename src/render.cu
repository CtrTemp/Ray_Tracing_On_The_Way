#include "render.h"

/* ##################################### 随机数初始化 ##################################### */

// __host__ curandStateXORWOW *init_rand(int block_size_width, int block_size_height)
// {
//     curandStateXORWOW *states;
//     cudaMalloc(&states, sizeof(curandStateXORWOW) * FRAME_WIDTH * FRAME_HEIGHT);

//     unsigned int grid_size_width = FRAME_WIDTH / block_size_width;
//     unsigned int grid_size_height = FRAME_HEIGHT / block_size_height;

//     dim3 dimBlock(block_size_height, block_size_width, 1);
//     dim3 dimGrid(grid_size_height, grid_size_width, 1);
//     initialize_device_random<<<dimGrid, dimBlock>>>(states, time(nullptr), FRAME_WIDTH * FRAME_HEIGHT);
//     cudaDeviceSynchronize();

//     return states;
// }

__global__ void initialize_device_random(curandStateXORWOW *states, unsigned long long seed, size_t size)
{
    int row_index = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程所在行索引
    int col_index = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程所在列索引

    int row_len = gridDim.x * blockDim.x; // 行宽（列数）
    // int col_len = gridDim.y * blockDim.y;                 // 列高（行数）
    int global_index = (row_len * row_index + col_index); // 全局索引

    curand_init(seed, global_index, 0, &states[global_index]);
}

/* ##################################### 摄像机初始化 ##################################### */

// __host__ void init_camera(void)
// {
//     int camera_size = sizeof(camera);
//     camera *cpu_camera = createCamera();
//     cudaMemcpyToSymbol(PRIMARY_CAMERA, cpu_camera, camera_size);
//     cudaDeviceSynchronize();
// }

__host__ camera *createCamera(void)
{
    cameraCreateInfo createCamera{};

    // *d_camera = new camera(vec3(-2, 2, 1),
    //                        vec3(0, 0, -1),
    //                        vec3(0, 1, 0),
    createCamera.lookfrom = vec3(-2, 2, 1);
    createCamera.lookat = vec3(0, 0, -1);

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

/* ##################################### 场景初始化 ##################################### */

// __host__ hitable **init_world(curandStateXORWOW *rand_states)
// {
//     hitable **world_device;
//     hitable **list_device;
//     cudaMalloc((void **)&world_device, 2 * sizeof(hitable *));
//     cudaMalloc((void **)&list_device, sizeof(hitable *));
//     gen_world<<<1, 1>>>(rand_states, world_device, list_device);

//     return world_device;
// }

__global__ void gen_world(curandStateXORWOW *rand_state, hitable **world, hitable **list)
{
    // 在设备端创建
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        list[0] = new sphere(vec3(0, 0, -1), 0.5,
                             new lambertian(vec3(0.1, 0.2, 0.5)));
        list[1] = new sphere(vec3(0, -100.5, -1), 100,
                             new lambertian(vec3(0.8, 0.8, 0.0)));
        list[2] = new sphere(vec3(1, 0, -1), 0.5,
                             new mental(vec3(0.8, 0.6, 0.2), 0.0));
        list[3] = new sphere(vec3(-1, 0, -1), 0.5,
                             new dielectric(1.5));
        list[4] = new sphere(vec3(-1, 0, -1), -0.45,
                             new dielectric(1.5));
        *world = new hitable_list(list, 5);

        // list[0] = new sphere(vec3(0, -100, 0), 100, new lambertian(vec3(0.5, 0.1, 0.1)));
        // list[1] = new sphere(vec3(0, 1, 0), 1, new lambertian(vec3(0.1, 0.1, 0.9)));
        // *world = new hitable_list(list, 2);
    }
}
/* ##################################### 光线投射全局渲染 ##################################### */

// __host__ void render(int block_size_width, int block_size_height, curandStateXORWOW *rand_states, hitable **world_device, vec3 *frame_buffer_host)
// {
//     vec3 *frame_buffer_device;
//     int size = FRAME_WIDTH * FRAME_HEIGHT * sizeof(vec3);
//     cudaMalloc((void **)&frame_buffer_device, size);
//     unsigned int grid_size_width = FRAME_WIDTH / block_size_width;
//     unsigned int grid_size_height = FRAME_HEIGHT / block_size_height;

//     dim3 dimBlock(block_size_height, block_size_width, 1);
//     dim3 dimGrid(grid_size_height, grid_size_width, 1);
//     cuda_shading_unit<<<dimGrid, dimBlock>>>(frame_buffer_device, world_device, rand_states);

//     cudaDeviceSynchronize();

//     cudaMemcpy(frame_buffer_host, frame_buffer_device, size, cudaMemcpyDeviceToHost);
// }

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
    vec3 offset = rd.x() * u + rd.y() * v;                          // origin视点中心偏移（由xoy平面映射到u、v平面）
    offset = vec3(0, 0, 0);                                         // 这里目前有bug，先置为0
    float time = time0 + random_double_device(rand_state) * (time1 - time0);
    return ray(origin + offset, upper_left_conner + s * horizontal + t * vertical - origin - offset);
    // return ray(origin, upper_left_conner + u * horizontal + v * vertical - origin);
}

__device__ vec3 shading_pixel(int depth, const ray &r, hitable **world, curandStateXORWOW *rand_state)
{

    hit_record rec;
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++)
    {
        if ((*world)->hit(cur_ray, 0.001f, 999999, rec))
        {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, rand_state))
            {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else
            {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else
        {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.90, 0.0, 0.0);
}
__global__ void cuda_shading_unit(vec3 *frame_buffer, hitable **world, curandStateXORWOW *rand_state, float *debug_buffer)
{
    int row_index = threadIdx.y + blockIdx.y * blockDim.y; // 当前线程所在行索引
    int col_index = threadIdx.x + blockIdx.x * blockDim.x; // 当前线程所在列索引

    // if ((row_index >= FRAME_HEIGHT) || (col_index >= FRAME_WIDTH))
    // {
    //     return;
    // }
    // if (row_index % 16 != 0)
    //     printf("row");
    // if (col_index % 16 != 0)
    //     printf("col");
    int row_len = gridDim.x * blockDim.x; // 行宽（列数）
    // int col_len = gridDim.y * blockDim.y;                 // 列高（行数）
    int global_index = (row_len * row_index + col_index); // 全局索引
    curandStateXORWOW local_rand_state = rand_state[global_index];

    vec3 col(0, 0, 0);
    for (int s = 0; s < PRIMARY_CAMERA.spp; s++)
    {
        float u = float(col_index + 0) / float(PRIMARY_CAMERA.frame_width);
        float v = float(row_index + 0) / float(PRIMARY_CAMERA.frame_height);

        ray kernal_ray = get_ray_device(u, v, &rand_state[global_index]);
        col += shading_pixel(3, kernal_ray, world, &local_rand_state);
    }
    rand_state[global_index] = local_rand_state;
    col /= float(PRIMARY_CAMERA.spp);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    // col = random_vec3_device(&local_rand_state);

    debug_buffer[global_index] = col[0];
    frame_buffer[global_index] = col;
}

/* ##################################### main 函数入口 ##################################### */

__host__ void init_and_render(void)
{
    int device = 0;        // 设置使用第0块GPU进行运算
    cudaSetDevice(device); // 设置运算显卡
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device); // 获取对应设备属性

    unsigned int block_size_width = 32;
    unsigned int block_size_height = 32;
    unsigned int grid_size_width = FRAME_WIDTH / block_size_width;
    unsigned int grid_size_height = FRAME_HEIGHT / block_size_height;
    dim3 dimBlock(block_size_height, block_size_width);
    dim3 dimGrid(grid_size_height, grid_size_width);

    /* ##################################### 随机数初始化 ##################################### */
    curandStateXORWOW *states;
    cudaMalloc((void **)&states, sizeof(curandStateXORWOW) * FRAME_WIDTH * FRAME_HEIGHT);
    initialize_device_random<<<dimGrid, dimBlock>>>(states, time(nullptr), FRAME_WIDTH * FRAME_HEIGHT);
    cudaDeviceSynchronize();
    // curandStateXORWOW *states = init_rand(block_size_width, block_size_height);

    /* ##################################### 摄像机初始化 ##################################### */
    int camera_size = sizeof(camera);
    camera *cpu_camera = createCamera();
    cudaMemcpyToSymbol(PRIMARY_CAMERA, cpu_camera, camera_size);
    cudaDeviceSynchronize();
    // init_camera();

    /* ##################################### 场景初始化 ##################################### */
    hitable **world_device;
    hitable **list_device;
    cudaMalloc((void **)&world_device, 5 * sizeof(hitable *));
    cudaMalloc((void **)&list_device, sizeof(hitable *));
    gen_world<<<1, 1>>>(states, world_device, list_device);
    // hitable **world = init_world(states);

    /* ##################################### 全局渲染入口 ##################################### */
    // 初始化帧缓存
    vec3 *frame_buffer_device;
    float *debug_buffer_device;
    int size = FRAME_WIDTH * FRAME_HEIGHT * sizeof(vec3);
    cudaMalloc((void **)&frame_buffer_device, size);
    cudaMalloc((void **)&debug_buffer_device, FRAME_WIDTH * FRAME_HEIGHT * sizeof(float));
    cuda_shading_unit<<<dimGrid, dimBlock>>>(frame_buffer_device, world_device, states, debug_buffer_device);
    cudaDeviceSynchronize();
    // 在主机开辟 framebuffer 空间
    vec3 *frame_buffer_host = new vec3[FRAME_WIDTH * FRAME_HEIGHT];
    float *debug_buffer_host = new float[FRAME_WIDTH * FRAME_HEIGHT];
    cudaMemcpy(frame_buffer_host, frame_buffer_device, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(debug_buffer_host, debug_buffer_device, FRAME_WIDTH * FRAME_HEIGHT * sizeof(float), cudaMemcpyDeviceToHost);
    // vec3 *frame_buffer_host = new vec3[FRAME_WIDTH * FRAME_HEIGHT];
    // render(block_size_height, block_size_height, states, world, frame_buffer_host);

    std::string debug_path = "./debug.txt";
    std::ofstream OutputDebugText;
    OutputDebugText.open(debug_path);

    for (int row = 0; row < FRAME_HEIGHT; row++)
    {
        for (int col = 0; col < FRAME_WIDTH; col++)
        {
            int global_index = row * FRAME_WIDTH + col;
            OutputDebugText << debug_buffer_host[global_index] << " ";
        }
        OutputDebugText << std::endl;
    }

    std::string file_path = "./any.ppm";
    std::ofstream OutputImage;
    OutputImage.open(file_path);
    OutputImage << "P3\n"
                << FRAME_WIDTH << " " << FRAME_HEIGHT << "\n255\n";

    for (int row = 0; row < FRAME_HEIGHT; row++)
    {
        for (int col = 0; col < FRAME_WIDTH; col++)
        {
            const int global_index = row * FRAME_WIDTH + col;
            vec3 pixelVal = frame_buffer_host[global_index];
            int ir = int(255.99 * pixelVal[0]);
            int ig = int(255.99 * pixelVal[1]);
            int ib = int(255.99 * pixelVal[2]);
            OutputImage << ir << " " << ig << " " << ib << "\n";
        }
    }
}
