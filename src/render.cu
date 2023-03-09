#include "render.h"

// 一定要在源文件中进行引入
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

__global__ void cpy_TextureMem_To_DeviceGlobalMem(float *device_mem)
{
    int row_index = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程所在行索引
    int col_index = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程所在列索引
    int row_len = blockDim.x * gridDim.x;
    int global_index = row_len * row_index + col_index;
    /**
     *  注意：
     *  1/ 这里函数调用的 row col 似乎是反过来的
     *  2/ 必须调用cuda提供的 runtime api 才能访问纹理内存中的数据
     * */
    device_mem[global_index] = tex2D(texRef2D_test, col_index, row_index);
}

__host__ void test_texture_mem()
{

    int width = TEXTURE_WIDTH;
    int height = TEXTURE_HEIGHT;

    float *host2D = new float[width * height];    // host 端纹理 buffer
    float *hostRet2D = new float[width * height]; // 返回 host 端纹理 buffer

    cudaArray *cuArray; // CUDA 数组类型定义
    float *devRet2D;    // 显存数据
    int row, col;
    std::cout << " host2D:" << std::endl;
    for (row = 0; row < height; ++row) // 初始化内存原数据
    {
        for (col = 0; col < width; ++col)
        {
            host2D[row * width + col] = row + col;
            std::cout << "  " << host2D[row * width + col] << " ";
        }
        std::cout << std::endl;
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>(); // 这一步是建立映射？？
    cudaMallocArray(&cuArray, &channelDesc, width, height);             // 申请显存空间
    cudaMalloc((void **)&devRet2D, sizeof(float) * width * height);
    cudaBindTextureToArray(texRef2D_test, cuArray); // 将显存数据和纹理绑定
    cudaMemcpyToArray(cuArray, 0, 0, host2D, sizeof(float) * width * height, cudaMemcpyHostToDevice);

    dim3 dimGridTex(1, 1, 1);
    dim3 dimBlockTex(width, height, 1);
    cpy_TextureMem_To_DeviceGlobalMem<<<dimGridTex, dimBlockTex>>>(devRet2D);

    cudaMemcpy(hostRet2D, devRet2D, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    // 打印内存数据
    std::cout << " hostRet2D:" << std::endl;
    for (row = 0; row < height; ++row)
    {
        for (col = 0; col < width; ++col)
            std::cout << "  " << hostRet2D[row * width + col] << " ";
        std::cout << std::endl;
    }
}

// 2D texture 这是一个全局数据，可以像 constant memory 那样在 device 端进行访问
// 注意，这里是一个引用，实际的数据存放在那个CudaArray中
// extern texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef2D_image;

/* #################################### 纹理贴图初始化 #################################### */
__host__ uchar4 *load_image_texture_host(std::string image_path, int *texWidth, int *texHeight, int *texChannels)
{
    // int texWidth, texHeight, texChannels;
    unsigned char *pixels = stbi_load(image_path.c_str(), texWidth, texHeight, texChannels, STBI_rgb_alpha);
    // size_t imageSize = texWidth * texHeight * 4; // RGB（A） 三（四）通道

    if (!pixels)
    {
        throw std::runtime_error("failed to load texture image!");
    }
    std::cout << "image size = [" << *texWidth << "," << *texHeight << "]" << std::endl;
    std::cout << "image channels = " << *texChannels << std::endl;

    std::string local_confirm_path = "./test_texture_channel.ppm";

    std::ofstream OutputImage;
    OutputImage.open(local_confirm_path);
    OutputImage << "P3\n"
                << *texWidth << " " << *texHeight << "\n255\n";

    // size_t global_size = (*texWidth) * (*texHeight) * (*texChannels);
    size_t global_size = (*texWidth) * (*texHeight) * (4);
    size_t pixel_num = (*texWidth) * (*texHeight);

    // for (int i = 0; i < 10; i++)
    // {
    //     std::cout << (int)pixels[i] << std::endl;
    // }

    uchar4 *texHost = new uchar4[(*texWidth) * (*texHeight)];

    for (int global_index = 0; global_index < pixel_num; global_index++)
    {
        texHost[global_index].x = pixels[global_index * 4 + 0];
        texHost[global_index].y = pixels[global_index * 4 + 1];
        texHost[global_index].z = pixels[global_index * 4 + 2];
        texHost[global_index].w = pixels[global_index * 4 + 3];
    }

    for (int global_index = 0; global_index < pixel_num; global_index++)
    {
        const int R = static_cast<int>(texHost[global_index].x);
        const int G = static_cast<int>(texHost[global_index].y);
        const int B = static_cast<int>(texHost[global_index].z);
        OutputImage << R << " " << G << " " << B << "\n";
    }

    return texHost;

    // for (int global_index = 0; global_index < global_size; global_index += 4)
    // {
    //     const int R = static_cast<int>(pixels[global_index + 0]);
    //     const int G = static_cast<int>(pixels[global_index + 1]);
    //     const int B = static_cast<int>(pixels[global_index + 2]);
    //     OutputImage << R << " " << G << " " << B << "\n";
    // }

    // std::cout << "test access img load = " << static_cast<int>(pixels[512 * 512 * 3 + 22]) << std::endl;

    // for (int row = 0; row < *texHeight; row++)
    // {
    //     for (int col = 0; col < *texWidth; col++)
    //     {
    //         const int global_index = (row * (*texWidth) + col) * 4;

    //         int ir = (int)pixels[global_index + 0];
    //         int ig = (int)pixels[global_index + 1];
    //         int ib = (int)pixels[global_index + 2];
    //         OutputImage << ir << " " << ig << " " << ib << "\n";
    //     }
    // }
}

/* ##################################### 随机数初始化 ##################################### */

__global__ void initialize_device_random(curandStateXORWOW *states, unsigned long long seed, size_t size)
{
    int row_index = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程所在行索引
    int col_index = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程所在列索引
    if ((row_index >= FRAME_HEIGHT) || (col_index >= FRAME_WIDTH))
    {
        return;
    }
    int row_len = FRAME_WIDTH; // 行宽（列数）
    // int col_len = FRAME_HEIGHT;                 // 列高（行数）
    int global_index = (row_len * row_index + col_index); // 全局索引

    curand_init(seed, global_index, 0, &states[global_index]);
}

/* ##################################### 摄像机初始化 ##################################### */

__host__ camera *createCamera(void)
{
    cameraCreateInfo createCamera{};
    // createCamera.lookfrom = vec3(-2, 2, 1);
    // createCamera.lookat = vec3(0, 0, -1);
    createCamera.lookfrom = vec3(5, 2, 5);
    createCamera.lookat = vec3(0, 2, 0);

    createCamera.up_dir = vec3(0, 1, 0);
    createCamera.fov = 40;
    createCamera.aspect = float(FRAME_WIDTH) / float(FRAME_HEIGHT);
    createCamera.focus_dist = 10.0; // 这里是焦距
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

__global__ void gen_world(curandStateXORWOW *rand_state, hitable **world, hitable **list)
{

    // 在设备端创建
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        material *noise = new lambertian(new noise_texture(2.5, rand_state));
        material *diffuse_steelblue = new lambertian(new constant_texture(vec3(0.1, 0.2, 0.5)));
        material *mental_copper = new mental(vec3(0.8, 0.6, 0.2), 0.1);
        material *glass = new dielectric(1.5);
        material *light = new diffuse_light(new constant_texture(vec3(6, 6, 6)));
        material *light_red = new diffuse_light(new constant_texture(vec3(70, 0, 0)));
        material *light_green = new diffuse_light(new constant_texture(vec3(0, 70, 0)));
        material *light_blue = new diffuse_light(new constant_texture(vec3(0, 0, 70)));

        // material *image_tex = new diffuse_light(new image_texture(512, 512, 4, 0));
        material *image_tex = new diffuse_light(new image_texture(512, 512, 4, 1));

        // vertex testVertexList[4] = {
        //     {vec3(5.66, 0.1, 0.1), vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0)},
        //     {vec3(0.1, 5.66, 0.1), vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 1, 0)},
        //     {vec3(0.1, 5.66, 8), vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 0, 0)},
        //     {vec3(5.66, 0.1, 8), vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0)}};

        vertex v1(vec3(0.1, 1.414, 1.0), vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0));
        vertex v2(vec3(0.1, 0.1, 1.0), vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 0, 0));
        vertex v3(vec3(1.0, 0.1, 0.1), vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 1, 0));
        vertex v4(vec3(1.0, 1.414, 0.1), vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0));
        triangle t1(v1, v2, v3, light_red);
        triangle t2(v1, v4, v3, light_red);

        int obj_index = 0;

        list[obj_index++] = new sphere(vec3(0, -100.5, -1), 100, noise); // ground
        list[obj_index++] = new triangle(v1, v2, v3, image_tex);
        list[obj_index++] = new triangle(v1, v3, v4, image_tex);
        // list[obj_index++] = new sphere(vec3(0, 0, -1), 0.5, diffuse_steelblue);
        // list[obj_index++] = new sphere(vec3(1, 0, -1), 0.5, mental_copper);
        // list[obj_index++] = new sphere(vec3(-1, 0, -1), -0.45, glass);
        *world = new hitable_list(list, 3);
    }
}
/* ##################################### 光线投射全局渲染 ##################################### */

__device__ ray get_ray_device(float s, float t, curandStateXORWOW *rand_state)
{
    vec3 temp01(1, 2, 3);
    vec3 temp02(3, 2, 1);

    temp02 = -temp01;

    // 全部相机参数
    vec3 u = PRIMARY_CAMERA.u;
    vec3 v = PRIMARY_CAMERA.v;
    float lens_radius = PRIMARY_CAMERA.lens_radius;
    float time0 = PRIMARY_CAMERA.time0, time1 = PRIMARY_CAMERA.time1;
    vec3 origin = PRIMARY_CAMERA.origin;
    vec3 upper_left_conner = PRIMARY_CAMERA.upper_left_conner;
    vec3 horizontal = PRIMARY_CAMERA.horizontal;
    vec3 vertical = PRIMARY_CAMERA.vertical;

    float hor_len = horizontal.length();
    float ver_len = vertical.length();

    vec3 rd = lens_radius * random_in_unit_disk_device(rand_state); // 得到设定光孔大小内的任意散点（即origin点——viewpoint）
    vec3 offset = rd.x() * u + rd.y() * v;                          // origin视点中心偏移（由xoy平面映射到u、v平面）
    offset = vec3(0, 0, 0);                                         // 这里目前有bug，先置为0
    float time = time0 + random_float_device(rand_state) * (time1 - time0);
    return ray(origin + offset, upper_left_conner + s * horizontal + t * vertical - origin - offset);
    // return ray(origin, upper_left_conner + u * horizontal + v * vertical - origin);
}

__device__ vec3 shading_pixel(int depth, const ray &r, hitable **world, curandStateXORWOW *rand_state)
{

    hit_record rec;
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < depth; i++)
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
            else if (rec.mat_ptr->hasEmission())
            {
                return rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
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
__global__ void cuda_shading_unit(vec3 *frame_buffer, hitable **world, curandStateXORWOW *rand_state)
{
    int row_index = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程所在行索引
    int col_index = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程所在列索引

    if ((row_index >= FRAME_HEIGHT) || (col_index >= FRAME_WIDTH))
    {
        return;
    }

    int row_len = FRAME_WIDTH; // 行宽（列数）
    // int col_len = FRAME_HEIGHT;                           // 列高（行数）
    int global_index = (row_len * row_index + col_index); // 全局索引
    curandStateXORWOW local_rand_state = rand_state[global_index];

    vec3 col(0, 0, 0);
    // random_float_device(&local_rand_state)
    for (int s = 0; s < PRIMARY_CAMERA.spp; s++)
    {
        float u = float(col_index + random_float_device(&local_rand_state)) / float(FRAME_WIDTH);
        float v = float(row_index + random_float_device(&local_rand_state)) / float(FRAME_HEIGHT);

        ray kernal_ray = get_ray_device(u, v, &local_rand_state);
        col += shading_pixel(BOUNCE_DEPTH, kernal_ray, world, &local_rand_state);
    }
    rand_state[global_index] = local_rand_state;
    col /= float(PRIMARY_CAMERA.spp);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);

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
    unsigned int grid_size_width = FRAME_WIDTH / block_size_width + 1;
    unsigned int grid_size_height = FRAME_HEIGHT / block_size_height + 1;
    dim3 dimBlock(block_size_width, block_size_height);
    dim3 dimGrid(grid_size_width, grid_size_height);

    /* ##################################### 纹理内存测试 ##################################### */

    test_texture_mem();

    /* ##################################### 纹理导入01 ##################################### */
    std::string test_texture_path = "../Pic/textures/texture.png";
    // std::string test_texture_path = "../Pic/textures/sky0_cube.png";
    uchar4 *texture_host; // = new u_char[TEXTURE_WIDTH * TEXTURE_HEIGHT];
    int texWidth;
    int texHeight;
    int texChannels;
    int texSize;

    texture_host = load_image_texture_host(test_texture_path, &texWidth, &texHeight, &texChannels);
    texSize = texWidth * texHeight * texChannels;

    size_t pixel_num = texWidth * texHeight;

    std::string local_confirm_path = "./test_texture_channel_alloc_verify.ppm";

    std::ofstream OutputImage__;
    OutputImage__.open(local_confirm_path);
    OutputImage__ << "P3\n"
                  << texWidth << " " << texHeight << "\n255\n";

    for (int global_index = 0; global_index < pixel_num; global_index++)
    {
        const int R = static_cast<int>(texture_host[global_index].x);
        const int G = static_cast<int>(texture_host[global_index].y);
        const int B = static_cast<int>(texture_host[global_index].z);
        OutputImage__ << R << " " << G << " " << B << "\n";
    }
    std::cout << "image size = [" << texWidth << "," << texHeight << "]" << std::endl;
    std::cout << "image channels = " << texChannels << std::endl;

    unsigned char *textureDevice;
    cudaMalloc((void **)&textureDevice, sizeof(u_char) * texSize);
    cudaMemcpy(textureDevice, texture_host, sizeof(u_char) * texSize, cudaMemcpyHostToDevice);

    cudaArray *cuArray;                                                  // CUDA 数组类型定义
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>(); // 这一步是建立映射？？
    cudaMallocArray(&cuArray, &channelDesc, texWidth, texHeight);        // 为array申请显存空间
    // 申请显存空间应该是以往的三倍（四通道应该是四倍）
    // 我们应该先对读到的数据进行转PPM验证，从而得出读取到的图像的RGB通道到底是如何排布的
    cudaBindTextureToArray(texRef2D_image_test, cuArray);
    // 将显存数据和纹理绑定，texRef2D_image 实际上是 cuArray 的一个副本引用
    cudaMemcpyToArray(cuArray, 0, 0, texture_host, sizeof(uchar4) * texWidth * texHeight, cudaMemcpyHostToDevice);

    /* ##################################### 纹理导入02 ##################################### */
    
    // std::string test_texture_path = "../Pic/textures/texture.png";
    test_texture_path = "../Pic/textures/sky0_cube.png";

    texture_host = load_image_texture_host(test_texture_path, &texWidth, &texHeight, &texChannels);
    texSize = texWidth * texHeight * texChannels;

    pixel_num = texWidth * texHeight;

    std::cout << "image size = [" << texWidth << "," << texHeight << "]" << std::endl;
    std::cout << "image channels = " << texChannels << std::endl;

    cudaArray *cuArray_sky_test;                                                  // CUDA 数组类型定义
    cudaChannelFormatDesc channelDesc_sky_test = cudaCreateChannelDesc<uchar4>(); // 这一步是建立映射？？
    cudaMallocArray(&cuArray_sky_test, &channelDesc_sky_test, texWidth, texHeight);        // 为array申请显存空间
    // 申请显存空间应该是以往的三倍（四通道应该是四倍）
    // 我们应该先对读到的数据进行转PPM验证，从而得出读取到的图像的RGB通道到底是如何排布的
    cudaBindTextureToArray(texRef2D_skybox_test, cuArray_sky_test);
    // 将显存数据和纹理绑定，texRef2D_image 实际上是 cuArray 的一个副本引用
    cudaMemcpyToArray(cuArray_sky_test, 0, 0, texture_host, sizeof(uchar4) * texWidth * texHeight, cudaMemcpyHostToDevice);

    
    
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
    cudaMalloc((void **)&world_device, 15 * sizeof(hitable *));
    cudaMalloc((void **)&list_device, sizeof(hitable *));
    gen_world<<<1, 1>>>(states, world_device, list_device);
    // hitable **world = init_world(states);

    /* ################################## 初始化 CUDA 计时器 ################################## */
    cudaEvent_t start, stop;
    float time_cost = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* ##################################### 全局渲染入口 ##################################### */
    // 初始化帧缓存
    vec3 *frame_buffer_device;
    int size = FRAME_WIDTH * FRAME_HEIGHT * sizeof(vec3);
    cudaMalloc((void **)&frame_buffer_device, size);
    cudaEventRecord(start); // device端 开始计时
    cuda_shading_unit<<<dimGrid, dimBlock>>>(frame_buffer_device, world_device, states);
    cudaEventRecord(stop); // device端 计时结束
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop); // 计时同步

    cudaEventElapsedTime(&time_cost, start, stop); // 计算用时，单位为ms
    // 停止计时
    std::cout << ": The total time of the pirmary loop is: " << time_cost << "ms" << std::endl;

    /* #################################### host端写图像文件 #################################### */

    // 在主机开辟 framebuffer 空间
    vec3 *frame_buffer_host = new vec3[FRAME_WIDTH * FRAME_HEIGHT];
    cudaMemcpy(frame_buffer_host, frame_buffer_device, size, cudaMemcpyDeviceToHost);
    // vec3 *frame_buffer_host = new vec3[FRAME_WIDTH * FRAME_HEIGHT];
    // render(block_size_height, block_size_height, states, world, frame_buffer_host);

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

    std::cout << "Render Loop ALL DONE" << std::endl;
}
