#include "render.h"
#define CUDA_LAUNCH_BLOCKING

/* #################################### 纹理贴图初始化 #################################### */
__host__ static void import_tex()
{
    std::string test_texture_path;
    uchar4 *texture_host;
    int texWidth;
    int texHeight;
    int texChannels;
    int texSize;
    size_t pixel_num;

    /* ##################################### Skybox-Front ##################################### */
    test_texture_path = "../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_0_Front+Z.png";
    texture_host = load_image_texture_host(test_texture_path, &texWidth, &texHeight, &texChannels);
    texWidth = 2048;
    texHeight = 2048;
    texChannels = 4;
    texSize = texWidth * texHeight * texChannels;
    pixel_num = texWidth * texHeight;

    cudaArray *cuArray_skybox_front;                                                        // CUDA 数组类型定义
    cudaChannelFormatDesc channelDesc_skybox_front = cudaCreateChannelDesc<uchar4>();       // 这一步是建立映射？？
    cudaMallocArray(&cuArray_skybox_front, &channelDesc_skybox_front, texWidth, texHeight); // 为array申请显存空间
    cudaBindTextureToArray(texRef2D_SkyBox_Front, cuArray_skybox_front);
    cudaMemcpyToArray(cuArray_skybox_front, 0, 0, texture_host, sizeof(uchar4) * texWidth * texHeight, cudaMemcpyHostToDevice);
    /* ##################################### Skybox-Back ##################################### */
    test_texture_path = "../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_1_Back-Z.png";
    texture_host = load_image_texture_host(test_texture_path, &texWidth, &texHeight, &texChannels);
    texSize = texWidth * texHeight * texChannels;
    pixel_num = texWidth * texHeight;

    cudaArray *cuArray_skybox_back;                                                       // CUDA 数组类型定义
    cudaChannelFormatDesc channelDesc_skybox_back = cudaCreateChannelDesc<uchar4>();      // 这一步是建立映射？？
    cudaMallocArray(&cuArray_skybox_back, &channelDesc_skybox_back, texWidth, texHeight); // 为array申请显存空间
    cudaBindTextureToArray(texRef2D_SkyBox_Back, cuArray_skybox_back);
    cudaMemcpyToArray(cuArray_skybox_back, 0, 0, texture_host, sizeof(uchar4) * texWidth * texHeight, cudaMemcpyHostToDevice);

    /* ##################################### Skybox-Left ##################################### */
    test_texture_path = "../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_2_Left+X.png";
    texture_host = load_image_texture_host(test_texture_path, &texWidth, &texHeight, &texChannels);
    texSize = texWidth * texHeight * texChannels;
    pixel_num = texWidth * texHeight;

    cudaArray *cuArray_skybox_left;                                                       // CUDA 数组类型定义
    cudaChannelFormatDesc channelDesc_skybox_left = cudaCreateChannelDesc<uchar4>();      // 这一步是建立映射？？
    cudaMallocArray(&cuArray_skybox_left, &channelDesc_skybox_left, texWidth, texHeight); // 为array申请显存空间
    cudaBindTextureToArray(texRef2D_SkyBox_Left, cuArray_skybox_left);
    cudaMemcpyToArray(cuArray_skybox_left, 0, 0, texture_host, sizeof(uchar4) * texWidth * texHeight, cudaMemcpyHostToDevice);

    /* ##################################### Skybox-Right ##################################### */
    test_texture_path = "../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_3_Right-X.png";
    texture_host = load_image_texture_host(test_texture_path, &texWidth, &texHeight, &texChannels);
    texSize = texWidth * texHeight * texChannels;
    pixel_num = texWidth * texHeight;

    cudaArray *cuArray_skybox_right;                                                        // CUDA 数组类型定义
    cudaChannelFormatDesc channelDesc_skybox_right = cudaCreateChannelDesc<uchar4>();       // 这一步是建立映射？？
    cudaMallocArray(&cuArray_skybox_right, &channelDesc_skybox_right, texWidth, texHeight); // 为array申请显存空间
    cudaBindTextureToArray(texRef2D_SkyBox_Right, cuArray_skybox_right);
    cudaMemcpyToArray(cuArray_skybox_right, 0, 0, texture_host, sizeof(uchar4) * texWidth * texHeight, cudaMemcpyHostToDevice);

    /* ##################################### Skybox-Up ##################################### */
    test_texture_path = "../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_4_Up+Y.png";
    texture_host = load_image_texture_host(test_texture_path, &texWidth, &texHeight, &texChannels);
    texSize = texWidth * texHeight * texChannels;
    pixel_num = texWidth * texHeight;

    cudaArray *cuArray_skybox_up;                                                     // CUDA 数组类型定义
    cudaChannelFormatDesc channelDesc_skybox_up = cudaCreateChannelDesc<uchar4>();    // 这一步是建立映射？？
    cudaMallocArray(&cuArray_skybox_up, &channelDesc_skybox_up, texWidth, texHeight); // 为array申请显存空间
    cudaBindTextureToArray(texRef2D_SkyBox_Up, cuArray_skybox_up);
    cudaMemcpyToArray(cuArray_skybox_up, 0, 0, texture_host, sizeof(uchar4) * texWidth * texHeight, cudaMemcpyHostToDevice);

    /* ##################################### Skybox-Down ##################################### */
    test_texture_path = "../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_2_Left+X.png";
    texture_host = load_image_texture_host(test_texture_path, &texWidth, &texHeight, &texChannels);
    texSize = texWidth * texHeight * texChannels;
    pixel_num = texWidth * texHeight;

    cudaArray *cuArray_skybox_down;                                                       // CUDA 数组类型定义
    cudaChannelFormatDesc channelDesc_skybox_down = cudaCreateChannelDesc<uchar4>();      // 这一步是建立映射？？
    cudaMallocArray(&cuArray_skybox_down, &channelDesc_skybox_down, texWidth, texHeight); // 为array申请显存空间
    cudaBindTextureToArray(texRef2D_SkyBox_Down, cuArray_skybox_down);
    cudaMemcpyToArray(cuArray_skybox_down, 0, 0, texture_host, sizeof(uchar4) * texWidth * texHeight, cudaMemcpyHostToDevice);
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
    // createCamera.lookfrom = vec3(10, 8, 10);
    // createCamera.lookat = vec3(0, 1, 0);
    // // 纹理贴图最佳视点
    // createCamera.lookfrom = vec3(4, 2, 4);
    // createCamera.lookat = vec3(0, 1, 0);
    // // bunny模型导入最佳视点
    // createCamera.lookfrom = vec3(2, 2, 2);
    // createCamera.lookat = vec3(0.25, 0, 0.25);
    // skybox 测试视点
    createCamera.lookfrom = vec3(0, 0, 0);
    createCamera.lookat = vec3(1, 1, 1);

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
// 最后两个参数是需要创建的 models，需要时，应该在host端预先对其进行初始化，并在device端进行空间分配/拷贝
__global__ void gen_world(curandStateXORWOW *rand_state, hitable **world, hitable **list, vertex *vertList, uint32_t *indList, int *vertOffset, int *indOffset, int model_counts)
{
    // 在设备端创建
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        // 一般表面材质/纹理
        material *noise = new lambertian(new noise_texture(2.5, rand_state));
        material *diffuse_steelblue = new lambertian(new constant_texture(vec3(0.1, 0.2, 0.5)));
        material *mental_copper = new mental(vec3(0.8, 0.6, 0.2), 0.001);
        material *glass = new dielectric(1.5);
        material *light = new diffuse_light(new constant_texture(vec3(6, 6, 6)));
        material *light_red = new diffuse_light(new constant_texture(vec3(70, 0, 0)));
        material *light_green = new diffuse_light(new constant_texture(vec3(0, 70, 0)));
        material *light_blue = new diffuse_light(new constant_texture(vec3(0, 0, 70)));

        // 纹理贴图
        material *image_sky_tex_front = new lambertian(new image_texture(2048, 2048, 4, image_texture::TextureCategory::SKYBOX_FRONT));
        material *image_sky_tex_back = new lambertian(new image_texture(2048, 2048, 4, image_texture::TextureCategory::SKYBOX_BACK));
        material *image_sky_tex_left = new lambertian(new image_texture(2048, 2048, 4, image_texture::TextureCategory::SKYBOX_LEFT));
        material *image_sky_tex_right = new lambertian(new image_texture(2048, 2048, 4, image_texture::TextureCategory::SKYBOX_RIGHT));
        material *image_sky_tex_up = new lambertian(new image_texture(2048, 2048, 4, image_texture::TextureCategory::SKYBOX_UP));
        material *image_sky_tex_down = new lambertian(new image_texture(2048, 2048, 4, image_texture::TextureCategory::SKYBOX_DOWN));

        // 如果没有这些语句，将会出现很大问题，后面的世界可以生成，但不能正确运行
        // 将以下的关于纹理贴图的顶点创建注释掉，你将可以复现这个问题
        vertex v1_statue(vec3(0.5, 2.0, 0.1), vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0));
        vertex v2_statue(vec3(0.5, 0.1, 0.1), vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 0, 0));
        vertex v3_statue(vec3(2.5, 0.1, 0.0), vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 1, 0));
        vertex v4_statue(vec3(2.5, 2.0, 0.0), vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0));

        vertex v1_ring(vec3(0.1, 2.0, 0.5), vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0));
        vertex v2_ring(vec3(0.1, 0.1, 0.5), vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 0, 0));
        vertex v3_ring(vec3(0.1, 0.1, 2.5), vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 1, 0));
        vertex v4_ring(vec3(0.1, 2.0, 2.5), vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0));

        vertex v1_skybox(vec3(0.1, 2.0, 0.5), vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0));

        vertex *skybox_vert_list;
        uint32_t *skybox_ind_list;
        gen_skybox_vertex_list(&skybox_vert_list, &skybox_ind_list, 5);
        printf("texture Imported done\n");

        int obj_index = 0;

        // list[obj_index++] = new sphere(vec3(0, -100.5, -1), 100, noise); // ground
        // list[obj_index++] = new triangle(v1_statue, v2_statue, v3_statue, image_statue_tex);
        // list[obj_index++] = new triangle(v1_statue, v3_statue, v4_statue, image_statue_tex);
        // list[obj_index++] = new triangle(v1_ring, v2_ring, v3_ring, image_ring_lord_tex);
        // list[obj_index++] = new triangle(v1_ring, v3_ring, v4_ring, image_ring_lord_tex);
        // list[obj_index++] = new sphere(vec3(0, 0, -1), 0.5, diffuse_steelblue);
        // list[obj_index++] = new sphere(vec3(1, 0, -1), 0.5, mental_copper);
        // list[obj_index++] = new sphere(vec3(-1, 0, -1), -0.45, glass);
        uint32_t sky_box_ind_list[] = {1, 0, 3, 2, 1, 3};
        list[obj_index++] = new models(skybox_vert_list + 0, sky_box_ind_list, 6, image_sky_tex_front, models::HitMethod::NAIVE, models::PrimType::TRIANGLE);
        list[obj_index++] = new models(skybox_vert_list + 4, sky_box_ind_list, 6, image_sky_tex_back, models::HitMethod::NAIVE, models::PrimType::TRIANGLE);
        list[obj_index++] = new models(skybox_vert_list + 8, sky_box_ind_list, 6, image_sky_tex_left, models::HitMethod::NAIVE, models::PrimType::TRIANGLE);
        list[obj_index++] = new models(skybox_vert_list + 12, sky_box_ind_list, 6, image_sky_tex_right, models::HitMethod::NAIVE, models::PrimType::TRIANGLE);
        list[obj_index++] = new models(skybox_vert_list + 16, sky_box_ind_list, 6, image_sky_tex_up, models::HitMethod::NAIVE, models::PrimType::TRIANGLE);
        list[obj_index++] = new models(skybox_vert_list + 20, sky_box_ind_list, 6, image_sky_tex_down, models::HitMethod::NAIVE, models::PrimType::TRIANGLE);
        // list[obj_index++] = new models(vertList, indList, 13500, mental_copper, models::HitMethod::NAIVE, models::PrimType::TRIANGLE);

        // printf("models count = %d\n", model_counts);
        // for (int models_index = 0; models_index < model_counts; models_index++)
        // {
        //     int model_ind_len = indOffset[models_index + 1] - indOffset[models_index + 0];
        //     printf("modelLen = %d\n", model_ind_len);
        //     list[obj_index++] = new models(&(vertList[vertOffset[models_index]]), &(indList[indOffset[models_index]]), model_ind_len, diffuse_steelblue, models::HitMethod::NAIVE, models::PrimType::TRIANGLE);
        // }
        // int models_index = 0;
        // list[obj_index++] = new models(&(vertList[vertOffset[models_index]]), &(indList[indOffset[models_index]]), indOffset[models_index + 1] - indOffset[models_index + 0], mental_copper, models::HitMethod::NAIVE, models::PrimType::TRIANGLE);
        // models_index++;
        // list[obj_index++] = new models(&(vertList[vertOffset[models_index]]), &(indList[indOffset[models_index]]), indOffset[models_index + 1] - indOffset[models_index + 0], glass, models::HitMethod::NAIVE, models::PrimType::TRIANGLE);
        // models_index++;
        // list[obj_index++] = new models(&(vertList[vertOffset[models_index]]), &(indList[indOffset[models_index]]), indOffset[models_index + 1] - indOffset[models_index + 0], diffuse_steelblue, models::HitMethod::NAIVE, models::PrimType::TRIANGLE);

        *world = new hitable_list(list, obj_index);
        printf("world generate done\n");
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

    /* ##################################### 纹理导入01 ##################################### */
    import_tex();

    /* ################################### 模型文件导入01 ################################### */
    vertex *vertList_host;
    uint32_t *indList_host;
    int *vertex_offset_host;
    int *ind_offset_host;
    std::vector<std::string> models_paths_host;

    // models_paths_host.push_back("../Models/bunny/bunny_low_resolution.obj");
    // models_paths_host.push_back("../Models/bunny/bunny_x.obj");
    // models_paths_host.push_back("../Models/bunny/bunny_z.obj");

    // import_obj_from_file(&vertList_host, &vertex_offset_host, &indList_host, &ind_offset_host, models_paths_host);

    // size_t vert_len = vertex_offset_host[models_paths_host.size()];
    // size_t ind_len = ind_offset_host[models_paths_host.size()];

    vertex *vertList_device;
    uint32_t *indList_device;
    int *vertex_offset_device;
    int *ind_offset_device;

    // cudaMalloc((void **)&vertList_device, vert_len * sizeof(vertex));
    // cudaMalloc((void **)&indList_device, ind_len * sizeof(uint32_t));
    // cudaMalloc((void **)&vertex_offset_device, (models_paths_host.size() + 1) * sizeof(int));
    // cudaMalloc((void **)&ind_offset_device, (models_paths_host.size() + 1) * sizeof(int));

    // cudaMemcpy(vertList_device, vertList_host, vert_len * sizeof(vertex), cudaMemcpyHostToDevice);
    // cudaMemcpy(indList_device, indList_host, ind_len * sizeof(uint32_t), cudaMemcpyHostToDevice);
    // cudaMemcpy(vertex_offset_device, vertex_offset_host, (models_paths_host.size() + 1) * sizeof(int), cudaMemcpyHostToDevice);
    // cudaMemcpy(ind_offset_device, ind_offset_host, (models_paths_host.size() + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // cudaDeviceSynchronize();

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
    gen_world<<<1, 1>>>(states, world_device, list_device, vertList_device, indList_device, vertex_offset_device, ind_offset_device, models_paths_host.size());
    // hitable **world = init_world(states);
    cudaDeviceSynchronize();

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
