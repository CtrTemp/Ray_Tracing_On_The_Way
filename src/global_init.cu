#include "global_init.cuh"

/**
 *  该文件用于用户的场景创建/摄像机拜访等基本的初始化操作。一些应该被导入到device端的常用变量在此被创建，并被声明
 * 为全局变量，所有位置均可访问。
 *
 * */
// extern __constant__ camera PRIMARY_CAMERA;

void global_initialization(void)
{
    int device = 0;        // 设置使用第0块GPU进行运算
    cudaSetDevice(device); // 设置运算显卡
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device); // 获取对应设备属性

    /* ############################### 初始化摄像机 ############################### */
    int camera_size = sizeof(camera);

    // std::cout << "camera size = " << camera_size << std::endl;
    camera *cpu_camera = createCamera();

    // 将host本地创建初始化好的摄像机，连带参数一同拷贝到device设备端
    cudaMemcpyToSymbol(PRIMARY_CAMERA, cpu_camera, camera_size);

    // std::cout << "camera height = " << cpu_camera->frame_height << std::endl;

    /* ############################### 初始化场景 ############################### */
}

camera *get_camera_info(void)
{

    int device = 0;        // 设置使用第0块GPU进行运算
    cudaSetDevice(device); // 设置运算显卡
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device); // 获取对应设备属性

    /* ############################### 初始化摄像机 ############################### */
    int camera_size = sizeof(camera);

    camera *cpu_camera;

    // 将host本地创建初始化好的摄像机，连带参数一同拷贝到device设备端
    cudaMemcpyFromSymbol(cpu_camera, PRIMARY_CAMERA, camera_size);

    // std::cout << "camera height fetch back = " << cpu_camera->frame_height << std::endl;
    return cpu_camera;
}

camera *createCamera(void)
{
    cameraCreateInfo createCamera{};

    createCamera.lookfrom = vec3(20, 15, 20);
    createCamera.lookat = vec3(0, 0, 0);

    createCamera.up_dir = vec3(0, 1, 0);
    createCamera.fov = 40;
    createCamera.aspect = float(FRAME_WIDTH) / float(FRAME_HEIGHT);
    createCamera.focus_dist = 10.0;
    createCamera.t0 = 0.0;
    createCamera.t1 = 1.0;
    createCamera.frame_width = FRAME_WIDTH;
    createCamera.frame_height = FRAME_HEIGHT;

    createCamera.spp = 1;

    // 学会像vulkan那样构建
    return new camera(createCamera);
}
