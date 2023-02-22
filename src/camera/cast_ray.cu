
/**
 *  我们应该考虑如何为C++的类内函数进行加速。
 *  首先一点就是你绝对不能将一个C++类文件直接改写成.cu文件，因为类内函数无法内联成C编译输出，这样在Cpp文件中
 * 调用就会存在未定义的错误。并且换个方向来讲，类内函数会有大量的函数重载，即便可以使用 extern "C" 强制使其
 * 转换成C编译成的函数名称，也会存在大量的函数重定义现象。所以无论如何你不能这样做。
 *  那么如果想使用CUDA加速类内函数，目前的唯一做法就是将具体需要加速的类内函数重新封装一次。具体做法就是原本的
 * 类的定义以及类内函数的实现还是分别使用.cpp以及.h文件实现；对于需要使用CUDA进行加速的类内函数，其具体实现我们
 * 单独放在另一个.cu文件中，并且使用.cuh文件内联成C编译输出，再将这个封装好的函数通过.cuh引入我们的类内函数。
 *  目前对于当前我们的camera.cpp文件，我们对于光线投射的部分用到CUDA加速，那么这部分我们就单独使用这个.cu文件
 * 重新实现一下那个对应的函数即可。
 * */

#include "cast_ray.cuh"
#include "iostream"
#include <sys/time.h>

__global__ void cuda_shading_unit(float *frame_buffer)
{
    // 这里使用二维线程开辟

    int row_index = blockDim.y * blockIdx.y + threadIdx.y; // 当前线程所在行索引
    int col_index = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程所在列索引

    int row_len = gridDim.x * blockDim.x;                 // 行宽（列数）
    int col_len = gridDim.y * blockDim.y;                 // 列高（行数）
    int global_index = (row_len * row_index + col_index); // 全局索引

    float color = (float)global_index / (row_len * col_len);

    // frame_buffer[1] = 1.0;

    float foo = 1.0f;
    for (int i = 0; i < 100000; i++)
    {
        for (int j = 0; j < 100000; j++)
        {
            foo *= foo;
            frame_buffer[global_index] = color;
        }
    }
    frame_buffer[global_index] = color * foo;
}

float *cast_ray_cu(float frame_width, float frame_height, int spp)
{
    int device = 0;        // 设置使用第0块GPU进行运算
    cudaSetDevice(device); // 设置运算显卡
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device); // 获取对应设备属性

    // 整个frame的大小
    int size = frame_width * frame_height * sizeof(float);
    // 开辟将要接收计算回传数据的内存空间
    float *frame_buffer_host = (float *)malloc(size);

    unsigned int block_size_width = 32;
    unsigned int block_size_height = 32;
    unsigned int grid_size_width = frame_width / block_size_width;
    unsigned int grid_size_height = frame_height / block_size_height;

    // std::cout << "grid size = [" << grid_size_height << ", " << grid_size_width << "]" << std::endl;
    // std::cout << "global size = " << size << std::endl;

    dim3 dimBlock(block_size_height, block_size_width, 1);
    dim3 dimGrid(grid_size_height, grid_size_width, 1);

    // 开辟显存空间
    float *frame_buffer_device;
    cudaMalloc((void **)&frame_buffer_device, size);

    // ##################### 这里看一下并行用时 #####################

    cudaEvent_t start, stop;
    float runTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // 不要忘了给模板函数添加模板参数
    cuda_shading_unit<<<dimGrid, dimBlock>>>(frame_buffer_device);

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
