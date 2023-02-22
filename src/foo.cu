#include "foo.cuh"
#include "iostream"

//定义核函数 __global__为声明关键字
template<typename T>
__global__ void matAdd_cuda(T *a,T *b,T *sum)
{
	//blockIdx代表block的索引,blockDim代表block的大小，threadIdx代表thread线程的索引，因此对于一维的block和thread索引的计算方式如下
    int i = blockIdx.x*blockDim.x+ threadIdx.x;
    sum[i] = a[i] + b[i];
}


// 核函数用模板不会报错，模板名字是具有链接的，但它们不能具有C链接，因此不能用在供调用的函数上
float *matAdd(float *a, float *b, int length)
{
    int device = 0;        // 设置使用第0块GPU进行运算
    cudaSetDevice(device); // 设置运算显卡
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device);                    // 获取对应设备属性
    int threadMaxSize = devProp.maxThreadsPerBlock;               // 每个线程块的最大线程数
    int blockSize = (length + threadMaxSize - 1) / threadMaxSize; // 计算Block大小,block一维度是最大的，一般不会溢出
    dim3 thread(threadMaxSize);                                   // 设置thread
    dim3 block(blockSize);                                        // 设置block
    int size = length * sizeof(float);                            // 计算空间大小
    float *sum = (float *)malloc(size);                           // 开辟动态内存空间
    // 开辟显存空间
    float *sumGPU, *aGPU, *bGPU;
    cudaMalloc((void **)&sumGPU, size);
    cudaMalloc((void **)&aGPU, size);
    cudaMalloc((void **)&bGPU, size);
    // 内存->显存
    cudaMemcpy((void *)aGPU, (void *)a, size, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)bGPU, (void *)b, size, cudaMemcpyHostToDevice);
    // 运算
    matAdd_cuda<float><<<block, thread>>>(aGPU, bGPU, sumGPU);
    // cudaThreadSynchronize();
    // 显存->内存
    cudaMemcpy(sum, sumGPU, size, cudaMemcpyDeviceToHost);

    std::cout << "sum[1] = " << sum[1] << std::endl;

    // 释放显存
    cudaFree(sumGPU);
    cudaFree(aGPU);
    cudaFree(bGPU);
    return sum;
}
