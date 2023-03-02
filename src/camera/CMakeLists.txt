# 生成 CAMERA 加速结构相关的库文件

# CMake 版本
cmake_minimum_required(VERSION 3.5)

# packages
find_package(CUDA)

# CUDA warnings added
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)

# nvcc flags 对应当前 GPU 版本优化 + 在生成文件中嵌入 cuda-gdb 调试代码
set(CUDA_NVCC_FLAGS -gencode arch=compute_86,code=sm_86;-G;-g;-O0)

# 添加源文件列表
aux_source_directory(. CAMERA_SRC_LIST)

# 添加头文件列表
set(CAMERA_INCLUDE_DIR_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}
    CACHE INTERNAL "CAMERA include dir"
)

# message(STATUS "CAMERA include dir = ${CAMERA_INCLUDE_DIR_LIST}")

# 选择生成静态库/动态库
cuda_add_library(CAMERA STATIC ${CAMERA_SRC_LIST})
# cuda_add_library(CAMERA SHARED ${CAMERA_SRC_LIST})
