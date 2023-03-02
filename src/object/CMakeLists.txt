# 生成 HITABLE 加速结构相关的库文件

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
aux_source_directory(. HITABLE_SRC_LIST)
aux_source_directory(./geometry HITABLE_SRC_LIST)
aux_source_directory(./group HITABLE_SRC_LIST)
# aux_source_directory(./model HITABLE_SRC_LIST)
# aux_source_directory(./primitive HITABLE_SRC_LIST)

# 添加头文件列表
set(HITABLE_INCLUDE_DIR_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}
    CACHE INTERNAL "HITABLE include dir"
)

set(HITABLE_INCLUDE_DIR_LIST
.
./geometry
./group
# ./model
# ./primitive
    CACHE INTERNAL "HITABLE include dir"
)

# message(STATUS "HITABLE include dir = ${HITABLE_INCLUDE_DIR_LIST}")

# 选择生成静态库/动态库
cuda_add_library(HITABLE STATIC ${HITABLE_SRC_LIST})
# cuda_add_library(HITABLE SHARED ${HITABLE_SRC_LIST})





