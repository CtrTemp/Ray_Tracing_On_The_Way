cmake_minimum_required(VERSION 3.9)#Cmake最低版本

project(demo)

# packages
find_package(CUDA)

# 注意 “不要进行代码优化！！！使用-O选项，否则默认为-O2”
set(CUDA_NVCC_FLAGS -gencode arch=compute_86,code=sm_86;-G;-g;-O0)
set(CMAKE_NVCC_FLAGS "CMAKE_NVCC_FLAGS -gencode arch=compute_86,code=sm_86;-G;-g;-O0")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    # set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)

#添加项目自身的库和依赖
aux_source_directory(src SRC_LIST)#添加src目录下的所有源文件
aux_source_directory(src/server SRC_LIST)#添加src/server目录下的所有源文件
# aux_source_directory(src/camera SRC_LIST)
# aux_source_directory(src/material SRC_LIST)
# aux_source_directory(src/object SRC_LIST)
# aux_source_directory(src/object/geometry SRC_LIST)
# aux_source_directory(src/object/group SRC_LIST)

message(STATUS "SRC_LIST = ${SRC_LIST}")

include_directories(./include ./src
# ${ACCEL_INCLUDE_DIR_LIST}
# ${CAMERA_INCLUDE_DIR_LIST}
# ${MATERIAL_INCLUDE_DIR_LIST}
# ${MATH_INCLUDE_DIR_LIST}
# ${HITABLE_INCLUDE_DIR_LIST}
# ${TEXTURE_INCLUDE_DIR_LIST}
)


# add_subdirectory(./src/accel)
# add_subdirectory(./src/model)
# add_subdirectory(./src/camera)
# add_subdirectory(./src/material)
# add_subdirectory(./src/object)
# add_subdirectory(./src/texture)

#编译列表
CUDA_ADD_EXECUTABLE(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} libglfw.so GL libjsoncpp.so libprotobuf.a pthread base64

opencv_calib3d
opencv_core
opencv_dnn
opencv_features2d
opencv_flann
opencv_gapi
opencv_highgui
opencv_imgcodecs
opencv_imgproc
opencv_ml
opencv_objdetect
opencv_photo
opencv_stitching
opencv_videoio
opencv_video
)
