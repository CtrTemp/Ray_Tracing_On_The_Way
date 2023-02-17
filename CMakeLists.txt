
# 顶层 CMakeList 文件

# C++ 最低版本
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 指定工程名
project(renderingCore LANGUAGES CXX)

# 允许GDB调试
set(CMAKE_BUILD_TYPE DEBUG)

# 添加头文件引用路径
# 注意，使用 INCLUDE_DIRECTORIES 包含的头文件路径可以被各级子目录中的目标所用，是可以传递下去的
# 而使用 TARGET_INCLUDE_DIRECTORIES 包含的头文件路径只能被特定目标所使用
# 所以对于上层目录中的CMakeList文件，我们只需要将本层可见的include的加入即可，
# 之后的 add_subdirectory 会将顶层目录中的头文件引入下级目录中
include_directories(./include ./src
${ACCEL_INCLUDE_DIR_LIST}
${CAMERA_INCLUDE_DIR_LIST}
${MATERIAL_INCLUDE_DIR_LIST}
${MATH_INCLUDE_DIR_LIST}
${HITABLE_INCLUDE_DIR_LIST}
${TEXTURE_INCLUDE_DIR_LIST}
)

message(STATUS "msg = ${ACCEL_INCLUDE_DIR_LIST}")

aux_source_directory(.  MAIN_SRC_LIST)

#第一个是指定可执行文件的名字 之后参数用分号或者空格进行间隔
add_executable(${PROJECT_NAME} ${MAIN_SRC_LIST})#指定生成可执行文件名称


target_link_libraries(${PROJECT_NAME} ACCEL CAMERA MATERIAL MATH HITABLE TEXTURE)

add_subdirectory(./src/accel)
add_subdirectory(./src/camera)
add_subdirectory(./src/material)
add_subdirectory(./src/math)
add_subdirectory(./src/object)
add_subdirectory(./src/texture)



 







# # cmake_minimum_required(VERSION 3.0)#指定cmake最小版本号
# cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_STANDARD_REQUIRED True)

# project(renderingCore LANGUAGES CXX)#指定工程名，这个是随意取的

# # 注意这里要进行配置，从而生成可调试文件
# set(CMAKE_BUILD_TYPE DEBUG)


# include_directories(./include)#添加头文件引用路径
# #可以添加多个include_directories(xxx)，其中xxx是相对路径


# #[[
# set(SRC 
# xxx
# )#定义变量
# ]]

# #link_directories(dir1 dir2)#库文件路径，可以在括号内依次添加多个，使用空格隔开
# #路径是相对路径（可以是绝对路径吗 ？）
# #相当于指定g++编译器的 -L 参数


# add_executable(simpleSceneGene_Cmake #第一个是指定可执行文件的名字 之后参数用分号或者空格进行间隔
# ./main.cpp
# ./scene.cpp
# ./src/math/random.cpp
# ./src/camera/camera.cpp
# ./src/object/hitable.cpp
# ./src/object/geometry/box.cpp
# ./src/object/geometry/sphere.cpp
# ./src/object/primitive/primitive.cpp
# ./src/object/primitive/triangle.cpp
# ./src/object/model/models.cpp
# ./src/object/group/hitableList.cpp
# ./src/material/material.cpp
# ./src/material/mental.cpp
# ./src/material/lambertian.cpp
# ./src/material/dielectric.cpp
# ./src/material/diffuseLight.cpp
# ./src/texture/textures.cpp
# ./src/texture/perlin.cpp
# ./src/accel/bvh.cpp
# ./src/accel/bounds.cpp
# )#指定生成可执行文件名称




