# cmake_minimum_required(VERSION 3.0)#指定cmake最小版本号
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(renderingCore LANGUAGES CXX)#指定工程名，这个是随意取的

# 注意这里要进行配置，从而生成可调试文件
set(CMAKE_BUILD_TYPE DEBUG)


include_directories(./include)#添加头文件引用路径
#可以添加多个include_directories(xxx)，其中xxx是相对路径


#[[
set(SRC 
xxx
)#定义变量
]]

#link_directories(dir1 dir2)#库文件路径，可以在括号内依次添加多个，使用空格隔开
#路径是相对路径（可以是绝对路径吗 ？）
#相当于指定g++编译器的 -L 参数

add_executable(simpleSceneGene_Cmake #第一个是指定可执行文件的名字 之后参数用分号或者空格进行间隔
./main.cpp
./scene.cpp
./src/math/random.cpp
./src/camera/camera.cpp
./src/object/hitable.cpp
./src/object/geometry/box.cpp
./src/object/geometry/sphere.cpp
./src/object/primitive/primitive.cpp
./src/object/primitive/triangle.cpp
./src/object/model/models.cpp
./src/object/group/hitableList.cpp
./src/material/material.cpp
./src/material/diffuseLight.cpp
./src/texture/textures.cpp
./src/texture/perlin.cpp
./src/accel/bvh.cpp
./src/accel/bounds.cpp
)#指定生成可执行文件名称







