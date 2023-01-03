# cmake_minimum_required(VERSION 3.0)#指定cmake最小版本号
cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(renderingCore LANGUAGES CXX)#指定工程名，这个是随意取的

include_directories(./GlobalInclude)#添加头文件引用路径
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
./Source/box.cpp
./Source/camera.cpp
./Source/diffuseLight.cpp
./Source/hitable.cpp
./Source/hitableList.cpp
./Source/material.cpp
./Source/perlin.cpp
./Source/random.cpp
./Source/sphere.cpp
# 一定注意新加入的源文件要在这里进行说明
./Source/triangle.cpp
./Source/triangleList.cpp
./Source/textures.cpp
# ./Source/imageTexture.cpp
)#指定生成可执行文件名称







