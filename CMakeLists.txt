cmake_minimum_required(VERSION 3.9)#Cmake最低版本
#project(demo LANGUAGES CXX CUDA) #这个写法也是可以的
project(demo)
enable_language(CUDA)#激活CUDA语言支持，使用第一个写法时要进行注释
#设置语言标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)   

#添加项目自身的库和依赖
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)#设置编译后的程序存放位置
aux_source_directory(src SRC_LIST)#添加src目录下的所有源文件
include_directories(include)#添加头文件路径


add_subdirectory(./src/camera)

#编译列表
add_executable(${PROJECT_NAME} ${SRC_LIST})

target_link_libraries(${PROJECT_NAME} CAMERA libglfw.so GL 


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
