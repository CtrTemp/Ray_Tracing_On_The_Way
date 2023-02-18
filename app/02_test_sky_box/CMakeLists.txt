# app 顶层 CMakeList 文件

# 指定当前测试app工程名
project(test_sky_box LANGUAGES CXX)

# 允许GDB调试
set(CMAKE_BUILD_TYPE DEBUG)

aux_source_directory(.  MAIN_SRC_LIST)

add_executable(${PROJECT_NAME} ${MAIN_SRC_LIST})

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/gen_pic")

target_link_libraries(${PROJECT_NAME} ACCEL CAMERA MATERIAL MATH HITABLE TEXTURE)


