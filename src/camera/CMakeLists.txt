# 生成 CAMERA 加速结构相关的库文件

aux_source_directory(. CAMERA_SRC_LIST)


set(CAMERA_INCLUDE_DIR_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}
    CACHE INTERNAL "CAMERA include dir"
)


message(STATUS "CAMERA include dir = ${CAMERA_INCLUDE_DIR_LIST}")


add_library(CAMERA ${CAMERA_SRC_LIST})
