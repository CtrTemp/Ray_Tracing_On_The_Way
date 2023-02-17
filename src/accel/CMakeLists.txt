# 生成 ACCEL 加速结构相关的库文件

aux_source_directory(. ACCEL_SRC_LIST)


set(ACCEL_INCLUDE_DIR_LIST
    ${CMAKE_CURRENT_SOURCE_DIR} # 这个变量是否也可以用 ./ 来代替
    CACHE INTERNAL "accel include dir"
)


message(STATUS "ACCEL include dir = ${ACCEL_INCLUDE_DIR_LIST}")


add_library(ACCEL ${ACCEL_SRC_LIST})
