# 生成 MATH 加速结构相关的库文件

aux_source_directory(. MATH_SRC_LIST)


set(MATH_INCLUDE_DIR_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}
    CACHE INTERNAL "MATH include dir"
)


message(STATUS "MATH include dir = ${MATH_INCLUDE_DIR_LIST}")


add_library(MATH ${MATH_SRC_LIST})
