# 生成 TEXTURE 加速结构相关的库文件

aux_source_directory(. TEXTURE_SRC_LIST)


set(TEXTURE_INCLUDE_DIR_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}
    CACHE INTERNAL "TEXTURE include dir"
)


message(STATUS "TEXTURE include dir = ${TEXTURE_INCLUDE_DIR_LIST}")


add_library(TEXTURE ${TEXTURE_SRC_LIST})