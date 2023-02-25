# 生成 MATERIAL 加速结构相关的库文件

aux_source_directory(. MATERIAL_SRC_LIST)


set(MATERIAL_INCLUDE_DIR_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}
    CACHE INTERNAL "MATERIAL include dir"
)


message(STATUS "MATERIAL include dir = ${MATERIAL_INCLUDE_DIR_LIST}")


add_library(MATERIAL ${MATERIAL_SRC_LIST})
