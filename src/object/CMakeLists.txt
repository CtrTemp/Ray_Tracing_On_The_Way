# 生成 HITABLE 加速结构相关的库文件

aux_source_directory(. HITABLE_SRC_LIST)
aux_source_directory(./geometry HITABLE_SRC_LIST)
aux_source_directory(./group HITABLE_SRC_LIST)
aux_source_directory(./model HITABLE_SRC_LIST)
aux_source_directory(./primitive HITABLE_SRC_LIST)



# message(STATUS "object ${HITABLE_SRC_LIST}")


set(HITABLE_INCLUDE_DIR_LIST
.
./geometry
./group
./model
./primitive
    CACHE INTERNAL "HITABLE include dir"
)


message(STATUS "HITABLE include dir = ${HITABLE_INCLUDE_DIR_LIST}")


add_library(HITABLE ${HITABLE_SRC_LIST})


