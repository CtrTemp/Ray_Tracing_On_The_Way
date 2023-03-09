#ifndef TEXTURES_H
#define TEXTURES_H

// 没有这个定义会报错？！@为啥
#define __CUDACC__

#include "utils/vec3.cuh"

// 下面这俩库必须手动添加，并不包含在 <cuda_runtime.h> 中
#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>
// 引入图片必要的stb_image库，这种定义写在头文件中的函数是否必须在cpp文件中引入？
// #define STB_IMAGE_IMPLEMENTATION
// #include "stb_image.h"
#include <fstream>
#include <string>

// texture<uchar4, cudaTextureType2D, cudaReadModeElementType> _texRef2D_;
// texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef2D_image;
#define TEXTURE_WIDTH 5
#define TEXTURE_HEIGHT 3

texture<float, 2> texRef2D_test;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef2D_image_test;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef2D_skybox_test;

// 贴图类 基类
// 注意贴图与材质不同, 可以理解为贴图是材质的一种附加属性, 主要展示材质的"颜色"属性
class textures
{
public:
	__device__ virtual vec3 value(float u, float v, const vec3 &p) const = 0;
};

// 常数纹理贴图（恒定颜色）
class constant_texture : public textures
{
public:
	__device__ constant_texture() = default;
	__device__ constant_texture(vec3 c) : color(c) {}

	__device__ virtual vec3 value(float u, float v, const vec3 &p) const
	{
		return color;
	}

	vec3 color;
};

// 棋盘格纹理贴图
class checker_texture : public textures
{
public:
	__device__ checker_texture() = default;
	__device__ checker_texture(textures *t0, textures *t1) : even(t0), odd(t1) {}
	__device__ virtual vec3 value(float u, float v, const vec3 &p) const
	{
		float sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
		if (sines < 0)
			return odd->value(u, v, p);
		else
			return even->value(u, v, p);
	}

	textures *odd;
	textures *even;
};

// 暂时先不引入图像贴图
class image_texture : public textures
{
public:
	__device__ image_texture() = default;
	__device__ image_texture(unsigned int w, unsigned int h, unsigned int ch, unsigned int texture_index)
	{
		textureWidth = w;
		textureHeight = h;
		channels = ch;
		global_texture_offset = texture_index;
	}
	__device__ virtual vec3 value(float u, float v, const vec3 &p) const
	{

		int col_index = u * textureWidth;
		int row_index = v * textureHeight;

		// printf("index = [%d,%d]", col_index, row_index);
		uchar4 pixel;
		switch (global_texture_offset)
		{
		case 0:
			pixel = tex2D(texRef2D_image_test, row_index, col_index);
			break;
		case 1:
			pixel = tex2D(texRef2D_skybox_test, row_index, col_index);
			break;

		default:
			break;
		}

		vec3 color = vec3((float)(pixel.x) / 256,
						  (float)(pixel.y) / 256,
						  (float)(pixel.z) / 256);

		// printf(" vec = [%d,%d,%d,%d] ", pixel.x, pixel.y, pixel.z, pixel.w);

		return color;
	}
	unsigned long global_texture_offset;
	unsigned int textureWidth;
	unsigned int textureHeight;
	unsigned int channels;
};

#endif

/************************* Perlin Noise Texture **************************/
