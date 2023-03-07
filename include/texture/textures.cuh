#ifndef TEXTURES_H
#define TEXTURES_H

#include "utils/vec3.cuh"
// 引入图片必要的stb_image库，这种定义写在头文件中的函数是否必须在cpp文件中引入？
// #define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <fstream>
#include <string>

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

// // 暂时先不引入图像贴图
// class image_texture : public textures
// {
// public:
// 	image_texture() = default;
// 	image_texture(std::string image_path)
// 	{
// 		int texWidth, texHeight, texChannels;
// 		stbi_uc *pixels = stbi_load(image_path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
// 		size_t imageSize = texWidth * texHeight * 4; // RGB（A） 三（四）通道

// 		if (!pixels)
// 		{
// 			throw std::runtime_error("failed to load texture image!");
// 		}

// 		map = pixels;
// 		textureWidth = texWidth;
// 		textureHeight = texHeight;
// 		// channels = texChannels;
// 		channels = 4;
// 		// 通道数强制为4！
		

// 		// 注意这里不能free掉，因为你上面的 map 进行的是浅拷贝，仅仅是传递了指针，内存中的值并没有被拷贝过去
// 		// 稍后这里一定要改成深拷贝而后free掉pixels
// 		// stbi_image_free(pixels);
// 	}
// 	virtual vec3 value(float u, float v, const vec3 &p) const
// 	{

// 		int index_x = u * textureWidth;
// 		int index_y = v * textureHeight;
// 		int index = (index_y * textureWidth + index_x) * static_cast<int>(channels);

// 		vec3 color = vec3(static_cast<float>(map[index + 0]) / 256,
// 						  static_cast<float>(map[index + 1]) / 256,
// 						  static_cast<float>(map[index + 2]) / 256);

// 		return color;
// 	}

// 	unsigned char *map;
// 	uint16_t textureWidth;
// 	uint16_t textureHeight;
// 	unsigned char channels;
// };

#endif

/************************* Perlin Noise Texture **************************/
