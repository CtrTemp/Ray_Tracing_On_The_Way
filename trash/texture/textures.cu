#include "textures.cuh"

// 引入图片必要的stb_image库，这种定义写在头文件中的函数是否必须在cpp文件中引入？
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <fstream>
//常数纹理贴图（恒定颜色）
vec3 constant_texture::value(float u, float v, const vec3 &p) const
{
	return color;
}

//棋盘格纹理贴图
vec3 checker_texture::value(float u, float v, const vec3 &p) const
{
	float sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
	if (sines < 0)
		return odd->value(u, v, p);
	else
		return even->value(u, v, p);
}

// u,v 都是归一化的值，取（0～1）
vec3 image_texture::value(float u, float v, const vec3 &p) const
{

	int index_x = u * textureWidth;
	int index_y = v * textureHeight;
	int index = (index_y * textureWidth + index_x) * static_cast<int>(channels);


	vec3 color = vec3(static_cast<float>(map[index + 0]) / 256,
					  static_cast<float>(map[index + 1]) / 256,
					  static_cast<float>(map[index + 2]) / 256);

	return color;
}

image_texture::image_texture(std::string image_path)
{
	int texWidth, texHeight, texChannels;
	stbi_uc *pixels = stbi_load(image_path.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
	size_t imageSize = texWidth * texHeight * 4; // RGB（A） 三（四）通道

	if (!pixels)
	{
		throw std::runtime_error("failed to load texture image!");
	}


	map = pixels;
	textureWidth = texWidth;
	textureHeight = texHeight;
	// channels = texChannels;
	channels = 4;
	// 通道数强制为4！


	// 注意这里不能free掉，因为你上面的 map 进行的是浅拷贝，仅仅是传递了指针，内存中的值并没有被拷贝过去
	// 稍后这里一定要改成深拷贝而后free掉pixels
	// stbi_image_free(pixels);
}
