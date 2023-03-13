#ifndef TEXTURES_H
#define TEXTURES_H

// 没有这个定义会报错？！@为啥
#define __CUDACC__

#include "utils/vec3.cuh"

// 下面这俩库必须手动添加，并不包含在 <cuda_runtime.h> 中
#include <cuda_texture_types.h>
#include <texture_fetch_functions.h>
// 引入图片必要的stb_image库，这种定义写在头文件中的函数是否必须在cpp文件中引入？
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <fstream>
#include <string>

// texture<uchar4, cudaTextureType2D, cudaReadModeElementType> _texRef2D_;
// texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef2D_image;
#define TEXTURE_WIDTH 5
#define TEXTURE_HEIGHT 3

// texture<float, 2> texRef2D_test;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef2D_SkyBox_Front;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef2D_SkyBox_Back;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef2D_SkyBox_Left;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef2D_SkyBox_Right;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef2D_SkyBox_Up;
texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef2D_SkyBox_Down;
// texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef2D_skybox_test;
// texture<uchar4, cudaTextureType2D, cudaReadModeElementType> texRef2D_ring_lord_test;
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
	__host__ __device__ constant_texture() = default;
	__host__ __device__ constant_texture(vec3 c) : color(c) {}

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
	enum class TextureCategory
	{
		SKYBOX_FRONT,
		SKYBOX_BACK,
		SKYBOX_LEFT,
		SKYBOX_RIGHT,
		SKYBOX_UP,
		SKYBOX_DOWN
	};

public:
	__device__ image_texture() = default;
	__device__ image_texture(unsigned int w, unsigned int h, unsigned int ch, TextureCategory choice)
	{
		textureWidth = w;
		textureHeight = h;
		channels = ch;
		texChoice = choice;
	}
	__device__ virtual vec3 value(float u, float v, const vec3 &p) const
	{
		int col_index = u * textureWidth;
		int row_index = v * textureHeight;

		uchar4 pixel;

		switch (texChoice)
		{
		case TextureCategory::SKYBOX_FRONT:
			// printf("front");
			pixel = tex2D(texRef2D_SkyBox_Front, col_index, row_index);
			break;
		case TextureCategory::SKYBOX_BACK:
			// printf("back");
			pixel = tex2D(texRef2D_SkyBox_Back, col_index, row_index);
			break;
		case TextureCategory::SKYBOX_LEFT:
			// printf("left");
			pixel = tex2D(texRef2D_SkyBox_Left, col_index, row_index);
			break;
		case TextureCategory::SKYBOX_RIGHT:
			pixel = tex2D(texRef2D_SkyBox_Right, col_index, row_index);
			break;
		case TextureCategory::SKYBOX_UP:
			// pixel = tex2D(texRef2D_SkyBox_Up, col_index, row_index);
			break;
		case TextureCategory::SKYBOX_DOWN:
			// pixel = tex2D(texRef2D_SkyBox_Down,  col_index, row_index);
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
	TextureCategory texChoice;
	unsigned int textureWidth;
	unsigned int textureHeight;
	unsigned int channels;
};

__host__ static uchar4 *load_image_texture_host(std::string image_path, int *texWidth, int *texHeight, int *texChannels)
{
	// int texWidth, texHeight, texChannels;
	unsigned char *pixels = stbi_load(image_path.c_str(), texWidth, texHeight, texChannels, STBI_rgb_alpha);
	// size_t imageSize = texWidth * texHeight * 4; // RGB（A） 三（四）通道

	// if (!pixels)
	// {
	// 	throw std::runtime_error("failed to load texture image!");
	// }
	std::cout << "image size = [" << *texWidth << "," << *texHeight << "]" << std::endl;
	std::cout << "image channels = " << *texChannels << std::endl;

	std::string local_confirm_path = "./test_texture_channel.ppm";

	std::ofstream OutputImage;
	OutputImage.open(local_confirm_path);
	OutputImage << "P3\n"
				<< *texWidth << " " << *texHeight << "\n255\n";

	// size_t global_size = (*texWidth) * (*texHeight) * (*texChannels);
	size_t global_size = (*texWidth) * (*texHeight) * (4);
	size_t pixel_num = (*texWidth) * (*texHeight);

	uchar4 *texHost = new uchar4[(*texWidth) * (*texHeight)];

	for (int global_index = 0; global_index < pixel_num; global_index++)
	{
		texHost[global_index].x = pixels[global_index * 4 + 0];
		texHost[global_index].y = pixels[global_index * 4 + 1];
		texHost[global_index].z = pixels[global_index * 4 + 2];
		texHost[global_index].w = pixels[global_index * 4 + 3];
	}

	for (int global_index = 0; global_index < pixel_num; global_index++)
	{
		const int R = static_cast<int>(texHost[global_index].x);
		const int G = static_cast<int>(texHost[global_index].y);
		const int B = static_cast<int>(texHost[global_index].z);
		OutputImage << R << " " << G << " " << B << "\n";
	}

	return texHost;
}

__device__ inline void gen_skybox_vertex_list(vertex **skybox_vert_list, uint32_t **skybox_ind_list, int skybox_half_range)
{
	*skybox_vert_list = new vertex[4];
	*skybox_ind_list = new uint32_t[6];

	vec3 seq_vec_list[6] = {vec3(0, 0, 0),
							vec3(0, 1, 0),
							vec3(1, 1, 0),
							vec3(0, 0, 0),
							vec3(1, 1, 0),
							vec3(1, 0, 0)};

	int s = skybox_half_range;

	(*skybox_vert_list)[0] = vertex(vec3(+s, +s, +s));
	(*skybox_vert_list)[1] = vertex(vec3(+s, +s, -s));
	(*skybox_vert_list)[2] = vertex(vec3(-s, +s, -s));
	(*skybox_vert_list)[3] = vertex(vec3(-s, +s, +s));

	(*skybox_ind_list)[0] = 1;
	(*skybox_ind_list)[1] = 0;
	(*skybox_ind_list)[2] = 3;
	(*skybox_ind_list)[3] = 1;
	(*skybox_ind_list)[4] = 3;
	(*skybox_ind_list)[5] = 2;

	for (int i = 0; i < 6; i++)
	{
		int local_index = (*skybox_ind_list)[i];
		(*skybox_vert_list)[local_index].tex_coord = seq_vec_list[i];
	}

	// (*skybox_vert_list)[0].tex_coord = vec3(0,0,0)
}

#endif

/************************* Perlin Noise Texture **************************/
