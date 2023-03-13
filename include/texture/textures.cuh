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

__device__ static uchar4 get_tex_val_front(int row, int col)
{
	// 又是遇到奇怪问题，这里不定义变量直接返回得到的是一个空值
	// 反应到像素上得到的就是vec3(0,0,0)??? 为啥
	uchar4 pixel;
	pixel = tex2D<uchar4>(texRef2D_SkyBox_Front, col, row);
	return pixel;
}

__device__ static uchar4 get_tex_val_back(int row, int col)
{
	uchar4 pixel;
	pixel = tex2D<uchar4>(texRef2D_SkyBox_Back, col, row);
	return pixel;
}
__device__ static uchar4 get_tex_val_left(int row, int col)
{
	uchar4 pixel;
	pixel = tex2D<uchar4>(texRef2D_SkyBox_Left, col, row);
	return pixel;
}
__device__ static uchar4 get_tex_val_right(int row, int col)
{
	uchar4 pixel;
	pixel = tex2D<uchar4>(texRef2D_SkyBox_Right, col, row);
	return pixel;
}
__device__ static uchar4 get_tex_val_up(int row, int col)
{
	uchar4 pixel;
	pixel = tex2D<uchar4>(texRef2D_SkyBox_Up, col, row);
	return pixel;
}
__device__ static uchar4 get_tex_val_down(int row, int col)
{
	uchar4 pixel;
	pixel = tex2D<uchar4>(texRef2D_SkyBox_Down, col, row);
	return pixel;
}

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
			pixel = get_tex_val_front(row_index, col_index);
			break;
		case TextureCategory::SKYBOX_BACK:
			// 如果这里不使用如下的外部自定义函数嵌套一层，直接使用tex2D的话，超过4次调用则会报错
			// 即使是在这种switch/if分支语句中
			pixel = get_tex_val_back(row_index, col_index);
			// pixel = tex2D(texRef2D_SkyBox_Back, col_index, row_index);
			break;
		case TextureCategory::SKYBOX_LEFT:
			pixel = get_tex_val_left(row_index, col_index);
			break;
		case TextureCategory::SKYBOX_RIGHT:
			pixel = get_tex_val_right(row_index, col_index);
			break;
		case TextureCategory::SKYBOX_UP:
			pixel = get_tex_val_up(row_index, col_index);
			break;
		case TextureCategory::SKYBOX_DOWN:
			pixel = get_tex_val_down(row_index, col_index);
			break;

		default:
			break;
		}

		vec3 color = vec3((float)(pixel.x) / 256,
						  (float)(pixel.y) / 256,
						  (float)(pixel.z) / 256);

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
	*skybox_vert_list = new vertex[24];	 // 顶点在每个面都会被复用
	*skybox_ind_list = new uint32_t[36]; // 每个面都有2个三角形，由6个坐标顺序索引指示

	// 顺序索引指示，imageTexture相对uv坐标值
	vec3 seq_vec_list[6] = {vec3(0, 0, 0),
							vec3(0, 1, 0),
							vec3(1, 1, 0),
							vec3(1, 0, 0),
							vec3(0, 0, 0),
							vec3(1, 1, 0)};

	uint32_t sky_box_front[] = {1, 5, 8, 4, 1, 8};
	uint32_t sky_box_back[] = {3, 7, 6, 2, 3, 6};
	// 左右居然是反过来的？！
	uint32_t sky_box_left[] = {4, 8, 7, 3, 4, 7};
	// uint32_t sky_box_left[] = {2, 6, 5, 1, 2, 5};
	uint32_t sky_box_right[] = {2, 6, 5, 1, 2, 5};
	// uint32_t sky_box_right[] = {4, 8, 7, 3, 4, 7};
	uint32_t sky_box_up[] = {2, 1, 4, 3, 2, 4};
	uint32_t sky_box_down[] = {5, 6, 7, 8, 5, 7};

	skybox_ind_list[0] = sky_box_front;
	skybox_ind_list[6] = sky_box_back;
	skybox_ind_list[12] = sky_box_left;
	skybox_ind_list[18] = sky_box_right;
	skybox_ind_list[24] = sky_box_up;
	skybox_ind_list[30] = sky_box_down;

	for (int i = 0; i < 12; i++)
	{
		printf("seq = %d\n", (*skybox_ind_list)[i]);
	}

	// skybox半径
	int s = skybox_half_range;
	// 八个实际顶点位置坐标
	vertex real_vert_list[8] = {
		vertex(vec3(+s, +s, +s)),
		vertex(vec3(+s, +s, -s)),
		vertex(vec3(-s, +s, -s)),
		vertex(vec3(-s, +s, +s)),
		vertex(vec3(+s, -s, +s)),
		vertex(vec3(+s, -s, -s)),
		vertex(vec3(-s, -s, -s)),
		vertex(vec3(-s, -s, +s))};

	// (*skybox_ind_list)[0] = 1;
	// (*skybox_ind_list)[1] = 0;
	// (*skybox_ind_list)[2] = 3;
	// (*skybox_ind_list)[3] = 1;
	// (*skybox_ind_list)[4] = 3;
	// (*skybox_ind_list)[5] = 2;

	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			printf("i = %d, j = %d, index = %d", i, j, i * 6 + j);
			printf("hah = %d\n", (*skybox_ind_list)[(i * 6 + j)] - 1);
			(*skybox_vert_list)[i * 4 + j] = real_vert_list[(*skybox_ind_list)[i * 6 + j] - 1];
			(*skybox_vert_list)[i * 4 + j].tex_coord = seq_vec_list[j];
		}
	}

	// (*skybox_vert_list)[0].tex_coord = vec3(0,0,0)
}

#endif

/************************* Perlin Noise Texture **************************/
