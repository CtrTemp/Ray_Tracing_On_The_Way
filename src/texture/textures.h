#ifndef TEXTURES
#define TEXTURES

#include "utils/vec3.h"
#include <string>

// 贴图类 基类
// 注意贴图与材质不同, 可以理解为贴图是材质的一种附加属性, 主要展示材质的"颜色"属性
class texture
{
public:
	virtual Vector3f value(float u, float v, const Vector3f &p) const = 0;
};

class constant_texture : public texture
{
public:
	constant_texture() = default;
	constant_texture(Vector3f c) : color(c) {}

	virtual Vector3f value(float u, float v, const Vector3f &p) const;

	Vector3f color;
};

class checker_texture : public texture
{
public:
	checker_texture() = default;
	checker_texture(texture *t0, texture *t1) : even(t0), odd(t1) {}
	checker_texture(Vector3f doom);
	virtual Vector3f value(float u, float v, const Vector3f &p) const;

	texture *odd;
	texture *even;
};


class image_texture : public texture
{
public:
	image_texture();
	image_texture(std::string path);
	virtual Vector3f value(float u, float v, const Vector3f &p) const;

	unsigned char *map;
	uint16_t textureWidth;
	uint16_t textureHeight;
	unsigned char channels;
};

#endif

/************************* Perlin Noise Texture **************************/
