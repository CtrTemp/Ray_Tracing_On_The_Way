#ifndef TEXTURES
#define TEXTURES

#include "utils/vec3.h"
#include <string>

// 贴图类 基类
// 注意贴图与材质不同, 可以理解为贴图是材质的一种附加属性, 主要展示材质的"颜色"属性
class texture
{
public:
	virtual vec3 value(float u, float v, const vec3 &p) const = 0;
};

class constant_texture : public texture
{
public:
	constant_texture() = default;
	constant_texture(vec3 c) : color(c) {}

	virtual vec3 value(float u, float v, const vec3 &p) const;

	vec3 color;
};

class checker_texture : public texture
{
public:
	checker_texture() = default;
	checker_texture(texture *t0, texture *t1) : even(t0), odd(t1) {}
	checker_texture(vec3 doom);
	virtual vec3 value(float u, float v, const vec3 &p) const;

	texture *odd;
	texture *even;
};

// 这里有奇怪的bug，！！！！
/*
	问题解决：子类没有实现父类的纯虚函数时，会报错：error: undefined reference to `vtable for xxx`
	注意，它的报错位点不准确，当你直接准备在cpp文件定义其构造函数，且没有在其之前定义实现父类的纯虚函数时，
也会报这个错误，且报错位点在构造函数上，导致一些误导！！！
*/
class image_texture : public texture
{
public:
	image_texture();
	image_texture(std::string path);
	virtual vec3 value(float u, float v, const vec3 &p) const;

	unsigned char *map;
	uint16_t textureWidth;
	uint16_t textureHeight;
	unsigned char channels;
};

#endif

/************************* Perlin Noise Texture **************************/
