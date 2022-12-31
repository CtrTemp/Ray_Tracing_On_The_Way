#ifndef TEXTURES
#define TEXTURES
//当前文件在GlobalInclude/一级目录下
#include "../GlobalInclude/basic/vec3.h"
//#include "GlobalInclude/vec3.h"



// 贴图类 基类
// 注意贴图与材质不同, 可以理解为贴图是材质的一种附加属性, 主要展示材质的"颜色"属性
class texture {
public:
	virtual vec3 value(float u, float v, const vec3 &p) const = 0;
};


class constant_texture : public texture {
public:
	constant_texture() = default;
	constant_texture(vec3 c) :color(c) {}

	virtual vec3 value(float u, float v, const vec3 &p) const;

	vec3 color;

};


class checker_texture :public texture {
public:
	checker_texture() = default;
	checker_texture(texture *t0, texture *t1) :even(t0), odd(t1) {}
	virtual vec3 value(float u, float v, const vec3& p) const;


	texture *odd;
	texture *even;

};



#endif

/************************* Perlin Noise Texture **************************/






