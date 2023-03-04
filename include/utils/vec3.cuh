#pragma once
#ifndef VEC3_CUH
#define VEC3_CUH

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <algorithm>

// 添加cuda库
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class vec3
{

public:
	__device__ __host__ vec3() {}
	__device__ __host__ vec3(float e0, float e1, float e2)
	{
		e[0] = e0;
		e[1] = e1;
		e[2] = e2;
	}

	__host__ __device__ inline float x() const { return e[0]; }
	__host__ __device__ inline float y() const { return e[1]; }
	__host__ __device__ inline float z() const { return e[2]; }
	__host__ __device__ inline float r() const { return e[0]; }
	__host__ __device__ inline float g() const { return e[1]; }
	__host__ __device__ inline float b() const { return e[2]; }

	__host__ __device__ inline const vec3 &operator+() const { return *this; }
	__host__ __device__ inline const vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
	__host__ __device__ inline float operator[](int i) const { return e[i]; }
	__host__ __device__ inline float &operator[](int i) { return e[i]; } // 为什么要定义两个这个？！和上面那个有什么区分

	__host__ __device__ inline vec3 &operator+=(const vec3 &v);
	__host__ __device__ inline vec3 &operator-=(const vec3 &v);
	__host__ __device__ inline vec3 &operator*=(const vec3 &v);
	__host__ __device__ inline vec3 &operator/=(const vec3 &v);
	__host__ __device__ inline vec3 &operator*=(const float t);
	__host__ __device__ inline vec3 &operator/=(const float t);
	__host__ __device__ inline bool operator==(const vec3 v);

	__host__ __device__ inline float length() const
	{
		return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
	}

	__host__ __device__ inline float squared_length() const
	{
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}

	static vec3 Min(const vec3 &p1, const vec3 &p2)
	{
		return vec3(fmin(p1.x(), p2.x()), fmin(p1.y(), p2.y()),
					fmin(p1.z(), p2.z()));
	}

	static vec3 Max(const vec3 &p1, const vec3 &p2)
	{
		return vec3(fmax(p1.x(), p2.x()), fmax(p1.y(), p2.y()),
					fmax(p1.z(), p2.z()));
	}

	__host__ __device__ inline void make_unit_vector();

	float e[3]; // 这个类只有这一个数据值
				// 这个类的其他部分都是关于这个数据的构造以及操作
};

__host__ inline std::istream &operator>>(std::istream &is, vec3 &t)
{
	is >> t.e[0] >> t.e[1] >> t.e[2];
	return is;
}

__host__ inline std::ostream &operator<<(std::ostream &os, const vec3 &t)
{
	os << "[" << t.e[0] << ", " << t.e[1] << ", " << t.e[2] << "]";
	return os;
}

__host__ __device__ inline void vec3::make_unit_vector()
{
	float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
}

__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2)
{
	return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2)
{
	return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2)
{
	return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2)
{
	return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3 &v)
{
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator/(vec3 v, float t)
{
	return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline vec3 operator*(const vec3 &v, float t)
{
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2)
{
	return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2)
{
	return vec3(
		(v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
		-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0]),
		(v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

__host__ __device__ inline vec3 &vec3::operator+=(const vec3 &v)
{
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];
	return *this;
}

__host__ __device__ inline vec3 &vec3::operator-=(const vec3 &v)
{
	e[0] -= v.e[0];
	e[1] -= v.e[1];
	e[2] -= v.e[2];
	return *this;
}

__host__ __device__ inline vec3 &vec3::operator*=(const vec3 &v)
{
	e[0] *= v.e[0];
	e[1] *= v.e[1];
	e[2] *= v.e[2];
	return *this;
}

__host__ __device__ inline vec3 &vec3::operator/=(const vec3 &v)
{
	e[0] /= v.e[0];
	e[1] /= v.e[1];
	e[2] /= v.e[2];
	return *this;
}

__host__ __device__ inline vec3 &vec3::operator*=(float t)
{
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}

__host__ __device__ inline vec3 &vec3::operator/=(float t)
{
	e[0] /= t;
	e[1] /= t;
	e[2] /= t;
	return *this;
}

__host__ __device__ inline bool vec3::operator==(const vec3 v)
{
	if (e[0] == v.e[0] && e[1] == v.e[1] && e[2] == v.e[2])
		return true;
	else
		return false;
}

__host__ __device__ inline vec3 unit_vector(vec3 v)
{
	return v / v.length();
}

/*
	色值归一化：
	取大于255的最大的通道色值，将其除以zone作为比例因子
	将三个通道的通道值按比例因子缩小
	得到三个均小于zone的通道值
*/
__host__ __device__ inline vec3 color_unit_normalization(vec3 v, float zone)
{
	float para = 0; // 比例因子
	vec3 return_vec;
	if (v[0] > zone)
		para = zone / v[0]; // 得到一个小于1的值
	else if (v[1] > zone)
		para = zone / v[1];
	else if (v[2] > zone)
		para = zone / v[2];

	else
		return v;

	for (int i = 0; i < 3; ++i)
		return_vec[i] = v[i] * para;

	return return_vec;
}

__host__ __device__ inline vec3 normalized_vec(vec3 vec)
{
	return vec /= vec.length();
}

__host__ inline vec3 random_in_unit_disk()
{
	vec3 p;
	do
	{
		p = 2.0 * vec3(drand48(), drand48(), 0) - vec3(1, 1, 0);
	} while (dot(p, p) >= 1.0);
	// 模拟在方格中撒点，掉入圆圈的点被收录返回
	return p;
}

#endif
