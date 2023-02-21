#include "material.h"

Vector3f reflect(const Vector3f &v, const Vector3f &n)
{
	return v - 2 * v.dot(n) * n; //获得纯镜面反射光线（方向射线）
}

//折射光，返回是否可以发生折射（是否不发生全反射）
//并通过传引用，隐式的计算出折射光线
bool refract(const Vector3f &v, const Vector3f &n, float ni_over_nt, Vector3f &refracted)
{
	Vector3f uv = v.normalized(); //单位入射光线
	float dt = uv.dot(n);	  //单位入射光线在法线（我们默认是从球心指向外部）上的投影就是入射角的cos值
	float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
	//全反射判断公式，可以自推，其中ni_over_nt是相对折射率，可知小于1时一定是不会发生全反射的
	//（因为是从折射率较小的物质射入折射率较大的物质一定不会发生全反射，这里或许可以优化加速）
	if (discriminant > 0)
	{
		refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
		return true;
	}
	else
		return false;
}

float schlick(float cosine, float ref_idx)
{
	float r0 = (1 - ref_idx) / (1 + ref_idx);
	r0 = r0 * r0;
	return r0 + (1 - r0) * pow((1 - cosine), 5);
}




//虚函数不能在类外定义？！
