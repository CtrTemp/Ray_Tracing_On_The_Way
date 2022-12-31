#include "../GlobalInclude/sphere.h"

bool sphere::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{
	vec3 oc = r.origin() - center;
	float a = dot(r.direction(), r.direction());
	float b = 2.0 * dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - 4 * a * c;
	//这里是不是写错了？不应该是b*b-4ac么
	//不错，已经进行了修改，就应该是b*b - 4*a*c

	//以下有一个优先返回原则：优先返回双解中离观察点（射线发射点）最近的击中点
	//注意！我们是传引用，在函数中就直接可以改变 rec结构体变量 的各类值
	if (discriminant > 0)
	{
		float temp = (-b - sqrt(discriminant)) / a / 2;
		if (temp < t_max && temp > t_min)
		{
			rec.t = temp;							//得到击中点的 t值 并储存入record
			rec.p = r.point_at_parameter(rec.t);	//得到击中点的坐标 并储存人record
			rec.normal = (rec.p - center) / radius; //得到击中点的单位法向向量
			rec.mat_ptr = this->mat_ptr;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a / 2;
		if (temp < t_max && temp > t_min)
		{
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = this->mat_ptr;
			return true;
		}
	}
	return false;
}

bool sphere::bounding_box(float t0, float t1, aabb &box) const
{
	box = aabb(center - vec3(radius, radius, radius), center + vec3(radius, radius, radius));
	// 修改指针指向的目标,来隐式返回bounding_box
	return true;
}

vec3 moving_sphere::center(float time) const
{
	return center0 + ((time - time0) / (time1 - time0) * (center1 - center0));
}

bool moving_sphere::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{

	vec3 oc = r.origin() - center(r.time());
	float a = dot(r.direction(), r.direction());
	float b = 2.0 * dot(oc, r.direction());
	float c = dot(oc, oc) - radius * radius;
	float discriminant = b * b - 4 * a * c;

	//以下有一个优先返回原则：优先返回双解中离观察点（射线发射点）最近的击中点
	//注意！我们是传引用，在函数中就直接可以改变 rec结构体变量 的各类值
	if (discriminant > 0)
	{
		float temp = (-b - sqrt(discriminant)) / a / 2;
		if (temp < t_max && temp > t_min)
		{
			rec.t = temp;									  //得到击中点的 t值 并储存入record
			rec.p = r.point_at_parameter(rec.t);			  //得到击中点的坐标 并储存人record
			rec.normal = (rec.p - center(r.time())) / radius; //得到击中点的单位法向向量
			rec.mat_ptr = this->mat_ptr;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a / 2;
		if (temp < t_max && temp > t_min)
		{
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center(r.time())) / radius;
			rec.mat_ptr = this->mat_ptr;
			return true;
		}
	}
	return false;
}
