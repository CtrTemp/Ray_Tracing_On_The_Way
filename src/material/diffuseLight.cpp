#include "material/diffuse_light.h"

bool diffuse_light::scatter(const ray &r_in, const hit_record &rec, vec3 &attenuated, ray &scattered) const
{
	return false;
}

vec3 diffuse_light::emitted(float u, float v, const vec3 &p) const
{
	return emit->value(u, v, p);
}

bool xy_rect::bounding_box(float t0, float t1, aabb &box) const
{
	box = aabb(vec3(x0, y0, k - 0.0001), vec3(x1, y1, k + 0.0001));
	return true;
}

aabb xy_rect::getBound(void) const
{
	return aabb(vec3(x0, y0, k - 0.0001), vec3(x1, y1, k + 0.0001));
}

bool xz_rect::bounding_box(float t0, float t1, aabb &box) const
{
	box = aabb(vec3(x0, k - 0.0001, z0), vec3(x1, k + 0.0001, z1));
	return true;
}

aabb xz_rect::getBound(void) const
{
	return aabb(vec3(x0, k - 0.0001, z0), vec3(x1, k + 0.0001, z1));
}

bool yz_rect::bounding_box(float t0, float t1, aabb &box) const
{
	box = aabb(vec3(k - 0.0001, y0, z0), vec3(k + 0.0001, y1, z1));
	return true;
}

aabb yz_rect::getBound(void) const
{
	return aabb(vec3(k - 0.0001, y0, z0), vec3(k + 0.0001, y1, z1));
}

bool xy_rect::hit(const ray &r, float t0, float t1, hit_record &rec) const
{
	float t = (k - r.origin().z()) / r.direction().z();
	if (t < t0 || t > t1)
		return false;
	float x = r.origin().x() + t * r.direction().x();
	float y = r.origin().y() + t * r.direction().y();

	if (x < x0 || x > x1 || y < y0 || y > y1)
		return false;

	rec.u = (x - x0) / (x1 - x0);
	rec.v = (y - y0) / (y1 - y0);

	rec.t = t;
	rec.mat_ptr = mp;
	rec.p = r.point_at_parameter(t);
	rec.normal = vec3(0, 0, 1);

	return true;
}

bool xz_rect::hit(const ray &r, float t0, float t1, hit_record &rec) const
{
	float t = (k - r.origin().y()) / r.direction().y();
	if (t < t0 || t > t1)
		return false;
	float x = r.origin().x() + t * r.direction().x();
	float z = r.origin().z() + t * r.direction().z();

	if (x < x0 || x > x1 || z < z0 || z > z1)
		return false;

	rec.u = (x - x0) / (x1 - x0);
	rec.v = (z - z0) / (z1 - z0);

	rec.t = t;
	rec.mat_ptr = mp;
	rec.p = r.point_at_parameter(t);
	rec.normal = vec3(0, 1, 0);

	return true;
}

bool yz_rect::hit(const ray &r, float t0, float t1, hit_record &rec) const
{
	float t = (k - r.origin().x()) / r.direction().x();
	if (t < t0 || t > t1)
		return false;

	float y = r.origin().y() + t * r.direction().y();
	float z = r.origin().z() + t * r.direction().z();

	if (y < y0 || y > y1 || z < z0 || z > z1)
		return false;

	rec.u = (y - y0) / (y1 - y0);
	rec.v = (z - z0) / (z1 - z0);

	rec.t = t;
	rec.mat_ptr = mp;
	rec.p = r.point_at_parameter(t);
	rec.normal = vec3(1, 0, 0);

	return true;
}

bool flip_normals::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{
	if (ptr->hit(r, t_min, t_max, rec))
	{
		rec.normal = -rec.normal;
		return true;
	}
	else
		return false;
}

bool flip_normals::bounding_box(float t0, float t1, aabb &box) const
{
	return ptr->bounding_box(t0, t1, box);
}

aabb flip_normals::getBound(void) const
{
	return ptr->getBound();
}
