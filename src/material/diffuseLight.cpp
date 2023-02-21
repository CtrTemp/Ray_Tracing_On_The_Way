#include "diffuse_light.h"

bool diffuse_light::scatter(const ray &r_in, const hit_record &rec, Vector3f &attenuated, ray &scattered) const
{
	return false;
}

Vector3f diffuse_light::emitted(float u, float v, const Vector3f &p) const
{
	return emit->value(u, v, p);
}
