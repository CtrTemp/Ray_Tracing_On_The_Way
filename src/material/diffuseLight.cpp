#include "diffuse_light.h"

bool diffuse_light::scatter(const ray &r_in, const hit_record &rec, vec3 &attenuated, ray &scattered) const
{
	return false;
}

vec3 diffuse_light::emitted(float u, float v, const vec3 &p) const
{
	return emit->value(u, v, p);
}
