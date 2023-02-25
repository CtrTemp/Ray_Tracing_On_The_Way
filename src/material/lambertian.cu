#include "lambertian.cuh"

__device__ bool lambertian::scatter(const ray &r_in, const hit_record &rec, vec3 &attenuated, ray &scattered, curandStateXORWOW_t *rand_state) const
{
	vec3 target = rec.p + rec.normal + random_in_unit_sphere_device(rand_state); // 获得本次打击后得到的下一个目标点

	scattered = ray(rec.p, target - rec.p, r_in.time()); // 本次击中一个目标点后的下一个射线（获得散射光线）
	// attenuated = albedo->value(rec.u, rec.v, rec.p);
	attenuated = albedo;

	return (dot(scattered.direction(), rec.normal) > 0);
}