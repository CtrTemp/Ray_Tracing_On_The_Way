#include "hitableList.cuh"

__device__ hitable_list::hitable_list(curandStateXORWOW_t *rand_state)
{
	list_size = 1;
	list = new hitable *[list_size];
	int index = 0;
	material *red_mat_ptr = new lambertian(vec3(0.8, 0.1, 0.1));
	list[index++] = new sphere(vec3(0, -1000, 0), 1000, red_mat_ptr);
}

__device__ bool hitable_list::hit(const ray &r, float tmin, float tmax, hit_record &rec) const
{
	hit_record temp_rec;
	bool hit_anything = false;
	double closest_so_far = tmax;

	for (int i = 0; i < list_size; ++i)
	{
		if (list[i]->hit(r, tmin, closest_so_far, temp_rec))
		{
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
	}
	return hit_anything;
}


__global__ void gen_world(curandStateXORWOW_t *rand_state, hitable_list **world)
{
	// 在设备端创建
	*world = new hitable_list(rand_state);
}
