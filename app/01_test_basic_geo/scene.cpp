#include "scene.h"


hitable_list sample_light_RGB()
{

	material *noise = new lambertian(new noise_texture(0.25));
	material *ball_noise = new lambertian(new noise_texture(2));
	material *red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
	material *white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material *green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
	material *mental_sur = new mental(vec3(0.8, 0.8, 0.8), 0.005); // mental(vec3(0.8, 0.8, 0.8), 0.02);
	material *glass_sur = new dielectric(1.5);
	material *light = new diffuse_light(new constant_texture(vec3(60, 60, 60)));
	material *light_red = new diffuse_light(new constant_texture(vec3(70, 0, 0)));
	material *light_green = new diffuse_light(new constant_texture(vec3(0, 70, 0)));
	material *light_blue = new diffuse_light(new constant_texture(vec3(0, 0, 70)));

	std::vector<hitable *> hit_list;

	hit_list.push_back(new sphere(vec3(0, -2000, 0), 2000, noise)); // Ground
	hit_list.push_back(new sphere(vec3(0, 2, 0), 2, glass_sur));
	// hit_list.push_back(new sphere(vec3(2, 2, -4), 2, new dielectric(1.5)));
	hit_list.push_back(new sphere(vec3(2, 2, -4), 2, mental_sur));
	// hit_list.push_back(new sphere(vec3(0, 2, 0), 2, new dielectric(1.5)));

	// hit_list.push_back(new sphere(vec3(-2, 2, 6), 2, noise));
	// hit_list.push_back(new sphere(vec3(-2, 2, 6), 2, new dielectric(1.5)));
	hit_list.push_back(new sphere(vec3(-2, 2, 6), 2, mental_sur));

	hit_list.push_back(new sphere(vec3(0, 15, 0), 2, light));
	hit_list.push_back(new sphere(vec3(10, 15, 10), 2, light));
	hit_list.push_back(new sphere(vec3(10, 15, -10), 2, light));
	// hit_list.push_back(new sphere(vec3(-10, 18, -10), 2, light));
	hit_list.push_back(new sphere(vec3(-10, 15, 10), 2, light));

	return hitable_list(hit_list);
}





