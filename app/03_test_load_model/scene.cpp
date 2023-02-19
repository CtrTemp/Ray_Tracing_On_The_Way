#include "scene.h"

std::vector<std::string> skybox_textures_heavy = {
	"../../../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_0_Front+Z.png",
	"../../../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_1_Back-Z.png",
	"../../../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_2_Left+X.png",
	"../../../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_3_Right-X.png",
	"../../../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_4_Up+Y.png",
	"../../../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_5_Down-Y.png"};
std::vector<std::string> skybox_textures_fire = {
	"../../../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_0_Front+Z.png",
	"../../../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_1_Back-Z.png",
	"../../../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_2_Left+X.png",
	"../../../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_3_Right-X.png",
	"../../../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_4_Up+Y.png",
	"../../../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_5_Down-Y.png"};
std::vector<std::string> skybox_textures_high = {
	"../../../Pic/skybox_high/Sky_FantasyClouds2_High_Cam_0_Front+Z.png",
	"../../../Pic/skybox_high/Sky_FantasyClouds2_High_Cam_1_Back-Z.png",
	"../../../Pic/skybox_high/Sky_FantasyClouds2_High_Cam_2_Left+X.png",
	"../../../Pic/skybox_high/Sky_FantasyClouds2_High_Cam_3_Right-X.png",
	"../../../Pic/skybox_high/Sky_FantasyClouds2_High_Cam_4_Up+Y.png",
	"../../../Pic/skybox_high/Sky_FantasyClouds2_High_Cam_5_Down-Y.png"};



hitable_list test_Load_Models()
{

	std::vector<hitable *> hit_list;

	texture *pertext = new noise_texture(1.5);
	material *light = new diffuse_light(new constant_texture(vec3(15, 15, 15)));
	material *yellow = new lambertian(new constant_texture(vec3(0.85, 0.55, 0.025)));
	material *grass = new lambertian(new constant_texture(vec3(0.65, 0.75, 0.05)));
	material *red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
	material *white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material *gray = new lambertian(new constant_texture(vec3(0.43, 0.43, 0.43)));
	material *green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
	material *blue = new lambertian(new constant_texture(vec3(0.12, 0.15, 0.85)));

	std::vector<std::string> module_path_list = {
		"../../../models/cornellbox/tallbox.obj",
		"../../../models/cornellbox/shortbox.obj",
		"../../../models/cornellbox/right.obj",
		"../../../models/cornellbox/left.obj",
		"../../../models/cornellbox/floor.obj",
		"../../../models/cornellbox/light.obj",
	};

	// hit_list.push_back(new models(module_path_list[0], new mental(vec3(0.9, 0.9, 0.9), 0.005), models::HitMethod::NAIVE, models::PrimType::TRIANGLE));
	hit_list.push_back(new models(module_path_list[0], gray, models::HitMethod::NAIVE, models::PrimType::TRIANGLE));
	hit_list.push_back(new models(module_path_list[1], gray, models::HitMethod::NAIVE, models::PrimType::TRIANGLE));
	hit_list.push_back(new models(module_path_list[2], green, models::HitMethod::NAIVE, models::PrimType::TRIANGLE));
	hit_list.push_back(new models(module_path_list[3], red, models::HitMethod::NAIVE, models::PrimType::TRIANGLE));
	hit_list.push_back(new models(module_path_list[4], white, models::HitMethod::NAIVE, models::PrimType::TRIANGLE));
	hit_list.push_back(new models(module_path_list[5], light, models::HitMethod::NAIVE, models::PrimType::TRIANGLE));

	return hitable_list(hit_list);
}

std::vector<hitable *> gen_sky_box(std::vector<std::string> textures_path, std::vector<hitable *> hit_list, int how_far)
{
	const texture *temp_tex = new image_texture(textures_path[0]);
	material *front = new diffuse_light(new image_texture(textures_path[0]));
	material *back = new diffuse_light(new image_texture(textures_path[1]));
	material *left = new diffuse_light(new image_texture(textures_path[2]));
	material *right = new diffuse_light(new image_texture(textures_path[3]));
	material *up = new diffuse_light(new image_texture(textures_path[4]));
	material *down = new diffuse_light(new image_texture(textures_path[5]));

	const float side_len = how_far;

	vec3 cubeVertexList[8] = {
		vec3(side_len / 2, side_len / 2, side_len / 2),
		vec3(side_len / 2, side_len / 2, -side_len / 2),
		vec3(-side_len / 2, side_len / 2, -side_len / 2),
		vec3(-side_len / 2, side_len / 2, side_len / 2),

		vec3(side_len / 2, -side_len / 2, side_len / 2),
		vec3(side_len / 2, -side_len / 2, -side_len / 2),
		vec3(-side_len / 2, -side_len / 2, -side_len / 2),
		vec3(-side_len / 2, -side_len / 2, side_len / 2)};

	// nice
	vertex frontVertexList[4] = {
		{cubeVertexList[0], vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0)},
		{cubeVertexList[4], vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0)},
		{cubeVertexList[7], vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 1, 0)},
		{cubeVertexList[3], vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 0, 0)}};

	// nice
	vertex backVertexList[4] = {
		{cubeVertexList[2], vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0)},
		{cubeVertexList[6], vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0)},
		{cubeVertexList[5], vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 1, 0)},
		{cubeVertexList[1], vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 0, 0)}};

	vertex leftVertexList[4] = {
		{cubeVertexList[1], vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0)},
		{cubeVertexList[5], vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0)},
		{cubeVertexList[4], vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 1, 0)},
		{cubeVertexList[0], vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 0, 0)}};

	vertex rightVertexList[4] = {
		{cubeVertexList[3], vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0)},
		{cubeVertexList[7], vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0)},
		{cubeVertexList[6], vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 1, 0)},
		{cubeVertexList[2], vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 0, 0)}};

	vertex upVertexList[4] = {
		{cubeVertexList[0], vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0)},
		{cubeVertexList[3], vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 1, 0)},
		{cubeVertexList[2], vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 0, 0)},
		{cubeVertexList[1], vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0)}};

	// nice
	vertex downVertexList[4] = {
		{cubeVertexList[4], vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0)},
		{cubeVertexList[5], vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0)},
		{cubeVertexList[6], vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 1, 0)},
		{cubeVertexList[7], vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 0, 0)}};

	uint32_t rectangleIndexList[6] = {
		0, 1, 2,
		2, 3, 0};

	hit_list.push_back(new models(frontVertexList, rectangleIndexList, 6, front, models::HitMethod::NAIVE, models::PrimType::TRIANGLE));
	hit_list.push_back(new models(backVertexList, rectangleIndexList, 6, back, models::HitMethod::NAIVE, models::PrimType::TRIANGLE));
	hit_list.push_back(new models(leftVertexList, rectangleIndexList, 6, right, models::HitMethod::NAIVE, models::PrimType::TRIANGLE));
	hit_list.push_back(new models(rightVertexList, rectangleIndexList, 6, left, models::HitMethod::NAIVE, models::PrimType::TRIANGLE));
	hit_list.push_back(new models(upVertexList, rectangleIndexList, 6, up, models::HitMethod::NAIVE, models::PrimType::TRIANGLE));
	hit_list.push_back(new models(downVertexList, rectangleIndexList, 6, down, models::HitMethod::NAIVE, models::PrimType::TRIANGLE));

	return hit_list;
}

hitable_list test_Load_complex_Models()
{

	std::vector<hitable *> hit_list;

	material *mental_sur = new mental(vec3(0.8, 0.8, 0.8), 0.005); // mental(vec3(0.8, 0.8, 0.8), 0.02);
	material *glass_sur = new dielectric(1.5);

	texture *pertext = new noise_texture(1.5);
	material *light = new diffuse_light(new constant_texture(vec3(15, 15, 15)));
	material *yellow = new lambertian(new constant_texture(vec3(0.85, 0.55, 0.025)));
	material *grass = new lambertian(new constant_texture(vec3(0.65, 0.75, 0.05)));
	material *red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
	material *white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material *gray = new lambertian(new constant_texture(vec3(0.43, 0.43, 0.43)));
	material *green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
	material *blue = new lambertian(new constant_texture(vec3(0.12, 0.15, 0.85)));

	// hit_list.push_back(new sphere(vec3(0, 0, -1000), 1000, new lambertian(pertext)); // Ground

	// hit_list.push_back(new sphere(vec3(0, 0, 0), 50, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48()));
	// hit_list.push_back(new sphere(vec3(0, 0, 250), 50, blue);

	// hit_list.push_back(new sphere(vec3(0, 250, 0), 50, grass);
	// hit_list.push_back(new sphere(vec3(250, 0, 0), 50, red);

	// std::string module_path = "../../../models/cornellbox/tallbox.obj";
	// std::string module_path = "../../../models/viking/viking_room.obj";
	std::vector<std::string> module_path_list = {
		"../../../models/basic_geo/cuboid.obj",
		"../../../models/basic_geo/dodecahedron.obj",
		"../../../models/bunny/bunny_low_resolution.obj"};

	// 这里渲染1000面的兔子模型
	// hit_list.push_back(new models(module_path_list[2], new mental(vec3(0.8, 0.8, 0.8), 0.01), models::HitMethod::NAIVE, models::PrimType::TRIANGLE));
	// hit_list.push_back(new models(module_path_list[2], new mental(vec3(0.8, 0.8, 0.8), 0.01), models::HitMethod::BVH_TREE, models::PrimType::TRIANGLE));

	// hit_list.push_back(new models(module_path_list[2], grass, models::HitMethod::NAIVE, models::PrimType::TRIANGLE));
	hit_list.push_back(new models(module_path_list[2], mental_sur, models::HitMethod::BVH_TREE, models::PrimType::TRIANGLE));

	// hit_list.push_back(new models(module_path_list[2], new dielectric(1.5), models::HitMethod::NAIVE, models::PrimType::TRIANGLE));
	// hit_list.push_back(new models(module_path_list[2], new dielectric(1.5), models::HitMethod::BVH_TREE, models::PrimType::TRIANGLE));

	hit_list = gen_sky_box(skybox_textures_high, hit_list, 10);

	return hitable_list(hit_list);
}
