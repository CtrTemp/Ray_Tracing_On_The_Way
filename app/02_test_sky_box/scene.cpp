#include "scene.h"

std::vector<std::string> skybox_textures_heavy = {
	"../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_0_Front+Z.png",
	"../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_1_Back-Z.png",
	"../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_2_Left+X.png",
	"../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_3_Right-X.png",
	"../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_4_Up+Y.png",
	"../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_5_Down-Y.png"};
std::vector<std::string> skybox_textures_fire = {
	"../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_0_Front+Z.png",
	"../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_1_Back-Z.png",
	"../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_2_Left+X.png",
	"../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_3_Right-X.png",
	"../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_4_Up+Y.png",
	"../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_5_Down-Y.png"};
std::vector<std::string> skybox_textures_high = {
	"../Pic/skybox_high/Sky_FantasyClouds2_High_Cam_0_Front+Z.png",
	"../Pic/skybox_high/Sky_FantasyClouds2_High_Cam_1_Back-Z.png",
	"../Pic/skybox_high/Sky_FantasyClouds2_High_Cam_2_Left+X.png",
	"../Pic/skybox_high/Sky_FantasyClouds2_High_Cam_3_Right-X.png",
	"../Pic/skybox_high/Sky_FantasyClouds2_High_Cam_4_Up+Y.png",
	"../Pic/skybox_high/Sky_FantasyClouds2_High_Cam_5_Down-Y.png"};



hitable_list test_sky_box()
{
	std::vector<hitable *> hit_list;
	material *mental_sur = new mental(vec3(0.8, 0.8, 0.8), 0.005); // mental(vec3(0.8, 0.8, 0.8), 0.02);
	material *glass_sur = new dielectric(1.5);

	hit_list = gen_sky_box(skybox_textures_fire, hit_list, 200);
	// 反光球体
	hit_list.push_back(new sphere(vec3(0, -5, 0), 10, mental_sur));

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
