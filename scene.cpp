#include "scene.h"

hitable *sample_light_RGB()
{

	material *red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
	material *white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material *green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
	material *light = new diffuse_light(new constant_texture(vec3(7, 7, 7)));

	std::vector<hitable *> hit_list;

	hit_list.push_back(new sphere(vec3(0, -1000, 0), 1000, white)); // Ground
	hit_list.push_back(new sphere(vec3(0, 2, 0), 2, white));
	hit_list.push_back(new sphere(vec3(2, 2, -4), 2, new dielectric(1.5)));

	hit_list.push_back(new sphere(vec3(-2, 2, 6), 2, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48())));

	hit_list.push_back(new sphere(vec3(0, 20, 0), 2, new diffuse_light(new constant_texture(vec3(10, 10, 10)))));

	hit_list.push_back(new xy_rect(5, 7, 1, 3, 0, new diffuse_light(new constant_texture(vec3(20, 0, 0)))));
	hit_list.push_back(new xy_rect(5, 7, 1, 3, 3, new diffuse_light(new constant_texture(vec3(0, 20, 0)))));
	hit_list.push_back(new xy_rect(5, 7, 1, 3, 6, new diffuse_light(new constant_texture(vec3(0, 0, 20)))));

	hit_list.push_back(new box(vec3(-2, 5, -2), vec3(2, 6, 2), green));

	return new hitable_list(hit_list);
}

hitable *cornell_box()
{
	std::vector<hitable *> hit_list;

	material *red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
	material *white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material *green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
	material *light = new diffuse_light(new constant_texture(vec3(7, 7, 7)));

	hit_list.push_back(new flip_normals(new yz_rect(0, 555, 0, 555, 555, green)));

	hit_list.push_back(new yz_rect(0, 555, 0, 555, 0, red));
	hit_list.push_back(new xz_rect(213, 343, 227, 332, 554, light));

	hit_list.push_back(new flip_normals(new xz_rect(0, 555, 0, 555, 555, white)));
	hit_list.push_back(new xz_rect(0, 555, 0, 555, 0, white));
	hit_list.push_back(new flip_normals(new xy_rect(0, 555, 0, 555, 555, white)));

	return new hitable_list(hit_list);
}

hitable *test_triangle()
{
	std::vector<hitable *> hit_list;

	/*
		由于当前我们直接对面进行采样，所以这里的颜色值定义目前没有意义；同样的，目前我们
	只关注面，所以顶点法相量的定义也没有任何意义，所以目前只关注顶点坐标就可以。
	*/
	vertex vert0 = {vec3(4, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0)};
	vertex vert1 = {vec3(0, 4, 0), vec3(0, 0, 0), vec3(0, 0, 0)};
	vertex vert2 = {vec3(0, 0, 4), vec3(0, 0, 0), vec3(0, 0, 0)};

	texture *pertext = new noise_texture(5);
	material *grass = new lambertian(new constant_texture(vec3(0.65, 0.75, 0.05)));
	material *red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
	material *white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material *green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
	material *light = new diffuse_light(new constant_texture(vec3(7, 7, 7)));
	material *light_r = new diffuse_light(new constant_texture(vec3(15, 0, 0)));
	material *light_g = new diffuse_light(new constant_texture(vec3(0, 15, 0)));
	material *light_b = new diffuse_light(new constant_texture(vec3(0, 0, 10)));

	// 为了方便，我们直接将其设置为光源
	// 注意，以下对顶点索引的设置目前也不存在任何意义
	hit_list.push_back(new sphere(vec3(0, 0, -1000), 1000, new lambertian(pertext))); // Ground
	hit_list.push_back(new sphere(vec3(0, 0, 0), 1, light));
	hit_list.push_back(new sphere(vec3(10, 0, 0), 1, new dielectric(1.5)));
	hit_list.push_back(new sphere(vec3(0, 10, 0), 1, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48())));
	hit_list.push_back(new sphere(vec3(0, 0, 10), 1, grass));
	hit_list.push_back(new triangle(vert0, vert1, vert2, white));

	return new hitable_list(hit_list);
}

hitable *test_triangleList()
{
	std::vector<hitable *> hit_list;

	texture *pertext = new noise_texture(1.5);
	material *yellow = new lambertian(new constant_texture(vec3(0.85, 0.55, 0.025)));
	material *grass = new lambertian(new constant_texture(vec3(0.65, 0.75, 0.05)));
	material *red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
	material *white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material *green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
	material *light = new diffuse_light(new constant_texture(vec3(7, 7, 7)));
	material *light_r = new diffuse_light(new constant_texture(vec3(15, 0, 0)));
	material *light_g = new diffuse_light(new constant_texture(vec3(0, 15, 0)));
	material *light_b = new diffuse_light(new constant_texture(vec3(0, 0, 10)));

	// 为了方便，我们直接将其设置为光源
	// 注意，以下对顶点索引的设置目前也不存在任何意义
	hit_list.push_back(new sphere(vec3(0, 0, -1000), 1000, new lambertian(pertext))); // Ground
	// hit_list.push_back(new sphere(vec3(0, 0, -1000), 1000, white); // Ground
	hit_list.push_back(new sphere(vec3(0, 0, 0), 1, red));
	hit_list.push_back(new sphere(vec3(10, 0, 0), 1, new dielectric(1.5)));
	hit_list.push_back(new sphere(vec3(0, 10, 0), 1, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48())));
	hit_list.push_back(new sphere(vec3(0, 0, 10), 1, grass));

	// 以下通过顶点列表+索引缓冲区进行创建三角形列表

	vertex testVertexList[6] = {
		{vec3(4, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0)},
		{vec3(0, 4, 0), vec3(0, 0, 0), vec3(0, 0, 0)},
		{vec3(0, 0, 4), vec3(0, 0, 0), vec3(0, 0, 0)},
		{vec3(6, 6, 1), vec3(0, 0, 0), vec3(0, 0, 0)},
		{vec3(1, 6, 6), vec3(0, 0, 0), vec3(0, 0, 0)},
		{vec3(6, 1, 6), vec3(0, 0, 0), vec3(0, 0, 0)}};

	uint32_t testIndexList[12] = {
		2, 1, 0,
		1, 3, 0,
		2, 4, 1,
		5, 2, 0};
	hit_list.push_back(new triangleList(testVertexList, testIndexList, 12, red, triangleList::HitMethod::NAIVE));

	return new hitable_list(hit_list);
}

vec3 gen_random_dir(void)
{
	srand(time(NULL));
	return normalized_vec(vec3(drand48(), drand48(), drand48()));
}

std::vector<triangle *> gen_random_triangleList(std::vector<triangle *> triangles, int size)
{
	// 预定义单一颜色
	material *grass = new lambertian(new constant_texture(vec3(0.65, 0.75, 0.05)));

	int x_range[2] = {-15, 15};
	int y_range[2] = {0, 15};
	int z_range[2] = {-15, 15};

	// 三角形边长
	float side_len = 2;

	for (size_t i = 0; i < size; i++)
	{
		float x_0_pos = drand48() * (x_range[1] - x_range[0]) + x_range[0];
		float y_0_pos = drand48() * (y_range[1] - y_range[0]) + y_range[0];
		float z_0_pos = drand48() * (z_range[1] - z_range[0]) + z_range[0];

		vec3 pos0 = {x_0_pos, y_0_pos, z_0_pos};
		vec3 pos1 = pos0 + gen_random_dir() * side_len;
		vec3 pos2 = pos0 + gen_random_dir() * side_len;

		vertex vert0 = vertex(pos0);
		vertex vert1 = vertex(pos1);
		vertex vert2 = vertex(pos2);

		triangles.push_back(new triangle(vert0, vert1, vert2, grass));
	}

	return triangles;
}

hitable *test_multi_triangleList()
{
	std::vector<hitable *> hit_list;

	texture *pertext = new noise_texture(1.5);
	material *light = new diffuse_light(new constant_texture(vec3(15, 15, 15)));
	material *yellow = new lambertian(new constant_texture(vec3(0.85, 0.55, 0.025)));
	material *grass = new lambertian(new constant_texture(vec3(0.65, 0.75, 0.05)));
	material *red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
	material *white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material *green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
	material *blue = new lambertian(new constant_texture(vec3(0.12, 0.15, 0.85)));

	int x_range[2] = {-15, 15};
	int y_range[2] = {0, 15};
	int z_range[2] = {-15, 15};
	// hit_list.push_back(new sphere(vec3(0, -1000, 0), 1000, new lambertian(pertext))); // Ground
	hit_list.push_back(new sphere(vec3(x_range[0], y_range[0], z_range[0]), 1, new lambertian(pertext)));
	hit_list.push_back(new sphere(vec3(x_range[0], y_range[0], z_range[1]), 1, blue));
	hit_list.push_back(new sphere(vec3(x_range[0], y_range[1], z_range[0]), 1, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48())));
	hit_list.push_back(new sphere(vec3(x_range[0], y_range[1], z_range[1]), 1, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48())));
	hit_list.push_back(new sphere(vec3(x_range[1], y_range[0], z_range[0]), 1, red));
	hit_list.push_back(new sphere(vec3(x_range[1], y_range[0], z_range[1]), 1, green));
	hit_list.push_back(new sphere(vec3(x_range[1], y_range[1], z_range[0]), 1, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48())));
	hit_list.push_back(new sphere(vec3(x_range[1], y_range[1], z_range[1]), 1, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48())));

	// 经过测试，10万面的三角形列表使用不到3s完成加速树结构的构建，所以bvh_tree的构建速度还是比较快的
	uint32_t triangles_num = 100;
	std::vector<triangle *> tri_list;
	tri_list = gen_random_triangleList(tri_list, triangles_num);

	// hit_list.push_back(new triangleList(tri_list, triangles_num, triangleList::HitMethod::NAIVE));
	hit_list.push_back(new triangleList(tri_list, triangles_num, triangleList::HitMethod::BVH_TREE));

	hit_list = gen_sky_box(hit_list,200);

	return new hitable_list(hit_list);
}

hitable *test_Load_Models()
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

	// hit_list.push_back(new sphere(vec3(0, 0, -1000), 1000, new lambertian(pertext)); // Ground

	// hit_list.push_back(new sphere(vec3(0, 0, 0), 50, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48()));
	// hit_list.push_back(new sphere(vec3(0, 0, 250), 50, blue);

	// hit_list.push_back(new sphere(vec3(0, 250, 0), 50, grass);
	// hit_list.push_back(new sphere(vec3(250, 0, 0), 50, red);

	// std::string module_path = "../models/cornellbox/tallbox.obj";
	// std::string module_path = "../models/viking/viking_room.obj";
	std::vector<std::string> module_path_list = {
		"../models/cornellbox/tallbox.obj",
		"../models/cornellbox/shortbox.obj",
		"../models/cornellbox/right.obj",
		"../models/cornellbox/left.obj",
		"../models/cornellbox/floor.obj",
		"../models/cornellbox/light.obj",
	};

	hit_list.push_back(new triangleList(module_path_list[0], new mental(vec3(0.9, 0.9, 0.9), 0.5 * drand48()), triangleList::HitMethod::NAIVE));
	hit_list.push_back(new triangleList(module_path_list[1], gray, triangleList::HitMethod::NAIVE));
	hit_list.push_back(new triangleList(module_path_list[2], green, triangleList::HitMethod::NAIVE));
	hit_list.push_back(new triangleList(module_path_list[3], red, triangleList::HitMethod::NAIVE));
	hit_list.push_back(new triangleList(module_path_list[4], white, triangleList::HitMethod::NAIVE));
	hit_list.push_back(new triangleList(module_path_list[5], light, triangleList::HitMethod::NAIVE));

	return new hitable_list(hit_list);
}

hitable *test_image_texture()
{

	std::vector<hitable *> hit_list;

	texture *pertext = new noise_texture(1.5);
	material *yellow = new lambertian(new constant_texture(vec3(0.85, 0.55, 0.025)));
	material *grass = new lambertian(new constant_texture(vec3(0.65, 0.75, 0.05)));
	material *red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
	material *white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material *green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
	material *light = new diffuse_light(new constant_texture(vec3(7, 7, 7)));
	material *light_r = new diffuse_light(new constant_texture(vec3(15, 0, 0)));
	material *light_g = new diffuse_light(new constant_texture(vec3(0, 15, 0)));
	material *light_b = new diffuse_light(new constant_texture(vec3(0, 0, 10)));

	// 为了方便，我们直接将其设置为光源
	// 注意，以下对顶点索引的设置目前也不存在任何意义
	hit_list.push_back(new sphere(vec3(0, 0, -1000), 1000, new lambertian(pertext))); // Ground
	// hit_list.push_back(new sphere(vec3(0, 0, -1000), 1000, white); // Ground
	hit_list.push_back(new sphere(vec3(0, 0, 0), 1, red));
	hit_list.push_back(new sphere(vec3(10, 0, 0), 1, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48())));
	// hit_list.push_back(new sphere(vec3(10, 0, 0), 1, new dielectric(1.5));
	hit_list.push_back(new sphere(vec3(0, 10, 0), 1, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48())));
	hit_list.push_back(new sphere(vec3(0, 0, 10), 1, new mental(vec3(0.65, 0.08, 0.08), 0.5 * drand48())));

	// 以下通过顶点列表+索引缓冲区进行创建三角形列表

	// vertex testVertexList[4] = {
	// 	{vec3(4, 0.1, 0.1), vec3(0, 0, 0), vec3(0, 0, 0)},
	// 	{vec3(0.1, 4, 0.1), vec3(0, 0, 0), vec3(0, 0, 0)},
	// 	{vec3(0.1, 4, 8), vec3(0, 0, 0), vec3(0, 0, 0)},
	// 	{vec3(4, 0.1, 8), vec3(0, 0, 0), vec3(0, 0, 0)}};

	// uint32_t testIndexList[6] = {
	// 	0, 1, 2,
	// 	2, 3, 0};

	// hit_list.push_back(new triangleList(testVertexList, testIndexList, 6);

	// 以下我们开始创建一个真正的有贴图的三角形
	// std::string texture_path = "../Pic/cubemaps/sky4_cube.png";
	std::string texture_path = "../Pic/textures/texture.png";
	// std::string texture_path = "../Pic/textures/potato.png";

	material *test_texture = new lambertian(new image_texture(texture_path));

	vertex testVertexList[4] = {
		{vec3(5.66, 0.1, 0.1), vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 1, 0)},
		{vec3(0.1, 5.66, 0.1), vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 1, 0)},
		{vec3(0.1, 5.66, 8), vec3(0, 0, 0), vec3(0, 0, 0), vec3(1, 0, 0)},
		{vec3(5.66, 0.1, 8), vec3(0, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0)}};

	uint32_t testIndexList[6] = {
		0, 1, 2,
		2, 3, 0};

	hit_list.push_back(new triangleList(testVertexList, testIndexList, 6, test_texture, triangleList::HitMethod::NAIVE));

	return new hitable_list(hit_list);
}

hitable *test_sky_box()
{

	std::vector<hitable *> hit_list;

	hit_list = gen_sky_box(hit_list, 200);
	// 反光球体
	hit_list.push_back(new sphere(vec3(0, -5, 0), 10, new mental(vec3(0.99, 0.99, 0.99), 0.01)));

	std::cout << "list size = " << hit_list.size() << std::endl;

	return new hitable_list(hit_list);
}

std::vector<hitable *> gen_sky_box(std::vector<hitable *> hit_list, int how_far)
{
	material *front = new diffuse_light(new image_texture("../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_0_Front+Z.png"));
	material *back = new diffuse_light(new image_texture("../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_1_Back-Z.png"));
	material *left = new diffuse_light(new image_texture("../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_2_Left+X.png"));
	material *right = new diffuse_light(new image_texture("../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_3_Right-X.png"));
	material *up = new diffuse_light(new image_texture("../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_4_Up+Y.png"));
	material *down = new diffuse_light(new image_texture("../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_5_Down-Y.png"));

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

	hit_list.push_back(new triangleList(frontVertexList, rectangleIndexList, 6, front, triangleList::HitMethod::NAIVE));
	hit_list.push_back(new triangleList(backVertexList, rectangleIndexList, 6, back, triangleList::HitMethod::NAIVE));
	hit_list.push_back(new triangleList(leftVertexList, rectangleIndexList, 6, right, triangleList::HitMethod::NAIVE));
	hit_list.push_back(new triangleList(rightVertexList, rectangleIndexList, 6, left, triangleList::HitMethod::NAIVE));
	hit_list.push_back(new triangleList(upVertexList, rectangleIndexList, 6, up, triangleList::HitMethod::NAIVE));
	hit_list.push_back(new triangleList(downVertexList, rectangleIndexList, 6, down, triangleList::HitMethod::NAIVE));

	return hit_list;
}

hitable *test_Load_complex_Models()
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

	// hit_list.push_back(new sphere(vec3(0, 0, -1000), 1000, new lambertian(pertext)); // Ground

	// hit_list.push_back(new sphere(vec3(0, 0, 0), 50, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48()));
	// hit_list.push_back(new sphere(vec3(0, 0, 250), 50, blue);

	// hit_list.push_back(new sphere(vec3(0, 250, 0), 50, grass);
	// hit_list.push_back(new sphere(vec3(250, 0, 0), 50, red);

	// std::string module_path = "../models/cornellbox/tallbox.obj";
	// std::string module_path = "../models/viking/viking_room.obj";
	std::vector<std::string> module_path_list = {
		"../models/basic_geo/cuboid.obj",
		"../models/basic_geo/dodecahedron.obj",
		"../models/bunny/bunny_low_resolution.obj"};

	// hit_list.push_back(new triangleList(module_path_list[2], grass, triangleList::HitMethod::BVH_TREE));
	hit_list.push_back(new triangleList(module_path_list[2], new mental(vec3(0.8, 0.8, 0.8), 0.99), triangleList::HitMethod::BVH_TREE));
	// hit_list.push_back(new triangleList(module_path_list[0], new mental(vec3(0.9, 0.9, 0.9), 0.5 * drand48()), triangleList::HitMethod::NAIVE));

	hit_list = gen_sky_box(hit_list, 5);

	return new hitable_list(hit_list);
}