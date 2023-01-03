#include "scene.h"

hitable *sample_light_RGB()
{

	material *red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
	material *white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material *green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
	material *light = new diffuse_light(new constant_texture(vec3(7, 7, 7)));

	hitable **list = new hitable *[9];

	int i = 0;
	list[i++] = new sphere(vec3(0, -1000, 0), 1000, white); // Ground
	list[i++] = new sphere(vec3(0, 2, 0), 2, white);
	list[i++] = new sphere(vec3(2, 2, -4), 2, new dielectric(1.5));

	list[i++] = new sphere(vec3(-2, 2, 6), 2, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48()));

	list[i++] = new sphere(vec3(0, 20, 0), 2, new diffuse_light(new constant_texture(vec3(10, 10, 10))));

	list[i++] = new xy_rect(5, 7, 1, 3, 0, new diffuse_light(new constant_texture(vec3(20, 0, 0))));
	list[i++] = new xy_rect(5, 7, 1, 3, 3, new diffuse_light(new constant_texture(vec3(0, 20, 0))));
	list[i++] = new xy_rect(5, 7, 1, 3, 6, new diffuse_light(new constant_texture(vec3(0, 0, 20))));

	list[i++] = new box(vec3(-2, 5, -2), vec3(2, 6, 2), green);

	return new hitable_list(list, 9);
}

hitable *cornell_box()
{
	hitable **list = new hitable *[6];
	int i = 0;

	material *red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
	material *white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material *green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
	material *light = new diffuse_light(new constant_texture(vec3(7, 7, 7)));

	list[i++] = new flip_normals(new yz_rect(0, 555, 0, 555, 555, green));

	list[i++] = new yz_rect(0, 555, 0, 555, 0, red);
	list[i++] = new xz_rect(213, 343, 227, 332, 554, light);

	list[i++] = new flip_normals(new xz_rect(0, 555, 0, 555, 555, white));
	list[i++] = new xz_rect(0, 555, 0, 555, 0, white);
	list[i++] = new flip_normals(new xy_rect(0, 555, 0, 555, 555, white));

	return new hitable_list(list, i);
}

hitable *test_triangle()
{
	hitable **list = new hitable *[6];
	int i = 0;

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
	list[i++] = new sphere(vec3(0, 0, -1000), 1000, new lambertian(pertext)); // Ground
	list[i++] = new sphere(vec3(0, 0, 0), 1, light);
	list[i++] = new sphere(vec3(10, 0, 0), 1, new dielectric(1.5));
	list[i++] = new sphere(vec3(0, 10, 0), 1, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48()));
	list[i++] = new sphere(vec3(0, 0, 10), 1, grass);
	list[i++] = new triangle(vert0, vert1, vert2, white);

	return new hitable_list(list, i);
}

hitable *test_triangleList()
{
	hitable **hit_list = new hitable *[6];
	int i = 0;

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
	hit_list[i++] = new sphere(vec3(0, 0, -1000), 1000, new lambertian(pertext)); // Ground
	// hit_list[i++] = new sphere(vec3(0, 0, -1000), 1000, white); // Ground
	hit_list[i++] = new sphere(vec3(0, 0, 0), 1, red);
	hit_list[i++] = new sphere(vec3(10, 0, 0), 1, new dielectric(1.5));
	hit_list[i++] = new sphere(vec3(0, 10, 0), 1, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48()));
	hit_list[i++] = new sphere(vec3(0, 0, 10), 1, grass);

	const vertex vertices[12] = {
		{vec3(4, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0)},
		{vec3(0, 4, 0), vec3(0, 0, 0), vec3(0, 0, 0)},
		{vec3(0, 0, 4), vec3(0, 0, 0), vec3(0, 0, 0)},

		{vec3(0, 4, 0), vec3(0, 0, 0), vec3(0, 0, 0)},
		{vec3(4, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0)},
		{vec3(6, 6, 1), vec3(0, 0, 0), vec3(0, 0, 0)},

		{vec3(0, 0, 4), vec3(0, 0, 0), vec3(0, 0, 0)},
		{vec3(0, 4, 0), vec3(0, 0, 0), vec3(0, 0, 0)},
		{vec3(1, 6, 6), vec3(0, 0, 0), vec3(0, 0, 0)},

		{vec3(4, 0, 0), vec3(0, 0, 0), vec3(0, 0, 0)},
		{vec3(0, 0, 4), vec3(0, 0, 0), vec3(0, 0, 0)},
		{vec3(6, 1, 6), vec3(0, 0, 0), vec3(0, 0, 0)}};
	// 以下通过三角形列表依次传入创建三角形列表

	triangle **tri_list = new triangle *[4];
	int tri_index = 0;

	tri_list[tri_index++] = new triangle(vertices[0], vertices[1], vertices[2], yellow);
	tri_list[tri_index++] = new triangle(vertices[3], vertices[4], vertices[5], new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48()));
	tri_list[tri_index++] = new triangle(vertices[6], vertices[7], vertices[8], new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48()));
	tri_list[tri_index++] = new triangle(vertices[9], vertices[10], vertices[11], new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48()));

	// hit_list[i++] = new triangleList(tri_list, tri_index);

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
	hit_list[i++] = new triangleList(testVertexList, testIndexList, 12, red);

	return new hitable_list(hit_list, i);
}

vec3 gen_random_dir(void)
{
	srand(time(NULL));
	return normalized_vec(vec3(drand48(), drand48(), drand48()));
}

void gen_random_triangleList(triangle **triangles, int size)
{
	// 预定义单一颜色
	material *grass = new lambertian(new constant_texture(vec3(0.65, 0.75, 0.05)));

	int x_range[2] = {-15, 15};
	int y_range[2] = {-15, 15};
	int z_range[2] = {0, 10};

	float side_len = 2;

	for (size_t i = 0; i < size; i++)
	{
		float x_0_pos = drand48() * (x_range[1] - x_range[0]) + x_range[0];
		float y_0_pos = drand48() * (y_range[1] - y_range[0]) + y_range[0];
		float z_0_pos = drand48() * (z_range[1] - z_range[0]) + z_range[0];

		vec3 pos0 = {x_0_pos, y_0_pos, z_0_pos};
		vec3 pos1 = pos0 + gen_random_dir() * side_len;
		vec3 pos2 = pos0 + gen_random_dir() * side_len;

		vertex vert0 = {pos0, vec3{0, 0, 0}, vec3{0, 0, 0}};
		vertex vert1 = {pos1, vec3{0, 0, 0}, vec3{0, 0, 0}};
		vertex vert2 = {pos2, vec3{0, 0, 0}, vec3{0, 0, 0}};

		triangles[i] = new triangle(vert0, vert1, vert2, grass);
	}
}

hitable *test_multi_triangleList()
{

	hitable **hit_list = new hitable *[11 + 1];
	int i = 0;

	texture *pertext = new noise_texture(1.5);
	material *light = new diffuse_light(new constant_texture(vec3(15, 15, 15)));
	material *yellow = new lambertian(new constant_texture(vec3(0.85, 0.55, 0.025)));
	material *grass = new lambertian(new constant_texture(vec3(0.65, 0.75, 0.05)));
	material *red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
	material *white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material *green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
	material *blue = new lambertian(new constant_texture(vec3(0.12, 0.15, 0.85)));

	int x_range[2] = {-15, 15};
	int y_range[2] = {-15, 15};
	int z_range[2] = {0, 10};
	hit_list[i++] = new sphere(vec3(0, 0, -1000), 1000, new lambertian(pertext)); // Ground

	hit_list[i++] = new sphere(vec3(0, 0, z_range[0]), 1, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48()));
	hit_list[i++] = new sphere(vec3(0, 0, z_range[1]), 1, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48()));

	hit_list[i++] = new sphere(vec3(0, y_range[1], z_range[0]), 1, grass);
	hit_list[i++] = new sphere(vec3(0, y_range[1], z_range[1]), 1, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48()));
	hit_list[i++] = new sphere(vec3(0, y_range[0], z_range[0]), 1, blue);
	hit_list[i++] = new sphere(vec3(0, y_range[0], z_range[1]), 1, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48()));
	hit_list[i++] = new sphere(vec3(x_range[0], 0, z_range[0]), 1, blue);
	hit_list[i++] = new sphere(vec3(x_range[0], 0, z_range[1]), 1, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48()));
	hit_list[i++] = new sphere(vec3(x_range[1], 0, z_range[0]), 1, red);
	hit_list[i++] = new sphere(vec3(x_range[1], 0, z_range[1]), 1, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48()));

	uint32_t triangles_num = 100;
	triangle **tri_list = new triangle *[triangles_num];
	gen_random_triangleList(tri_list, triangles_num);

	hit_list[i++] = new triangleList(tri_list, triangles_num);

	return new hitable_list(hit_list, i);
}

hitable *test_Load_Models()
{

	hitable **hit_list = new hitable *[6];
	int i = 0;

	texture *pertext = new noise_texture(1.5);
	material *light = new diffuse_light(new constant_texture(vec3(15, 15, 15)));
	material *yellow = new lambertian(new constant_texture(vec3(0.85, 0.55, 0.025)));
	material *grass = new lambertian(new constant_texture(vec3(0.65, 0.75, 0.05)));
	material *red = new lambertian(new constant_texture(vec3(0.65, 0.05, 0.05)));
	material *white = new lambertian(new constant_texture(vec3(0.73, 0.73, 0.73)));
	material *gray = new lambertian(new constant_texture(vec3(0.43, 0.43, 0.43)));
	material *green = new lambertian(new constant_texture(vec3(0.12, 0.45, 0.15)));
	material *blue = new lambertian(new constant_texture(vec3(0.12, 0.15, 0.85)));

	// hit_list[i++] = new sphere(vec3(0, 0, -1000), 1000, new lambertian(pertext)); // Ground

	// hit_list[i++] = new sphere(vec3(0, 0, 0), 50, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48()));
	// hit_list[i++] = new sphere(vec3(0, 0, 250), 50, blue);

	// hit_list[i++] = new sphere(vec3(0, 250, 0), 50, grass);
	// hit_list[i++] = new sphere(vec3(250, 0, 0), 50, red);

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

	hit_list[i++] = new triangleList(module_path_list[0], new mental(vec3(0.9, 0.9, 0.9), 0.5 * drand48()));
	hit_list[i++] = new triangleList(module_path_list[1], gray);
	hit_list[i++] = new triangleList(module_path_list[2], green);
	hit_list[i++] = new triangleList(module_path_list[3], red);
	hit_list[i++] = new triangleList(module_path_list[4], white);
	hit_list[i++] = new triangleList(module_path_list[5], light);

	return new hitable_list(hit_list, i);
}

hitable *test_image_texture()
{

	hitable **hit_list = new hitable *[6];
	int i = 0;

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
	hit_list[i++] = new sphere(vec3(0, 0, -1000), 1000, new lambertian(pertext)); // Ground
	// hit_list[i++] = new sphere(vec3(0, 0, -1000), 1000, white); // Ground
	hit_list[i++] = new sphere(vec3(0, 0, 0), 1, red);
	hit_list[i++] = new sphere(vec3(10, 0, 0), 1, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48()));
	// hit_list[i++] = new sphere(vec3(10, 0, 0), 1, new dielectric(1.5));
	hit_list[i++] = new sphere(vec3(0, 10, 0), 1, new mental(vec3(0.8, 0.8, 0.8), 0.5 * drand48()));
	hit_list[i++] = new sphere(vec3(0, 0, 10), 1, new mental(vec3(0.65, 0.08, 0.08), 0.5 * drand48()));

	// 以下通过顶点列表+索引缓冲区进行创建三角形列表

	// vertex testVertexList[4] = {
	// 	{vec3(4, 0.1, 0.1), vec3(0, 0, 0), vec3(0, 0, 0)},
	// 	{vec3(0.1, 4, 0.1), vec3(0, 0, 0), vec3(0, 0, 0)},
	// 	{vec3(0.1, 4, 8), vec3(0, 0, 0), vec3(0, 0, 0)},
	// 	{vec3(4, 0.1, 8), vec3(0, 0, 0), vec3(0, 0, 0)}};

	// uint32_t testIndexList[6] = {
	// 	0, 1, 2,
	// 	2, 3, 0};

	// hit_list[i++] = new triangleList(testVertexList, testIndexList, 6);

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

	hit_list[i++] = new triangleList(testVertexList, testIndexList, 6, test_texture);

	return new hitable_list(hit_list, i);
}

hitable *test_sky_box()
{

	hitable **hit_list = new hitable *[7];
	int i = 0;

	// material *front = new diffuse_light(new image_texture("../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_0_Front.png"));
	// material *back = new diffuse_light(new image_texture("../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_1_Back.png"));
	// material *left = new diffuse_light(new image_texture("../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_2_Left.png"));
	// material *right = new diffuse_light(new image_texture("../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_3_Right.png"));
	// material *up = new diffuse_light(new image_texture("../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_4_Up.png"));
	// material *down = new diffuse_light(new image_texture("../Pic/skybox_sunset/Sky_FantasySky_Fire_Cam_5_Down.png"));

	material *front = new diffuse_light(new image_texture("../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_0_Front+Z.png"));
	material *back = new diffuse_light(new image_texture("../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_1_Back-Z.png"));
	material *left = new diffuse_light(new image_texture("../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_2_Left+X.png"));
	material *right = new diffuse_light(new image_texture("../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_3_Right-X.png"));
	material *up = new diffuse_light(new image_texture("../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_4_Up+Y.png"));
	material *down = new diffuse_light(new image_texture("../Pic/skybox_heavy/Sky_FantasySky_Heavy_1_Cam_5_Down-Y.png"));

	const float side_len = 200;

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

	hit_list[i++] = new triangleList(frontVertexList, rectangleIndexList, 6, front);
	hit_list[i++] = new triangleList(backVertexList, rectangleIndexList, 6, back);
	hit_list[i++] = new triangleList(leftVertexList, rectangleIndexList, 6, right);
	hit_list[i++] = new triangleList(rightVertexList, rectangleIndexList, 6, left);
	hit_list[i++] = new triangleList(upVertexList, rectangleIndexList, 6, up);
	hit_list[i++] = new triangleList(downVertexList, rectangleIndexList, 6, down);


	// 反光球体
	hit_list[i++] = new sphere(vec3(0, -5, 0), 10, new mental(vec3(0.99, 0.99, 0.99), 0.01));


	return new hitable_list(hit_list, i);
}
