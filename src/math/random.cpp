#include "math/random.h"
/*
float drand48(void)
{
	float ran = rand() % 101 / float(101);
	return ran;
}
*/

vec3 random_in_unit_sphere()
{
	vec3 p;
	do
	{
		p = 2.0 * vec3(drand48(), drand48(), drand48()) - vec3(1, 1, 1);
	} while (p.squared_length() >= 1.0);

	return p;
}

float get_random_float()
{
	std::random_device dev; // the seed
	std::mt19937 rng(dev());
	std::uniform_real_distribution<float> dist(0.f, 0.f);

	// gen random float between (0-1)
	// return dist(rng);
	return rand()/double(RAND_MAX);
}