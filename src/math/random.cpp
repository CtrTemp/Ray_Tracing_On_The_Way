#include "random.h"
/*
float drand48(void)
{
	float ran = rand() % 101 / float(101);
	return ran;
}
*/

Vector3f random_in_unit_sphere()
{
	Vector3f p;
	do
	{
		p = 2.0 * Vector3f(drand48(), drand48(), drand48()) - Vector3f(1, 1, 1);
	} while (p.norm() >= 1.0);

	return p;
}

float get_random_float()
{
	// std::random_device dev; // the seed
	// std::mt19937 rng(dev());
	// std::uniform_real_distribution<float> dist(0.f, 0.f);

	// gen random float between (0-1)
	// return dist(rng);
	return rand()/double(RAND_MAX);
}