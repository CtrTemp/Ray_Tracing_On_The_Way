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
	do {
		p = 2.0*vec3(drand48(), drand48(), drand48()) - vec3(1, 1, 1);
	} while (p.squared_length() >= 1.0);

	return p;
}

