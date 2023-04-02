#include <stdio.h>
#include <iostream>
#include <sys/time.h>

#include <string>

#include "render.h"
// #include "scene.h"

// using namespace cv;
#include <vector>

extern "C" __host__ void init_and_render(void);

int main(void)
{
	init_and_render();

	// std::vector<int> i;
	// i.push_back(1);
	// i.push_back(2);
	// i.push_back(3);
	// std::cout << "vec end = " << *(i.end()-1) << std::endl;
	// i.pop_back();
	// std::cout << "vec end = " << *(i.end()-1) << std::endl;
	// i.pop_back();
	// std::cout << "vec end = " << *(i.end()-1) << std::endl;

	return 0;
}
