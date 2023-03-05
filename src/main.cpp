#include <stdio.h>
#include <iostream>
#include <sys/time.h>

#include <string>

#include "render.h"
// #include "scene.h"

// using namespace cv;

unsigned int frame_width = 512;
unsigned int frame_height = 512;


extern "C" __host__ void init_and_render(void);

int main(void)
{
	init_and_render();

	return 0;
}
