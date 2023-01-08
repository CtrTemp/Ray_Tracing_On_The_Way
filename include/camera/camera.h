#ifndef CAMERA_H
#define CAMERA_H

#include "utils/ray.h"

#include <math.h>


//#define M_PI	acos(-1)

//用于在圆盘光孔平面中模拟射入光孔的光线
vec3 random_in_unit_disk();
/*

{
	vec3 p;
	do {
		p = 2.0*vec3(drand48(), drand48(), 0) - vec3(1, 1, 0);
	} while (dot(p, p) >= 1.0);
	//模拟在方格中撒点，掉入圆圈的点被收录返回
	return p;
}

*/


class camera {
public:
	camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist, float t0, float t1) {

		//vec3 u, v, w;
		time0 = t0;
		time1 = t1;
		lens_radius = aperture / 2;
		float theta = vfov*M_PI / 180;
		float half_height = tan(theta / 2);
		float half_width = aspect*half_height;


		origin = lookfrom;

		w = unit_vector(lookfrom - lookat);//view_ray direction
		u = unit_vector(cross(vup, w));//camera plane horizontal direction vec 
		v = cross(w, u);//camera plane vertical direction vec

		
		lower_left_conner = origin - half_width*focus_dist*u - half_height*focus_dist*v - focus_dist*w;
		horizontal = 2 * half_width*focus_dist*u;
		vertical = 2 * half_height*focus_dist*v;
		//origin = 2 * half_height*v;
		//lower_left_conner = origin - half_width*u - half_height*v - w;
		//horizontal = 2 * half_width*u;
		//vertical = 2 * half_height*v;

	}

	ray get_ray(float s, float t); 
	

	vec3 lower_left_conner;
	vec3 horizontal;
	vec3 vertical;
	vec3 origin;
	vec3 u, v, w;
	float lens_radius;
	float time0, time1;

};

#endif // !1



/*






*/
