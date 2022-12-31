#include "../GlobalInclude/camera.h"


vec3 random_in_unit_disk()
{
	vec3 p;
	do {
		p = 2.0*vec3(drand48(), drand48(), 0) - vec3(1, 1, 0);
	} while (dot(p, p) >= 1.0);
	//模拟在方格中撒点，掉入圆圈的点被收录返回
	return p;
}



ray camera::get_ray(float s, float t) 
{
		vec3 rd = lens_radius*random_in_unit_disk();//得到设定光孔大小内的任意散点（即origin点——viewpoint）
		//（该乘积的后一项是单位光孔）
		vec3 offset = rd.x()*u + rd.y()*v;//origin视点中心偏移（由xoy平面映射到u、v平面）
		//return ray(origin + offset, lower_left_conner + s*horizontal + t*vertical - origin - offset);
		float time = time0 + drand48()*(time1 - time0);
		return ray(origin + offset, lower_left_conner + s*horizontal + t*vertical - origin - offset, time);
}


