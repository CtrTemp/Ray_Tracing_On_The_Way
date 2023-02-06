#include "material/lambertian.h"

//  材质中的scatter函数用于决定二次射线如何生成，这是根据材料属性变化而变化的
bool lambertian::scatter(const ray &r_in, const hit_record &rec, vec3 &attenuated, ray &scattered) const
{
	// std::cout << "rec.u = " << rec.u << "; "
	// 		  << "rec.v = " << rec.v << "; "
	// 		  << std::endl;
	vec3 target = rec.p + rec.normal + random_in_unit_sphere(); // 获得本次打击后得到的下一个目标点
	scattered = ray(rec.p, target - rec.p, r_in.time());		// 本次击中一个目标点后的下一个射线（获得散射光线）
	// std::cout << "this" << std::endl;
	attenuated = albedo->value(rec.u, rec.v, rec.p);

	// return true;
	return (dot(scattered.direction(), rec.normal) > 0);
}

// Vector3f Material::eval(const Vector3f &wi, const Vector3f &wo, const Vector3f &N){
//     switch(m_type){
//         case DIFFUSE:
//         {
//             // calculate the contribution of diffuse   model
//             float cosalpha = dotProduct(N, wo);
//             if (cosalpha > 0.0f) {
//                 Vector3f diffuse = Kd / M_PI;
//                 return diffuse;
//             }
//             else
//                 return Vector3f(0.0f);
//             break;
//         }
//     }
// }

vec3 lambertian::computeBRDF(const vec3 wi, const vec3 wo, const hit_record p)
{
	float cosalpha = dot(p.normal, wo);
	if (cosalpha > 0.0f)
	{
		vec3 diffuse = this->albedo->value(p.u, p.v, p.p) / M_PI;
		return diffuse;
	}
	return vec3(0, 0, 0);
}
