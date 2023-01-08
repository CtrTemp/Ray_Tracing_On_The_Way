#include "object/hitable.h"

bool aabb::hit(const ray& r, float tmin, float tmax) const {

	for (int a = 0; a < 3; ++a)
	{
		float invD = 1.0f / r.direction()[a];
		float t0 = (min()[a] - r.origin()[a])*invD;
		float t1 = (max()[a] - r.origin()[a])*invD;
		if (invD < 0.0f)
			std::swap(t0, t1);
		tmin = t0 > tmin ? t0 : tmin;
		tmin = t0 < tmax ? t0 : tmax;
		if (tmax <= tmin)
			return false;
	}

	return true;

}
