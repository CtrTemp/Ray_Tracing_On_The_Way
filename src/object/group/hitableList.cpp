#include "hitableList.h"

aabb surronding_box(aabb box0, aabb box1)
{
	vec3 small(
		fmin(box0.min().x(), box1.min().x()),
		fmin(box0.min().y(), box1.min().y()),
		fmin(box0.min().z(), box1.min().z()));

	vec3 big(
		fmax(box0.max().x(), box1.max().x()),
		fmax(box0.max().y(), box1.max().y()),
		fmax(box0.max().z(), box1.max().z()));
	return aabb(small, big);
}

bool hitable_list::hit(const ray &r, float tmin, float tmax, hit_record &rec) const
{
	hit_record temp_rec;
	bool hit_anything = false;
	double closest_so_far = tmax;

	switch (method)
	{
	case HitMethod::NAIVE:
		for (int i = 0; i < list_size; ++i)
		{
			if (list[i]->hit(r, tmin, closest_so_far, temp_rec))
			{
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}
		break;
	case HitMethod::BVH_TREE:
		// std::cout << "tree hit" << std::endl;
		temp_rec = tree->getHitpoint(tree->root, r);
		if (temp_rec.happened)
		{
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
		}
		break;
	default:
		throw std::runtime_error("invalid iteration ergodic methods--scene");
		break;
	}
	return hit_anything;
}

bool hitable_list::bounding_box(float t0, float t1, aabb &box) const
{
	if (list_size < 1)
		return false;
	aabb temp_box;
	bool first_true = list[0]->bounding_box(t0, t1, temp_box);
	if (!first_true)
		return false;
	else
		box = temp_box;

	for (int i = 1; i < list_size; ++i)
	{
		if (list[0]->bounding_box(t0, t1, temp_box))
			box = surronding_box(box, temp_box);
		else
			return false;
	}
	return true;
}

aabb hitable_list::getBound(void) const
{

	// 如果是空列表则会构造一个无穷大的包围盒
	if (list_size < 1)
		return aabb();

	aabb bound_temp = list[0]->getBound();

	for (int i = 0; i < list_size; i++)
	{
		bound_temp = Union(bound_temp, list[i]->getBound());
	}

	return bound_temp;
}

void hitable_list::Sample(hit_record &pos, float &probability)
{
    std::cout << "目前一般情况下不会执行到这个函数，看到我说明你的程序出错了" << std::endl;
    std::cout << "目前执行的是 hitable_list 的采样函数" << std::endl;
}

float hitable_list::getArea()
{
    std::cout << "目前一般情况下不会执行到这个函数，看到我说明你的程序出错了" << std::endl;
    std::cout << "目前执行的是 hitable_list 的面积获取函数" << std::endl;
    return 0.0;
}
