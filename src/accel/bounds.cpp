#include "bounds.h"

bool aabb::IntersectP(const ray &ray, const Vector3f &invDir, const std::array<int, 3> dirIsNeg) const
{
    float x_min = this->_min.x(), y_min = this->_min.y(), z_min = this->_min.z();
    float x_max = this->_max.x(), y_max = this->_max.y(), z_max = this->_max.z();

    // float x_min_distance = , x_max_distance = ;

    float t_x_min, t_y_min, t_z_min;
    float t_x_max, t_y_max, t_z_max;

    // 注意,以上为第一次尝试写的,逻辑已经出现了问题! 虽然要保证的条件没错,
    // 但应如果大小颠倒,该做的操作是交换 max 和 min 的值, 而非直接取反!这造成了数值错误
    // 故修改成以下
    t_x_min = dirIsNeg[0] == 0 ? (x_min - ray.origin().x()) * invDir.x() : (x_max - ray.origin().x()) * invDir.x();
    t_x_max = dirIsNeg[0] == 0 ? (x_max - ray.origin().x()) * invDir.x() : (x_min - ray.origin().x()) * invDir.x();

    t_y_min = dirIsNeg[1] == 0 ? (y_min - ray.origin().y()) * invDir.y() : (y_max - ray.origin().y()) * invDir.y();
    t_y_max = dirIsNeg[1] == 0 ? (y_max - ray.origin().y()) * invDir.y() : (y_min - ray.origin().y()) * invDir.y();

    t_z_min = dirIsNeg[2] == 0 ? (z_min - ray.origin().z()) * invDir.z() : (z_max - ray.origin().z()) * invDir.z();
    t_z_max = dirIsNeg[2] == 0 ? (z_max - ray.origin().z()) * invDir.z() : (z_min - ray.origin().z()) * invDir.z();

    const float t_global_min = std::min(t_x_max, std::min(t_y_max, t_z_max));
    const float t_global_max = std::max(t_x_min, std::max(t_y_min, t_z_min));

    if (t_global_max <= t_global_min && t_global_min >= 0)
        return true;
    else
        return false;
}