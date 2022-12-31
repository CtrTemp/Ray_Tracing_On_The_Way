#include "../GlobalInclude/triangle.h"

// 针对三角形面元，重写相交测试函数
bool triangle::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{

    /*
        第一步是求解三角形平面与直线是否有交点：
        三角形所在平面当前最适合使用点法式来表示，由一般形式为：
        (x0-x, y0-y, z0-z)表示平面上任意一点与另一确定点的连线，n为法线，可知二者
    点乘结果为0，于是表示为(x0-x, y0-y, z0-z)*n = 0.进一步拆解为：
        n*P_any = n*p0
        以上p0表示平面上一确定点，P_any表示平面上任意一点。
        于是我们选取三角形任意一个顶点作为p0代入，并将直线的参数方程代入P_any，
    直线参数方程为：L(t) = Ori+t*Dir;
    */
    float t = t_max;

    const float temp_num_1 = dot((vertices[0].position - r.origin()), normal);
    const float temp_num_2 = dot(r.direction(), normal);

    t = temp_num_1 / temp_num_2;

    // float constant = dot(vertices[0].position, normal);
    // t = constant - dot(normal, r.origin()) / dot(normal, r.direction());
    if (t > t_max || t < t_min)
    {
        // std::cout << "abandon 01" << std::endl;
        return false;
    }

    // std::cout << "current t = " << t << std::endl;

    // 以下检查平面法相量是否有问题
    // std::cout << "triangle normal = "
    //           << normal[0] << "; "
    //           << normal[1] << "; "
    //           << normal[2] << "; "
    //           << std::endl;

    vec3 current_point = r.point_at_parameter(t);

    // std::cout << "current_point = "
    //           << current_point.x() << " ; "
    //           << current_point.y() << " ; "
    //           << current_point.z() << " ; "
    //           << std::endl;
    // std::cout << std::endl;

    /*
        第二步是判断当前点是否在三角形内部，实际上就是看当前点是否可以用三角形的质心
    坐标来表示。
        以下我们先用简单的向量乘法来判断，即，每个三角形顶点与已知交点的连成的向量与
    三角形三条边（按照法则规定顺序）的叉乘必须符号相同，才认为点在三角形平面内。
    */

    vec3 e_temp1 = current_point - vertices[0].position;
    vec3 e_temp2 = current_point - vertices[1].position;
    vec3 e_temp3 = current_point - vertices[2].position;

    // std::cout << "edge1 = " << edges[0].x() << "; " << edges[0].y() << "; " << edges[0].z() << "; " << std::endl;
    // std::cout << "edge2 = " << edges[1].x() << "; " << edges[1].y() << "; " << edges[1].z() << "; " << std::endl;
    // std::cout << "edge3 = " << edges[2].x() << "; " << edges[2].y() << "; " << edges[2].z() << "; " << std::endl;

    // vec3 edge2 = -edge2;
    vec3 judgeVec1 = normalized_vec(cross(edges[0], e_temp1));
    vec3 judgeVec2 = normalized_vec(cross(edges[1], e_temp2));
    vec3 judgeVec3 = normalized_vec(cross(edges[2], e_temp3));

    // std::cout << "judgeVec1 = " << judgeVec1[0] << "; " << judgeVec1[1] << "; " << judgeVec1[2] << "; " << std::endl;
    // std::cout << "judgeVec2 = " << judgeVec2[0] << "; " << judgeVec2[1] << "; " << judgeVec2[2] << "; " << std::endl;
    // std::cout << "judgeVec3 = " << judgeVec3[0] << "; " << judgeVec3[1] << "; " << judgeVec3[2] << "; " << std::endl;
    // std::cout << std::endl;

    float judge1 = dot(judgeVec1, judgeVec2);
    float judge2 = dot(judgeVec2, judgeVec3);
    float judge3 = dot(judgeVec3, judgeVec1);

    // std::cout << "judge1 = " << judge1 << " ; "
    //           << "judge2 = " << judge2 << " ; "
    //           << "judge3 = " << judge3 << " ; " << std::endl;
    // if ((judgeVec1.x() <= 0 && judgeVec2.x() <= 0 && judgeVec3.x() <= 0) ||
    //     (judgeVec1.x() >= 0 && judgeVec2.x() >= 0 && judgeVec3.x() >= 0))
    if (judge1 > 0 && judge2 > 0 && judge3 > 0)
    {
        // std::cout << "sss" << std::endl;
        rec.t = t;
        rec.p = current_point;
        rec.normal = normal;
        rec.mat_ptr = mat_ptr;
        return true;
    }

    // std::cout << "any_ray_hit_triangle???" << std::endl;

    return false;
}

inline float get_max_float_val(float val1, float val2)
{
    return val1 > val2 ? val1 : val2;
}
inline float get_min_float_val(float val1, float val2)
{
    return val1 > val2 ? val2 : val1;
}

bool triangle::bounding_box(float t0, float t1, aabb &box) const
{
    // 找到“左下角点”和“右上角点”即可构造包围盒

    vec3 v0 = vertices[0].position;
    vec3 v1 = vertices[1].position;
    vec3 v2 = vertices[2].position;

    float max_x = get_max_float_val(get_max_float_val(v0[0], v1[0]), v2[0]);
    float max_y = get_max_float_val(get_max_float_val(v0[1], v1[1]), v2[1]);
    float max_z = get_max_float_val(get_max_float_val(v0[2], v1[2]), v2[2]);

    float min_x = get_min_float_val(get_min_float_val(v0[0], v1[0]), v2[0]);
    float min_y = get_min_float_val(get_min_float_val(v0[1], v1[1]), v2[1]);
    float min_z = get_min_float_val(get_min_float_val(v0[2], v1[2]), v2[2]);

    vec3 min_point(min_x, min_y, min_z);
    vec3 max_point(max_x, max_y, max_z);
    box = aabb(min_point, max_point);

    return true;
}
