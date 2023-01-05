#include "../GlobalInclude/triangle.h"

// #include <tuple>

// // 质心计算
// void getBarycentricCoord(vec3 p, vec3 v1, vec3 v2, vec3 v3, float *alpha, float *beta, float *gamma)
// {
//     // *beta = ((v1[0] - p[0]) * (v1[1] - v3[1]) - (v1[0] - v3[0]) * (v1[1] - p[1])) / ((v1[0] - v3[0]) * (v2[1] - v1[1]) - (v2[0] - v1[0]) * (v1[1] - v3[1]));
//     // *gamma = ((v2[1] - v1[1]) * (p[0] - v1[0]) - (v2[0] - v1[0]) * (p[1] - v1[1])) / ((v1[1] - v3[1]) * (v2[0] - v1[0]) - (v1[0] - v3[0]) * (v2[1] - v1[1]));
//     // *alpha = 1 - *beta - *gamma;

// }

// 质心计算 这次我们使用莱布尼茨公式进行求解
void getBarycentricCoord(vec3 P, vec3 A, vec3 B, vec3 C, float *alpha, float *beta, float *gamma)
{
    vec3 v0 = B - A;
    vec3 v1 = C - A;
    vec3 v2 = P - A;

    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float d00 = dot(v0, v0);
    float d11 = dot(v1, v1);
    float d01 = dot(v0, v1);

    float d = (d00 * d11 - d01 * d01);
    *beta = (d20 * d11 - d21 * d01) / d;
    *gamma = (d21 * d00 - d20 * d01) / d;
    *alpha = 1 - *beta - *gamma;

    // 打印输出验证
    // std::cout << "alpha = " << *alpha << "; "
    //           << "beta = " << *beta << "; "
    //           << "gamma = " << *gamma << "; "
    //           << std::endl;
}

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
        rec.happened = false;
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
        /*
            这里要补充记录 uv 值，从三角形的三个顶点出发，获取三个顶点的uv值，最终插值得到当前点的uv值
        */
        float alpha, beta, gamma;
        getBarycentricCoord(current_point, vertices[0].position, vertices[1].position, vertices[2].position, &alpha, &beta, &gamma);
        // std::cout << "alpha = " << alpha << "; "
        //           << "beta = " << beta << "; "
        //           << "gamma = " << gamma << "; " << std::endl;
        // std::cout << std::endl;

        // 2023-01-01结果
        // 这里暂时得出的是错误结果，但是我们先不管它，先写后面的逻辑
        // 以下应该记录计算出的该点的uv值，使用以上计算出的的质心坐标系下的值进行变换
        float u_temp = vertices[0].tex_coord[0] * alpha + vertices[1].tex_coord[0] * beta + vertices[2].tex_coord[0] * gamma;
        float v_temp = vertices[0].tex_coord[1] * alpha + vertices[1].tex_coord[1] * beta + vertices[2].tex_coord[1] * gamma;
        // float w = vertices[0].tex_coord[2] * alpha + vertices[1].tex_coord[2] * beta + vertices[2].tex_coord[2] * gamma;

        rec.u = u_temp;
        rec.v = v_temp;

        // std::cout << "u_temp = " << u_temp << "; "
        //           << "v_temp = " << v_temp << "; "
        //           //   << "w_temp = " << w_temp << "; "
        //           << std::endl;
        // std::cout << std::endl;

        rec.happened = true;
        return true;
    }

    // std::cout << "any_ray_hit_triangle???" << std::endl;

    rec.happened = false;
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
