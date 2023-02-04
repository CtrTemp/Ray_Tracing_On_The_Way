#include "accel/bvh.h"
#include <cassert>
#include <algorithm>

bvh_tree::bvh_tree(std::vector<primitive *> prim_list, int maxPrimsInNode) : maxPrimsInNode(std::min(255, maxPrimsInNode))
{
    time_t start, stop;
    time(&start);
    if (prim_list.empty())
        return;

    // 递归构造 : 传入总体的片元列表(当前网格的所有多边形面片列表)
    root = recursiveConstructTree(prim_list);

    time(&stop);

    // 计算构造这棵树的耗时
    double diff = difftime(stop, start);
    int hrs = (int)diff / 3600;
    int mins = ((int)diff / 60) - (hrs * 60);
    int secs = (int)diff - (hrs * 3600) - (mins * 60);

    // 打印构造二叉树的耗时
    printf("\rBVH Generation complete: \nTime Taken: %i hrs, %i mins, %i secs\n\n", hrs, mins, secs);

    std::cout << "max prims in nodes = " << maxPrimsInNode << std::endl;

    for (int i = 0; i < prim_list.size(); i++)
    {
        // // 这里出错了，所有的 bounds 都是 inf ！ 说明之前的递归创建有问题
        // std::cout << "bounds = "
        //           << tri_list[i]->bounds.min()[0] << "; "
        //           << tri_list[i]->bounds.min()[1] << "; "
        //           << tri_list[i]->bounds.min()[2] << "; || "
        //           << tri_list[i]->bounds.max()[0] << "; "
        //           << tri_list[i]->bounds.max()[1] << "; "
        //           << tri_list[i]->bounds.max()[2] << "; "
        //           << std::endl;
    }
}

bvh_node *bvh_tree::recursiveConstructTree(std::vector<primitive *> primitives)
{
    // bvh_root_node 创建根节点
    bvh_node *node = new bvh_node();

    // 通过归并, 将得到一个包围住当前多边形片元列表的一个大包围盒
    // aabb bounds; // 这种创建方式绝对有问题！你默认构建了一个无穷大的包围盒，于是以下做merge一直是无穷大
    aabb global_bound = primitives[0]->getBound(); // 正确做法是传入一个当前第一个三角形的bounds

    for (int i = 0; i < primitives.size(); ++i)
        global_bound = Union(global_bound, primitives[i]->getBound());

    // 最终递归返回情况01: 我们已经递归到了树的叶子节点,当前列表中只有一个元素了
    if (primitives.size() == 1)
    {
        // 那么我们创建这个叶子节点, 并将其左右子树指针置为空
        node->bound = primitives[0]->getBound();
        node->object = primitives[0];
        node->left = nullptr;
        node->right = nullptr;
        return node;
    }

    // 情况02: 叶子节点的上一层, 当前节点中有两个多边形片元, 刚好够劈开成两半
    else if (primitives.size() == 2)
    {
        // 那么, 为当前节点的左右孩子节点分别分配指针, 传入这两个元素(构造叶子节点)
        node->left = recursiveConstructTree(std::vector{primitives[0]});
        node->right = recursiveConstructTree(std::vector{primitives[1]});
        // 当前节点的包围盒为下属两个叶子节点的包围盒的并集
        node->bound = Union(node->left->bound, node->right->bound);
        return node;
    }

    // 情况03: 最为广泛出现的一种情形, 当前节点包括3个及以上的多边形面片,
    // 这里将体现我们的 bvh_node_tree 的划分准则
    else
    {
        aabb centroidBounds; // 理解为 "质心包围盒"

        // 首先 for 循环得到当前节点的整体包围盒
        for (int i = 0; i < primitives.size(); ++i)
            centroidBounds = Union(centroidBounds, primitives[i]->getBound().center());

        // 得到当前最大跨幅的坐标轴(包围盒哪个维度横跨尺度最大)
        int dim = centroidBounds.maxExtent();

        // 使用泛型算法按照最大跨幅坐标轴为其进行编组排序, 排序的依据是每个片元的质心坐标
        switch (dim)
        {
        case 0:
            // f 是 fragment 的简称 (残片/碎片)
            std::sort(primitives.begin(), primitives.end(), [](auto f1, auto f2)
                      { return f1->getBound().center().x() <
                               f2->getBound().center().x(); });
            break;
        case 1:
            std::sort(primitives.begin(), primitives.end(), [](auto f1, auto f2)
                      { return f1->getBound().center().y() <
                               f2->getBound().center().y(); });
            break;
        case 2:
            std::sort(primitives.begin(), primitives.end(), [](auto f1, auto f2)
                      { return f1->getBound().center().z() <
                               f2->getBound().center().z(); });
            break;
        }

        // 给出三个迭代器/指针, 用于左右子树的范围划分
        auto beginning = primitives.begin();
        auto middling = primitives.begin() + (primitives.size() / 2);
        auto ending = primitives.end();

        // 分配左右子树的范围
        auto leftshapes = std::vector<primitive *>(beginning, middling);
        auto rightshapes = std::vector<primitive *>(middling, ending);

        // 这里的指针失效了？！！！！！！
        // 这一步校验的意义何在?
        assert(primitives.size() == (leftshapes.size() + rightshapes.size()));

        // 分配左右子树
        node->left = recursiveConstructTree(leftshapes);
        node->right = recursiveConstructTree(rightshapes);

        // 当前节点的包围盒是左右子树包围盒的并集
        node->bound = Union(node->left->bound, node->right->bound);
    }

    // 最终返回 bvh_node_tree 的根节点
    return node;
}

// 将传入当前根节点以及一条射线, 返回打击点的所有信息
// 需要手动书写补全 射线与 bvh_node_tree 的相交函数,
// 注意,这个函数应该也是要递归遍历的

hit_record bvh_tree::getHitpoint(bvh_node *node, const ray &ray) const
{
    hit_record intersectionRecord;
    intersectionRecord.happened = false;

    // 需要对 bvh_node_tree 进行深度优先遍历
    aabb current_bound = node->bound;
    std::array<int, 3> dirIsNeg = {
        int(ray.direction().x() < 0),
        int(ray.direction().y() < 0),
        int(ray.direction().z() < 0)};

    // 如果有交点, 下一步应该进一步遍历其左右子树
    // 注意, 射线的direction_inv是在射线初始化的时候通过传入方向向量自动生成的,可以查看其构造函数
    // 所以我们不必对此进行进一步计算,直接用以下形式传入即可, 但 dirIsNeg 必须自行计算
    // 2023/01/04 截至点，目前的bug是当前传到这里，如下的判断语句总是false
    // 验证 ray_inv 是否出现问题
    // std::cout << "ray_inv = "
    //           << ray.inv_dir.x() << "; "
    //           << ray.inv_dir.y() << "; "
    //           << ray.inv_dir.z() << "; "
    //           << std::endl;
    if (current_bound.IntersectP(ray, ray.inv_dir, dirIsNeg))
    {
        // std::cout << "bound intersected" << std::endl;

        // 感觉左右子树要么全空,要么都不空,,,
        // 如果左右子树为空, 那么说明当前节点中有且只有一个Object, 我们应该针对该Object进行射线求交测试
        if (node->left == nullptr && node->right == nullptr)
        {
            // 这里是与特定多边形面元求交的基本测试, 好在程序框架帮我们集成了这个函数
            // node->object->hit(ray, std::numeric_limits<float>::min(), std::numeric_limits<float>::max(), intersectionRecord);
            
            // vital problem !!! do not ! use std::numeric_limits<float>::min()
            /*
                cause the secondary scattered ray may intersect to the surface who generate it,
            that is to say intersect to itself, which can make the rec.t parameter extremely 
            small but still larger than "std::numeric_limits<float>::max()", which makes the 
            ray iteratively bounce and intersect to itself and return an vec(0,0,0) as black 
            shading point.
            */ 
            node->object->hit(ray, 0.01, std::numeric_limits<float>::max(), intersectionRecord);
            // 第一次这里逻辑错误, 少了一个return!!!?
            // std::cout << "hit something: " << intersectionRecord.happened << std::endl;
            // std::cout << "hitrec.t = " << intersectionRecord.t << std::endl;
            // std::cout << "happened ?? = " << intersectionRecord.happened << std::endl;
            return intersectionRecord;
        }
        else
        {
            // 这里开始出现 happened 未被赋值的情况！其实最终原因是我们没有给它初始值!!
            // 就这样一个小问题，困扰你很久，但最后还是在调试过程中发现了这一问题，但最终触发失败的原因还是未知，，
            // std::cout << "hit tree: " << intersectionRecord.happened << std::endl;
            hit_record intersectionRecordLeft = getHitpoint(node->left, ray);
            hit_record intersectionRecordRight = getHitpoint(node->right, ray);
            if (intersectionRecordLeft.happened == true && intersectionRecordRight.happened == true)
            {
                // std::cout << "both hitted" << std::endl;
                // 如果两者都有交点,那么我们应该取更近的一个交点
                intersectionRecord = intersectionRecordLeft.t > intersectionRecordRight.t ? intersectionRecordRight : intersectionRecordLeft;
                // std::cout << "compared little t = " << intersectionRecord.t << std::endl;
                return intersectionRecord;
            }
            // 这里在最开始少考虑了一个条件: 当左右叶子节点二者有其一发生相交事件后,也要择机返回一个值
            // 并且此处应该将 intersectionRecord 的值更新为 左右子节点其中之一
            else if (intersectionRecordLeft.happened == true)
            {
                // std::cout << "left hitted" << std::endl;
                return intersectionRecordLeft;
            }
            else if (intersectionRecordRight.happened == true)
            {
                // std::cout << "right hitted" << std::endl;
                return intersectionRecordRight;
            }
            else
            {
                // std::cout << "bound hitted but missed triangle" << std::endl;
                return intersectionRecord;
            }
        }
    }
    // 如果压根与当前包围盒没有相交, 则将 intersectionRecord 的 happend 置为 false 即可
    else
    {
        intersectionRecord.happened = false;
        // 其实这步没什么必要, 初始化默认为false
    }

    // std::cout << "final ret hit rec.t = " << intersectionRecord.t << std::endl;
    // std::cout << std::endl;
    // std::cout << std::endl;

    return intersectionRecord;
}

bvh_tree_scene::bvh_tree_scene(std::vector<hitable *> obj_list, int maxPrimsInNode) : maxPrimsInNode(std::min(255, maxPrimsInNode))
{
    time_t start, stop;
    time(&start);
    if (obj_list.empty())
        return;

    // 递归构造 : 传入总体的片元列表(当前网格的所有多边形面片列表)
    root = recursiveConstructSceneTree(obj_list);

    time(&stop);

    // 计算构造这棵树的耗时
    double diff = difftime(stop, start);
    int hrs = (int)diff / 3600;
    int mins = ((int)diff / 60) - (hrs * 60);
    int secs = (int)diff - (hrs * 3600) - (mins * 60);

    // 打印构造二叉树的耗时
    printf("\r Scene BVH Generation complete: \nTime Taken: %i hrs, %i mins, %i secs\n\n", hrs, mins, secs);
}

hit_record bvh_tree_scene::getHitpoint(bvh_node_scene *node, const ray &ray) const
{
    hit_record intersectionRecord;
    intersectionRecord.happened = false;

    // 需要对 bvh_node_tree 进行深度优先遍历
    aabb current_bound = node->bound;
    std::array<int, 3> dirIsNeg = {
        int(ray.direction().x() < 0),
        int(ray.direction().y() < 0),
        int(ray.direction().z() < 0)};

    // 如果有交点, 下一步应该进一步遍历其左右子树
    // 注意, 射线的direction_inv是在射线初始化的时候通过传入方向向量自动生成的,可以查看其构造函数
    // 所以我们不必对此进行进一步计算,直接用以下形式传入即可, 但 dirIsNeg 必须自行计算
    // 2023/01/04 截至点，目前的bug是当前传到这里，如下的判断语句总是false
    // 验证 ray_inv 是否出现问题
    // std::cout << "ray_inv = "
    //           << ray.inv_dir.x() << "; "
    //           << ray.inv_dir.y() << "; "
    //           << ray.inv_dir.z() << "; "
    //           << std::endl;
    if (current_bound.IntersectP(ray, ray.inv_dir, dirIsNeg))
    {
        // std::cout << "bound intersected" << std::endl;

        // 感觉左右子树要么全空,要么都不空,,,
        // 如果左右子树为空, 那么说明当前节点中有且只有一个Object, 我们应该针对该Object进行射线求交测试
        if (node->left == nullptr && node->right == nullptr)
        {
            // 这里是与特定多边形面元求交的基本测试, 好在程序框架帮我们集成了这个函数
            node->object->hit(ray, 0.01, std::numeric_limits<float>::max(), intersectionRecord);
            // node->object->hit(ray, std::numeric_limits<float>::min(), std::numeric_limits<float>::max(), intersectionRecord);
            // 第一次这里逻辑错误, 少了一个return!!!?
            // std::cout << "hit something: " << intersectionRecord.happened << std::endl;
            // std::cout << "current bound = "
            //           << node->bounds.min()[0] << "; "
            //           << node->bounds.min()[1] << "; "
            //           << node->bounds.min()[2] << "; "
            //           << node->bounds.max()[0] << "; "
            //           << node->bounds.max()[1] << "; "
            //           << node->bounds.max()[2] << "; "
            //           << std::endl;
            return intersectionRecord;
        }
        else
        {
            // 这里开始出现 happened 未被赋值的情况！其实最终原因是我们没有给它初始值!!
            // 就这样一个小问题，困扰你很久，但最后还是在调试过程中发现了这一问题，但最终触发失败的原因还是未知，，
            // std::cout << "hit tree: " << intersectionRecord.happened << std::endl;
            hit_record intersectionRecordLeft = getHitpoint(node->left, ray);
            hit_record intersectionRecordRight = getHitpoint(node->right, ray);
            if (intersectionRecordLeft.happened == true && intersectionRecordRight.happened == true)
            {
                // std::cout << "both hitted" << std::endl;
                // 如果两者都有交点,那么我们应该取更近的一个交点
                intersectionRecord = intersectionRecordLeft.t > intersectionRecordRight.t ? intersectionRecordRight : intersectionRecordLeft;
                return intersectionRecord;
            }
            // 这里在最开始少考虑了一个条件: 当左右叶子节点二者有其一发生相交事件后,也要择机返回一个值
            // 并且此处应该将 intersectionRecord 的值更新为 左右子节点其中之一
            else if (intersectionRecordLeft.happened == true)
            {
                // std::cout << "left hitted" << std::endl;
                return intersectionRecordLeft;
            }
            else if (intersectionRecordRight.happened == true)
            {
                // std::cout << "right hitted" << std::endl;
                return intersectionRecordRight;
            }
            else
            {
                // std::cout << "bound hitted but missed triangle" << std::endl;
                return intersectionRecord;
            }
        }
    }
    // 如果压根与当前包围盒没有相交, 则将 intersectionRecord 的 happend 置为 false 即可
    else
    {
        intersectionRecord.happened = false;
        // 其实这步没什么必要, 初始化默认为false
    }

    return intersectionRecord;
}

bvh_node_scene *bvh_tree_scene::recursiveConstructSceneTree(std::vector<hitable *> objs)
{
    // bvh_root_node 创建根节点
    bvh_node_scene *node = new bvh_node_scene();

    // 通过归并, 将得到一个包围住当前多边形片元列表的一个大包围盒
    // aabb bounds; // 这种创建方式绝对有问题！你默认构建了一个无穷大的包围盒，于是以下做merge一直是无穷大
    aabb global_bound = objs[0]->getBound(); // 正确做法是传入一个当前第一个三角形的bounds

    // for (int i = 0; i < objs.size(); i++)
    // {
    //     std::cout << "objs[i]  bounds = "
    //               << objs[i]->bounds.min()[0] << "; "
    //               << objs[i]->bounds.min()[1] << "; "
    //               << objs[i]->bounds.min()[2] << "; || "
    //               << objs[i]->bounds.max()[0] << "; "
    //               << objs[i]->bounds.max()[1] << "; "
    //               << objs[i]->bounds.max()[2] << "; "
    //               << std::endl;
    // }

    // throw std::runtime_error("break point check objs bounds");

    for (int i = 0; i < objs.size(); ++i)
        global_bound = Union(global_bound, objs[i]->getBound());

    // std::cout << "sss" << std::endl;
    // std::cout << "global big bounds = "
    //           << bounds.min()[0] << "; "
    //           << bounds.min()[1] << "; "
    //           << bounds.min()[2] << "; || "
    //           << bounds.max()[0] << "; "
    //           << bounds.max()[1] << "; "
    //           << bounds.max()[2] << "; "
    //           << std::endl;
    // throw std::runtime_error("break point check global bounds");

    // 最终递归返回情况01: 我们已经递归到了树的叶子节点,当前列表中只有一个元素了
    if (objs.size() == 1)
    {
        // 那么我们创建这个叶子节点, 并将其左右子树指针置为空
        node->bound = objs[0]->getBound();
        node->object = objs[0];
        node->left = nullptr;
        node->right = nullptr;
        return node;
    }

    // 情况02: 叶子节点的上一层, 当前节点中有两个多边形片元, 刚好够劈开成两半
    else if (objs.size() == 2)
    {
        // 那么, 为当前节点的左右孩子节点分别分配指针, 传入这两个元素(构造叶子节点)
        node->left = recursiveConstructSceneTree(std::vector{objs[0]});
        node->right = recursiveConstructSceneTree(std::vector{objs[1]});
        // 当前节点的包围盒为下属两个叶子节点的包围盒的并集
        node->bound = Union(node->left->bound, node->right->bound);
        return node;
    }

    // 情况03: 最为广泛出现的一种情形, 当前节点包括3个及以上的多边形面片,
    // 这里将体现我们的 bvh_node_tree 的划分准则
    else
    {
        aabb centroidBounds; // 理解为 "质心包围盒"

        // 首先 for 循环得到当前节点的整体包围盒
        for (int i = 0; i < objs.size(); ++i)
            centroidBounds = Union(centroidBounds, objs[i]->getBound().center());

        // 得到当前最大跨幅的坐标轴(包围盒哪个维度横跨尺度最大)
        int dim = centroidBounds.maxExtent();

        // 使用泛型算法按照最大跨幅坐标轴为其进行编组排序, 排序的依据是每个片元的质心坐标
        switch (dim)
        {
        case 0:
            // f 是 fragment 的简称 (残片/碎片)
            std::sort(objs.begin(), objs.end(), [](auto f1, auto f2)
                      { return f1->getBound().center().x() <
                               f2->getBound().center().x(); });
            break;
        case 1:
            std::sort(objs.begin(), objs.end(), [](auto f1, auto f2)
                      { return f1->getBound().center().y() <
                               f2->getBound().center().y(); });
            break;
        case 2:
            std::sort(objs.begin(), objs.end(), [](auto f1, auto f2)
                      { return f1->getBound().center().z() <
                               f2->getBound().center().z(); });
            break;
        }

        // 给出三个迭代器/指针, 用于左右子树的范围划分
        auto beginning = objs.begin();
        auto middling = objs.begin() + (objs.size() / 2);
        auto ending = objs.end();

        // 分配左右子树的范围
        auto leftshapes = std::vector<hitable *>(beginning, middling);
        auto rightshapes = std::vector<hitable *>(middling, ending);

        // 这一步校验的意义何在?
        assert(objs.size() == (leftshapes.size() + rightshapes.size()));

        // 分配左右子树
        node->left = recursiveConstructSceneTree(leftshapes);
        node->right = recursiveConstructSceneTree(rightshapes);

        // 当前节点的包围盒是左右子树包围盒的并集
        node->bound = Union(node->left->bound, node->right->bound);
    }

    // 最终返回 bvh_node_tree 的根节点
    return node;
}
