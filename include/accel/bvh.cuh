#pragma once
#ifndef BVH_TREE_H
#define BVH_TREE_H

#include "accel/bounds.cuh"
#include "object/primitive/triangle.cuh"
#include <vector>

/* ########################### 迭代构建 bvh_tree 所需的辅助函数 ###########################*/
__device__ static int *stack_init(uint32_t len)
{
    int *ret_stack = new int[len];
    for (int i = 0; i < len; i++)
    {
        ret_stack[i] = -1;
    }
    return ret_stack;
}

__device__ static bool *mark_list_init(uint32_t len)
{
    bool *ret_mark_list = new bool[len];
    for (int i = 0; i < len; i++)
    {
        ret_mark_list[i] = false;
    }
    return ret_mark_list;
}

// 模拟出栈
__device__ static int pop_stack(int *stack, uint32_t *simu_ptr)
{
    if (*simu_ptr == -1)
    {
        // printf("current stack is empty, cannot pop\n");
        return -1;
    }

    int ret_val = stack[*simu_ptr];
    stack[*simu_ptr] = -1;
    *simu_ptr -= 1;

    return ret_val;
}

// 模拟入栈
__device__ static void push_stack(int *stack, uint32_t *simu_ptr, int val)
{
    if (*simu_ptr == -1)
    {
        // printf("current stack is empty, ready to push\n");
    }
    *simu_ptr += 1;
    stack[*simu_ptr] = val;
}

/* ########################### 迭代快速排序 所需的辅助函数 ###########################*/

// 第一轮循环就出错了，回来看这个问题 2023-03-30
__device__ static int prims_sort_fast_single_ride(triangle **prims_begin, triangle **prims_end, int mode)
{

    // printf("access bounds test in single_ride = %f\n", (*(prims_end - 1))->getBound().center().e[0]);

    if (prims_begin >= prims_end)
    {
        return 0;
    }
    triangle **left_cursor = prims_begin;
    triangle **right_cursor = prims_end;
    triangle key = **prims_begin;

    // printf("begin single ride loop\n");
    while (left_cursor != right_cursor)
    {
        // right_cursor--;
        // float comp_key = key.getBound().center().e[mode];
        // float comp_left = (*left_cursor)->getBound().center().e[mode];
        // float comp_right = (*right_cursor)->getBound().center().e[mode];
        // (*right_cursor)->getBound().center().e[mode];
        // printf("access test passed %f\n", (*right_cursor)->getBound().center().e[mode]);
        while (left_cursor != right_cursor && (*right_cursor)->getBound().center().e[mode] >= key.getBound().center().e[mode])
        {
            right_cursor--;
        }
        // printf("single ride mark1\n");
        if (left_cursor < right_cursor)
        {
            triangle temp_loop = **left_cursor;
            **left_cursor = **right_cursor;
            **right_cursor = temp_loop;
        }
        // printf("single ride mark2\n");
        while (left_cursor != right_cursor && (*left_cursor)->getBound().center().e[mode] <= key.getBound().center().e[mode])
        {
            left_cursor++;
        }
        // printf("single ride mark3\n");
        if (left_cursor < right_cursor)
        {
            triangle temp_loop = **left_cursor;
            **left_cursor = **right_cursor;
            **right_cursor = temp_loop;
        }
        // printf("single ride mark4\n");
    }
    if ((*prims_begin)->getBound().center().e[mode] > (*left_cursor)->getBound().center().e[mode])
    {
        **prims_begin = **left_cursor;
        **left_cursor = key;
    }

    return left_cursor - prims_begin;
}

__device__ static void prims_sort_fast(triangle **prims_begin, triangle **prims_end, int mode)
{
    // printf("access bounds test in prims_sort_fast = %f\n", (*prims_begin)->getBound().center().e[0]);

    size_t size = prims_end - prims_begin + 1;
    int *simu_stack = new int[size];
    uint32_t simu_ptr = -1;
    int global_left_index = 0;
    int global_right_index = size - 1;
    // printf("func prims_sort_fast begin push stack\n");
    push_stack(simu_stack, &simu_ptr, global_left_index);
    push_stack(simu_stack, &simu_ptr, global_right_index);
    // printf("func prims_sort_fast begin loop\n");
    while (simu_ptr != -1)
    {

        // printf("loop test mark 1\n");
        int right_index = pop_stack(simu_stack, &simu_ptr);
        int left_index = pop_stack(simu_stack, &simu_ptr);
        triangle **left_cursor = &(prims_begin[left_index]);
        triangle **right_cursor = &(prims_begin[right_index]);

        // printf("loop test mark 2\n");

        int middle_index = prims_sort_fast_single_ride(left_cursor, right_cursor, mode);
        middle_index += left_index;
        // printf("loop test mark 3\n");

        if (middle_index - 1 > left_index)
        {
            push_stack(simu_stack, &simu_ptr, left_index);
            push_stack(simu_stack, &simu_ptr, middle_index - 1);
        }
        if (middle_index + 1 < right_index)
        {
            push_stack(simu_stack, &simu_ptr, middle_index + 1);
            push_stack(simu_stack, &simu_ptr, right_index);
        }
        // printf("loop test mark 4\n");
    }
}

// 最简单的冒泡排序，之后再作优化 （事实证明过于低效，弃用）
__device__ static void prims_sort_bubble(triangle **prims_begin, triangle **prims_end, int mode)
{
    for (triangle **i = prims_begin; i != prims_end; i++)
    {
        printf("haha\n");
        for (triangle **j = i; j != prims_end; j++)
        {
            // mode==0 x轴优先排序；mode==1 y轴优先排序；mode==2 z轴优先排序
            // 默认升序
            float comp_1 = (*i)->getBound().center().e[mode];
            float comp_2 = (*j)->getBound().center().e[mode];
            if (comp_1 < comp_2)
            {
                printf("exchange triangle list\n");
                // 这里暂时默认是三角形面元
                triangle prim_temp_val = **i;
                **i = **j;
                **j = prim_temp_val;
            }
        }
    }
}

class bvh_node
{
public:
    __device__ bvh_node()
    {
        bound = aabb(); // 包围盒初始化为无限大
        left = nullptr; // 左右子节点均初始化指向空
        right = nullptr;
        object = nullptr;
    }

    // int splitAxis = 0, firstPrimOffset = 0, ntriangles = 0;
    aabb bound;
    bvh_node *left;
    bvh_node *right;
    // node 节点的主体可以是多种多样的
    triangle *object;
    size_t index;
};

// 模拟 node 出栈
__device__ static bvh_node *pop_stack(bvh_node **stack, uint32_t *simu_node_ptr)
{
    if (*simu_node_ptr == -1)
    {
        printf("current node stack is empty, cannot pop\n");
        return nullptr;
    }

    bvh_node *ret_val = stack[*simu_node_ptr];
    stack[*simu_node_ptr] = nullptr;
    *simu_node_ptr -= 1;

    return ret_val;
}

// 模拟 node 入栈
__device__ static void push_stack(bvh_node **stack, uint32_t *simu_node_ptr, bvh_node *node)
{
    if (*simu_node_ptr == -1)
    {
        printf("current node stack is empty, ready to push\n");
    }
    *simu_node_ptr += 1;
    stack[*simu_node_ptr] = node;
}

__device__ static bvh_node **node_stack_init(uint32_t len)
{
    bvh_node **ret_node_stack = new bvh_node *[len];
    for (int i = 0; i < len; i++)
    {
        ret_node_stack[i] = nullptr;
    }
    return ret_node_stack;
}

class bvh_tree
{
public:
    __device__ bvh_tree() = default;
    __device__ bvh_tree(triangle **prim_list, int maxPrims, size_t list_size) : prims_size(list_size), maxPrimsInNode(maxPrims)
    {
        // time_t start, stop;
        // time(&start);
        // if (prim_list.empty())
        //     return;
        printf("bvh construct function\n");

        // 递归构造 : 传入总体的片元列表(当前网格的所有多边形面片列表)
        // root = recursiveConstructTree(prim_list, prims_size);
        root = iterativeConstructTree(prim_list);

        printf("bvh tree constructed done");

        // time(&stop);

        // 计算构造这棵树的耗时
        // double diff = difftime(stop, start);
        // int hrs = (int)diff / 3600;
        // int mins = ((int)diff / 60) - (hrs * 60);
        // int secs = (int)diff - (hrs * 3600) - (mins * 60);

        // 打印构造二叉树的耗时
        // printf("\rBVH Generation complete: \nTime Taken: %i hrs, %i mins, %i secs\n\n", hrs, mins, secs);

        // std::cout << "max prims in nodes = " << maxPrimsInNode << std::endl;
    }

    // 选用全新的迭代构建 bvh_tree
    __device__ bvh_node *iterativeConstructTree(triangle **prim_list)
    {

        // 初始化数字索引栈，栈中的元素值将是要排序列表的索引，这个栈用作快速排序
        int *simu_stack = stack_init(prims_size);
        // 初始化节点栈，栈中元素为指向树中节点的指针，这个栈用作 bvh_tree 的构建
        bvh_node **simu_node_stack = node_stack_init(prims_size);
        uint32_t simu_ptr = -1;      // 初始化栈为空
        uint32_t simu_node_ptr = -1; // 初始化栈为空

        // 初始化根节点并入栈
        bvh_node *root_node = new bvh_node;
        push_stack(simu_node_stack, &simu_node_ptr, root_node);
        // 列表左右端点索引入栈，两个端点之间的值将被排序，push 的时候记得先右后左
        push_stack(simu_stack, &simu_ptr, prims_size - 1);
        push_stack(simu_stack, &simu_ptr, 0);

        // 以上迭代的启动条件设置完毕
        // 开启构建循环，当栈不为空时一直执行
        printf("ready to jump in main construct bvh_tree loop\n");
        while (simu_ptr != -1)
        {

            // 第一步 pop 得到当前要排序的部分
            int left_index = pop_stack(simu_stack, &simu_ptr);
            int right_index = pop_stack(simu_stack, &simu_ptr);

            // pop 得到当前节点
            bvh_node *current_node = pop_stack(simu_node_stack, &simu_node_ptr);

            // 第二步 获取当前的包围盒，并
            aabb global_bound = prim_list[left_index]->getBound();
            for (int i = left_index; i <= right_index; ++i)
                global_bound = Union(global_bound, prim_list[i]->getBound()); // 得到总体包围盒

            // 为当前节点的 bound 进行赋值
            current_node->bound = global_bound;
            // 仅当叶子节点时才会赋值其 object
            if (left_index == right_index)
            {
                current_node->object = prim_list[left_index];
            }

            else
            {
                // 得到当前包围盒的最大跨度轴，并根据最大轴跨度进行排序
                int dim = global_bound.maxExtent();
                prims_sort_fast(&(prim_list[left_index]), &(prim_list[right_index]), dim);

                // 第三步 给出当前部分的 middle_index
                int middle_index = left_index + (right_index - left_index) / 2;

                int middleLeft_index_next_round = middle_index;
                int middleRight_index_next_round = middle_index + 1;

                // 第四步 右子树处理
                push_stack(simu_stack, &simu_ptr, right_index);
                push_stack(simu_stack, &simu_ptr, middleRight_index_next_round);
                bvh_node *right_node = new bvh_node;
                current_node->right = right_node;
                push_stack(simu_node_stack, &simu_node_ptr, right_node);

                // 第五步 左子树处理
                push_stack(simu_stack, &simu_ptr, middleLeft_index_next_round);
                push_stack(simu_stack, &simu_ptr, left_index);
                bvh_node *left_node = new bvh_node;
                current_node->left = left_node;
                push_stack(simu_node_stack, &simu_node_ptr, left_node);
            }
        }

        return root_node;
    }
    // CUDA 中不允许函数递归
    __device__ bvh_node *recursiveConstructTree(triangle **prim_list, size_t current_size)
    {
        // bvh_root_node 创建根节点
        bvh_node *node = new bvh_node();

        // 通过归并, 将得到一个包围住当前多边形片元列表的一个大包围盒
        // aabb bounds; // 这种创建方式绝对有问题！你默认构建了一个无穷大的包围盒，于是以下做merge一直是无穷大
        aabb global_bound = prim_list[0]->getBound(); // 正确做法是传入一个当前第一个三角形的bounds

        for (int i = 0; i < current_size; ++i)
            global_bound = Union(global_bound, prim_list[i]->getBound());

        printf("recursive Construct Tree \n");

        // 最终递归返回情况01: 我们已经递归到了树的叶子节点,当前列表中只有一个元素了
        if (current_size == 1)
        {
            printf("current size = 1\n");
            // 那么我们创建这个叶子节点, 并将其左右子树指针置为空
            node->bound = prim_list[0]->getBound();
            node->object = prim_list[0];
            node->left = nullptr;
            node->right = nullptr;
            return node;
        }

        // 情况02: 叶子节点的上一层, 当前节点中有两个多边形片元, 刚好够劈开成两半
        else if (current_size == 2)
        {
            printf("current size = 2\n");
            for (int i = 0; i < current_size; i++)
            {
                printf("watch its list val [%f,%f,%f]\n",
                       prim_list[i]->vertices->position.e[0],
                       prim_list[i]->vertices->position.e[1],
                       prim_list[i]->vertices->position.e[2]);
            }
            // // 那么, 为当前节点的左右孩子节点分别分配指针, 传入这两个元素(构造叶子节点)
            node->left = recursiveConstructTree(prim_list, 1);
            // node->right = recursiveConstructTree(&prim_list[1], 1);
            // // 当前节点的包围盒为下属两个叶子节点的包围盒的并集
            // node->bound = Union(node->left->bound, node->right->bound);
            // return node;
        }

        // // 情况03: 最为广泛出现的一种情形, 当前节点包括3个及以上的多边形面片,
        // // 这里将体现我们的 bvh_node_tree 的划分准则
        // else
        // {
        //     aabb centroidBounds; // 理解为 "质心包围盒"

        //     // 首先 for 循环得到当前节点的整体包围盒
        //     for (int i = 0; i < current_size; ++i)
        //         centroidBounds = Union(centroidBounds, prim_list[i]->getBound().center());

        //     // 得到当前最大跨幅的坐标轴(包围盒哪个维度横跨尺度最大)
        //     int dim = centroidBounds.maxExtent();

        //     // 自定义排序算法
        //     prims_sort(&prim_list[0], &prim_list[current_size], dim);

        //     // 给出三个迭代器/指针, 用于左右子树的范围划分
        //     triangle **beginning = &prim_list[0];
        //     triangle **middling = prim_list + current_size / 2;
        //     triangle **ending = &prim_list[current_size];

        //     // // 分配左右子树的范围
        //     // auto leftshapes = std::vector<triangle *>(beginning, middling);
        //     // auto rightshapes = std::vector<triangle *>(middling, ending);

        //     // 这里的指针失效了？！！！！！！
        //     // 这一步校验的意义何在?
        //     // assert(current_size == (leftshapes.size() + rightshapes.size()));

        //     // 分配左右子树 （思考在3个子节点时候的情况）
        //     node->left = recursiveConstructTree(beginning, current_size / 2 + current_size % 2);
        //     node->right = recursiveConstructTree(middling, current_size / 2);

        //     // 当前节点的包围盒是左右子树包围盒的并集
        //     node->bound = Union(node->left->bound, node->right->bound);
        // }

        // 最终返回 bvh_node_tree 的根节点
        return node;
    }

    // hit_record Intersect(const ray &ray) const; // 目前这个函数并没有被定义，，，

    // 传入一个树状结构的根节点，返回光线与树中叶子节点obj的交点
    // 实质上是 bvh_tree 的遍历，但不允许用递归算法，必须迭代

    __device__ hit_record iterativeGetHitPoint(bvh_node *node, const ray &ray) const
    {
        hit_record intersectionRecord;
        intersectionRecord.happened = false;

        aabb current_bound = node->bound;
        int dirIsNeg[3] = {
            int(ray.direction().x() < 0),
            int(ray.direction().y() < 0),
            int(ray.direction().z() < 0)};

        bvh_node **current_node = &node;
        // 如果根节点没有交点，则直接返回
        if ((*current_node)->bound.IntersectP(ray, ray.inv_dir, dirIsNeg).happened == false)
        {
            return intersectionRecord;
        }
        // 开始进行遍历
        while (true)
        {
            // 左右子树均存在的情况
            if ((*current_node)->left != nullptr && (*current_node)->right != nullptr)
            {
                hit_record intersectionRecordLeft = (*current_node)->left->bound.IntersectP(ray, ray.inv_dir, dirIsNeg);
                hit_record intersectionRecordRight = (*current_node)->right->bound.IntersectP(ray, ray.inv_dir, dirIsNeg);

                // 两个子树的包围盒均有交点
                if (intersectionRecordLeft.happened == true && intersectionRecordRight.happened == true)
                {
                    // 如果两者都有交点,那么我们应该取更近的一个交点
                    if (intersectionRecordLeft.t > intersectionRecordRight.t)
                    {
                        intersectionRecord = intersectionRecordRight;
                        current_node = &((*current_node)->right);
                    }
                    else
                    {
                        intersectionRecord = intersectionRecordLeft;
                        current_node = &((*current_node)->left);
                    }
                }
                // 仅左子树包围盒存在交点
                else if (intersectionRecordLeft.happened == true)
                {
                    intersectionRecord = intersectionRecordLeft;
                    current_node = &((*current_node)->left);
                }
                // 仅右子树包围盒存在交点
                else if (intersectionRecordRight.happened == true)
                {
                    intersectionRecord = intersectionRecordRight;
                    current_node = &((*current_node)->right);
                }
                // 均无交点，可以直接返回了
                else
                {
                    return intersectionRecord;
                }
            }
            // 仅左子树存在
            else if ((*current_node)->left != nullptr)
            {
                hit_record intersectionRecordLeft = (*current_node)->left->bound.IntersectP(ray, ray.inv_dir, dirIsNeg);
                if (intersectionRecordLeft.happened == true)
                {
                    intersectionRecord = intersectionRecordLeft;
                    current_node = &((*current_node)->left);
                }
                else
                {
                    return intersectionRecord;
                }
            }
            // 仅右子树存在
            else if ((*current_node)->right != nullptr)
            {
                hit_record intersectionRecordRight = (*current_node)->right->bound.IntersectP(ray, ray.inv_dir, dirIsNeg);
                if (intersectionRecordRight.happened == true)
                {
                    intersectionRecord = intersectionRecordRight;
                    current_node = &((*current_node)->right);
                }
                else
                {
                    return intersectionRecord;
                }
            }
            // 左右子树均空，那么此时访问到的是一个叶子节点
            else if ((*current_node)->left == nullptr && (*current_node)->right == nullptr)
            {
                // 直接访问节点中的object对象
                (*current_node)->object->hit(ray, 0.0001, 999999, intersectionRecord);
                // 断出
                break;
            }
        }
        return intersectionRecord;
    }

    __device__ hit_record getHitpoint(bvh_node *node, const ray &ray) const
    {
        hit_record intersectionRecord;
        intersectionRecord.happened = false;

        // 需要对 bvh_node_tree 进行深度优先遍历
        aabb current_bound = node->bound;
        int dirIsNeg[3] = {
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
        if (current_bound.IntersectP(ray, ray.inv_dir, dirIsNeg).happened == true)
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
                node->object->hit(ray, 0.01, 999999, intersectionRecord);
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

    const int maxPrimsInNode; // 常量只初始化一次, 定义当前BVH节点所能容纳最大三角形面片数
    // 当前BVH加速结构所囊括的三角形面片组
    // 这里应该作出改变以适应不同的情况，首先应该适应传入 hitableList 的情况，为世界坐标系中的不同物体构建树状结构
    triangle **triangles;
    bvh_node *root;
    size_t prims_size;
};

// class bvh_node_scene
// {
// public:
//     bvh_node_scene()
//     {
//         bound = aabb(); // 包围盒初始化为无限大
//         left = nullptr;  // 左右子节点均初始化指向空
//         right = nullptr;
//         object = nullptr;
//     }

//     // int splitAxis = 0, firstPrimOffset = 0, ntriangles = 0;
//     aabb bound;
//     bvh_node_scene *left;
//     bvh_node_scene *right;
//     // node 节点的主体可以是多种多样的
//     hitable *object;
// };

// class bvh_tree_scene
// {
// public:
//     bvh_tree_scene() = default;
//     bvh_tree_scene(std::vector<hitable *> tri_list, int maxPrimsInNode = 1);
//     bvh_node_scene *recursiveConstructSceneTree(std::vector<hitable *> triangles);
//     hit_record Intersect(const ray &ray) const;
//     hit_record getHitpoint(bvh_node_scene *node, const ray &ray) const;

//     const int maxPrimsInNode; // 常量只初始化一次, 定义当前BVH节点所能容纳最大三角形面片数
//     // 当前BVH加速结构所囊括的三角形面片组
//     // 这里应该作出改变以适应不同的情况，首先应该适应传入 hitableList 的情况，为世界坐标系中的不同物体构建树状结构
//     std::vector<hitable *> obj_list;
//     bvh_node_scene *root;
// };

#endif

// 选用全新的迭代构建 bvh_tree
// __device__ bvh_node *iterativeConstructTree(triangle **prim_list)
// {

//     int *simu_stack = stack_init(prims_size);
//     uint32_t simu_ptr = -1;
//     bool *mark_list = mark_list_init(prims_size);
//     bvh_node *node_list = new bvh_node[prims_size]; // 这个最后是不是可以free掉？？ 不行，坚决不行

//     aabb global_bound = prim_list[0]->getBound(); // 正确做法是传入一个当前第一个三角形的bounds
//     for (int i = 0; i < prims_size; ++i)
//         global_bound = Union(global_bound, prim_list[i]->getBound()); // 得到总体包围盒

//     int dim = global_bound.maxExtent();
//     printf("start to prims sort, dim = %d\n", dim);
//     printf("access bounds test = %f\n", (*prim_list)->getBound().center().e[0]);
//     prims_sort_fast(&(prim_list[0]), &(prim_list[prims_size - 1]), dim); // 首次排序出错了！！！
//     printf("prims sort end\n");
//     int root_index = prims_size / 2;
//     mark_list[root_index] = true;
//     push_stack(simu_stack, &simu_ptr, root_index);
//     // bound在元素压栈时赋值，之后的主构建循环中也是如此
//     node_list[root_index].bound = global_bound;

//     // 以上迭代的启动条件设置完毕
//     // 开启构建循环，当栈不为空时一直执行
//     printf("ready to jump in main construct bvh_tree loop\n");
//     while (simu_ptr != -1)
//     {
//         int middle_index = pop_stack(simu_stack, &simu_ptr);
//         printf("current stack size = %d\n", simu_ptr);
//         // 这里有一个问题，分支节点不应该有object值
//         node_list[middle_index].object = prim_list[middle_index];
//         node_list[middle_index].index = middle_index;
//         // node_list[middle_index].bound = ???
//         mark_list[middle_index] = true; // 标记为已找到正确位置

//         if (middle_index == 0 || middle_index == prims_size - 1) // 这一定是一个叶子节点了
//         {
//             node_list[middle_index].left == nullptr;
//             node_list[middle_index].right == nullptr;
//             continue;
//         }

//         int forward_sort_index = middle_index - 1;
//         int backward_sort_index = middle_index + 1;

//         /* ############################## 前向序遍历 ##############################*/
//         while (true)
//         {
//             // 遇到第一个 mark 为 true 的则断出，当遍历到 index = 0 的位置也会断出
//             if (mark_list[forward_sort_index] == true || forward_sort_index <= 0)
//             {
//                 break;
//             }
//             forward_sort_index--;
//         }

//         if ((middle_index - forward_sort_index) == 1 && (forward_sort_index != 0)) // 这个节点没有左子树，不必进行 sort 操作
//         {
//             node_list[middle_index].left == nullptr;
//         }
//         else
//         {
//             // 计算左子树当前bound
//             aabb left_bound = prim_list[forward_sort_index]->getBound(); // 左树列表第一个面元的 bound
//             for (int i = forward_sort_index; i < middle_index; ++i)
//                 left_bound = Union(left_bound, prim_list[i]->getBound()); // 得到左子树总体包围盒
//             int left_dim = left_bound.maxExtent();
//             // sort 被排序列表中的元素是左闭右闭的
//             prims_sort_fast(&(prim_list[forward_sort_index]), &(prim_list[middle_index - 1]), left_dim);
//             int front_sort_size = middle_index - forward_sort_index;
//             int front_middle = forward_sort_index + front_sort_size / 2;
//             push_stack(simu_stack, &simu_ptr, front_middle);
//             // printf("left push %d\n", front_middle + 1);
//             node_list[front_middle].bound = left_bound;              // 为左子树根节点添加 bound
//             node_list[middle_index].left = &node_list[front_middle]; // 将左子树根节点链接到当前节点
//         }

//         /* ############################## 后向序遍历 ##############################*/
//         while (true)
//         {
//             // 遇到第一个 mark 为 true 的则断出，当遍历到 list 末尾也会断出
//             if (mark_list[backward_sort_index] == true || backward_sort_index >= prims_size - 1)
//             {
//                 break;
//             }
//             backward_sort_index++;
//         }

//         if (backward_sort_index - middle_index == 1) // 这个节点没有右子树，不必进行 sort 操作
//         {
//             node_list[middle_index].right == nullptr;
//         }
//         else
//         {
//             // 计算右子树当前bound
//             aabb right_bound = prim_list[middle_index + 1]->getBound(); // 右树列表第一个面元的 bound
//             for (int i = middle_index + 1; i < backward_sort_index; ++i)
//                 right_bound = Union(right_bound, prim_list[i]->getBound()); // 得到右子树总体包围盒
//             int right_dim = right_bound.maxExtent();
//             prims_sort_fast(&(prim_list[middle_index + 1]), &(prim_list[backward_sort_index - 1]), right_dim);
//             int back_sort_size = backward_sort_index - middle_index;
//             int back_middle = middle_index + back_sort_size / 2;
//             push_stack(simu_stack, &simu_ptr, back_middle);
//             printf("right push %d\n", back_middle + 1);
//             node_list[back_middle].bound = right_bound;              // 为右子树根节点添加 bound
//             node_list[middle_index].right = &node_list[back_middle]; // 将右子树根节点链接到当前节点
//         }
//     }

//     return &(node_list[root_index]);
// }