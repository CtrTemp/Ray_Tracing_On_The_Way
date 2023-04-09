#pragma once
#ifndef BVH_TREE_H
#define BVH_TREE_H

#include "accel/bounds.cuh"
#include "object/primitive/triangle.cuh"

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
        // printf("current node stack is empty, ready to push\n");
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

        // 这里简单做一个深度测试
        int depth_counter = 0;
        bvh_node **current_node = &root_node;
        while ((*current_node)->left != nullptr && (*current_node)->right != nullptr)
        {
            current_node = &((*current_node)->left);
            depth_counter++;
        }

        printf("bvh node tree left depth = %d\n", depth_counter);

        depth_counter = 0;
        current_node = &root_node;
        while ((*current_node)->left != nullptr && (*current_node)->right != nullptr)
        {
            current_node = &((*current_node)->right);
            depth_counter++;
        }

        printf("bvh node tree right depth = %d\n", depth_counter);

        /**
         *  验证结果是，左边到11层，右边到10层，由于我们创建的 bvh_tree 是一棵完全二叉树，
         * 所以可以推断其叶子节点包含的面元应该在1024～2048之间。
         *  事实的确如此，我们的 bunny 面元为1500片
         * */

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

    // 传入一个树状结构的根节点，返回光线与树中叶子节点obj的交点
    // 实质上是 bvh_tree 的遍历，但不允许用递归算法，必须迭代
    // 注意，所构建的 bvh_tree 中的分支节点均不存在 object 实体，都是抽象化的包围盒，只有叶子节点中存放 prims 实体
    // 必须借助栈的方式进行查找遍历，才能保证不会丢失 prims 2023-04-01-noon
    // 当前函数不允许使用 new 关键字进行内存分配！
    __device__ hit_record iterativeGetHitPoint(const ray &ray) const
    {

        hit_record intersectionRecord;
        intersectionRecord.happened = false;

        // bvh_node node[33];
        // bvh_node* node[100];

        // return intersectionRecord;

        // aabb current_bound = root->bound;
        int dirIsNeg[3] = {
            int(ray.direction().x() < 0),
            int(ray.direction().y() < 0),
            int(ray.direction().z() < 0)};

        // 初始化节点栈，栈中元素为指向树中节点的指针，这个栈用作 bvh_tree 的构建
        bvh_node *simu_node_stack[50];
        uint32_t simu_node_ptr = -1; // 初始化栈为空

        // // 这个栈只用于存放存在交点的叶子节点
        bvh_node *simu_leaf_node_stack[25];
        uint32_t simu_leaf_node_ptr = -1; // 初始化栈为空

        // push_stack(simu_node_stack, &simu_node_ptr, root_node);
        simu_node_ptr += 1;
        simu_node_stack[simu_node_ptr] = root; // 根节点入栈

        // 如果根节点没有交点，则直接返回
        if (root->bound.IntersectP(ray, ray.inv_dir, dirIsNeg) == false)
        // if (root->bound.IntersectP(ray, ray.inv_dir, dirIsNeg).happened == false)
        {
            // printf("Not Hit!!\n");
            intersectionRecord.happened = false;
            return intersectionRecord;
        }

        // printf("hitted!!\n");

        // int counter = 0;

        while (simu_node_ptr != -1) // 当节点栈不为空时一直遍历
        {
            // counter++;
            // printf("simu_node_ptr = %d\n", simu_node_ptr);
            bvh_node *current_node = simu_node_stack[simu_node_ptr];
            simu_node_ptr -= 1; // 节点出栈

            // 左右子树均为空，叶子节点情况
            if (current_node->left == nullptr && current_node->right == nullptr)
            {
                // 直接访问叶子节点中的 object 对象，并将相交信息传递给 intersectionRecord
                current_node->object->hit(ray, 0.0001, 999999, intersectionRecord);
                // 如果当前射线和叶子节点中的 object 有交点就入栈
                if (intersectionRecord.happened == true)
                {
                    simu_leaf_node_ptr += 1;
                    simu_leaf_node_stack[simu_leaf_node_ptr] = current_node;
                }
            }
            // 左子树非空
            if (current_node->left != nullptr)
            {
                bool intersectionRecordLeft = current_node->left->bound.IntersectP(ray, ray.inv_dir, dirIsNeg);
                if (intersectionRecordLeft == true)
                {
                    simu_node_ptr += 1;
                    simu_node_stack[simu_node_ptr] = current_node->left; // 左子树跟节点入栈
                }
            }
            // 右子树非空
            if (current_node->left != nullptr)
            {
                bool intersectionRecordRight = current_node->right->bound.IntersectP(ray, ray.inv_dir, dirIsNeg);
                if (intersectionRecordRight == true)
                {
                    simu_node_ptr += 1;
                    simu_node_stack[simu_node_ptr] = current_node->right; // 右子树跟节点入栈
                }
            }
        }

        // printf("loop counter = %d\n", counter);

        // return intersectionRecord;

        // // 这里可以先free掉节点栈
        // free(simu_node_stack);

        // 如果整个叶子节点栈为空，说明当前射线和整个树状结构都没有交点，直接返回即可
        if (simu_leaf_node_ptr == -1)
        {
            return intersectionRecord;
        }

        // 否则遍历叶子节点栈，并找到最近交点
        hit_record nearest_rec;
        bvh_node *first_leaf_node = simu_leaf_node_stack[simu_leaf_node_ptr];
        simu_leaf_node_ptr -= 1; // 首个叶子节点出栈
        first_leaf_node->object->hit(ray, 0.0001, 999999, nearest_rec);

        while (simu_leaf_node_ptr != -1) // 叶子节点栈非空则遍历整个栈，寻找最近的射线交点
        {
            hit_record rec_temp;
            // printf("simu_leaf_node_ptr = %d\n", simu_leaf_node_ptr);
            bvh_node *leaf_node = simu_leaf_node_stack[simu_leaf_node_ptr];
            simu_leaf_node_ptr -= 1; // 叶子节点出栈
            leaf_node->object->hit(ray, 0.0001, 999999, rec_temp);
            if (rec_temp.t < nearest_rec.t) // 若当前叶子节点更近，则进行替换
            {
                nearest_rec = rec_temp;
            }
        }

        return nearest_rec;
    }

    const int maxPrimsInNode; // 常量只初始化一次, 定义当前BVH节点所能容纳最大三角形面片数
    // 当前BVH加速结构所囊括的三角形面片组
    // 这里应该作出改变以适应不同的情况，首先应该适应传入 hitableList 的情况，为世界坐标系中的不同物体构建树状结构
    triangle **triangles;
    bvh_node *root;
    size_t prims_size;
};

#endif
