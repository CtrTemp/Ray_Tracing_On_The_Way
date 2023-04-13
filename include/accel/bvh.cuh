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

/* ########################## BVH with SHA Related Function ########################## */
/**
 *  这里再思考一下，是否应该是先做一个快速排序更划算？？？
 *  目前方法的复杂度是 n^2 快速排序后再分割的复杂度是n*nlogn？？？ 再思考一下
 *  思考结束：不需要做快排，partition只需要保证分割点左右的列表的大小在分割点两侧即可，不需要保证分割后
 * 分割点两侧的序列内部有序。所以算法的复杂度应该是O(n)，过一遍即可。同时也可以在这一遍的过程中计算两边的
 * 包围盒。
 * */
__device__ static void generate_space_cut_strategy(aabb current_bound, int dim, int list_len, float *dim_cut_point_list)
{
    float gap = current_bound.Diagonal().e[dim] / (list_len + 1);
    float offset = current_bound.min().e[dim];
    for (int i = 1; i <= list_len; i++)
    {
        dim_cut_point_list[i - 1] = offset + gap * i;
    }
}

class bvh_tree
{
public:
    __device__ bvh_tree() = default;
    __device__ bvh_tree(triangle **prim_list, int maxPrims, size_t list_size) : prims_size(list_size), maxPrimsInNode(maxPrims)
    {
        printf("bvh construct function\n");

        // 迭代构造 BVH Tree : 传入总体的片元列表(当前网格的所有多边形面片列表)
        root = iterativeConstructTree(prim_list);

        printf("bvh tree constructed done");
    }

    // 选用全新的迭代构建 bvh_tree
    /**
     *  2023-04-13
     *  现在我们要引入一种构建更优质树的方法，即 BVH with SAH
     *  修改的点在于，之前我们的划分策略是对当前的节点内的面元列表进行排序，而后取最中间的节点作为分割点划分左右子树
     * 也即“节点数平均”法，当然还有另外的“空间平均”划分方法。但总体来说这两种方法在“左右子树存在较大空间重叠”的情况
     * 下的表现均不好，这是由于当左右子树存在大量重叠的时候，在遍历过程中，射线就不可避免的要对这两颗子树均进行相交测
     * 试，大大增加了树的遍历计算次数。可悲的是，以上两种方法完全“随机”，不考虑左右子树是否重叠的影响，也就无从避免
     * 这种左右子树包围盒存在大量重叠的情况。
     *  目前需要做的就是在每次划分的过程中，可以不进行排序，取而代之的是沿着当前节点包围盒的最长轴，给出n-1个划分策略
     * （假设当前节点内共有n个面元），在这里我们就选择在包围盒之间等距切分，否则仍要进行排序，计算开销过于巨大。而后，
     * 对于每一个划分策略使用评估函数对其进行评估，评估函数得到代价最小的划分即为最优划分。
     *  对于评估函数：对于当前划分策略，得到了两个子树面元列表，当二者重叠最小时，我们认为其达到了最优。具体的评估
     * 方法是测算两个子树面元列表的包围盒面积分别与其中面元数量之积，这个值越大，说明二者更有可能被同一条射线同时击中，
     * 且遍历代价越大。我们求解的问题是求这个最小值。
     * */
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

            // 第一步 pop 得到当前要划分的部分
            int left_index = pop_stack(simu_stack, &simu_ptr);
            int right_index = pop_stack(simu_stack, &simu_ptr);

            // pop 得到当前节点
            bvh_node *current_node = pop_stack(simu_node_stack, &simu_node_ptr);

            // 第二步 获取当前的包围盒
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
                // prims_sort_fast(&(prim_list[left_index]), &(prim_list[right_index]), dim);
                // 从这里开始使用 BVH with SAH 的方法，不进行排序，而是沿着当前轴进行等距切分
                // 根据这些切分点，划分出对应个数的划分策略
                // 最开始的一步是根据当前最长轴，按照空间等分的策略给出按最长轴的n-1个切分点（假设当前节点中共有n个面元）
                int current_prim_size = right_index - left_index + 1; // 注意这个其实比实际列表长度小1
                float *cut_point_list = new float[current_prim_size - 1];
                generate_space_cut_strategy(global_bound, dim, current_prim_size, cut_point_list);
                for (int i = 0; i < current_prim_size; i++)
                {
                    printf("%f  ", cut_point_list[i]);
                }
                printf("done please break current_prim_size = %d\n", current_prim_size);
                break;

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

    // 迭代方式，遍历寻找最近的射线与树的交点
    __device__ hit_record iterativeGetHitPoint(const ray &ray) const
    {

        hit_record intersectionRecord;
        intersectionRecord.happened = false;

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

    const int maxPrimsInNode; // 常量只初始化一次, 定义当前BVH节点所能容纳最大三角形面片数（现在没用到）
    triangle **triangles;
    bvh_node *root;
    size_t prims_size;
};

#endif

// // 这里简单做一个深度测试
// int depth_counter = 0;
// bvh_node **current_node = &root_node;
// while ((*current_node)->left != nullptr && (*current_node)->right != nullptr)
// {
//     current_node = &((*current_node)->left);
//     depth_counter++;
// }

// printf("bvh node tree left depth = %d\n", depth_counter);

// depth_counter = 0;
// current_node = &root_node;
// while ((*current_node)->left != nullptr && (*current_node)->right != nullptr)
// {
//     current_node = &((*current_node)->right);
//     depth_counter++;
// }

// printf("bvh node tree right depth = %d\n", depth_counter);
