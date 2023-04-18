#ifndef OUTPUT_H
#define OUTPUT_H

#include "utils/vec3.cuh"

#include <string>
#include <vector>
#include <ostream>
#include <fstream>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

// 引入C++标准队列容器
#include <queue>

using namespace std;

enum class frame_compute
{
    IMAGE,
    DEPTH,
    LOAD
};

// 这里允许定义多个不同的缓冲区
// 缓冲区池应该只定义在主机端，使用设备端计算好的数据来填充
// 池子应该是一个FIFO类型的队列容器
class frame_pool
{
public:
    __host__ frame_pool() = default;
    __host__ frame_pool(int p_depth, int f_size)
    {
        pool_max_depth = p_depth;
        frame_size = f_size;
    }

    // 使用新得到的 frame 填充 pool
    __host__ void frame_pool_push_queue(frame_compute multi_buffer, vec3 *buffer)
    {
        // 这里不能使用指针传地址，而是应该使用值拷贝的方式，并且需要你新开辟内存空间
        vec3 *new_list_unit = new vec3[frame_size];
        memcpy(new_list_unit, buffer, frame_size);

        switch (multi_buffer)
        {
        case frame_compute::IMAGE:
            image_buffer.push(new_list_unit);
            break;
        case frame_compute::DEPTH:
            depth_buffer.push(new_list_unit);
            break;
        case frame_compute::LOAD:
            ray_compute_load_buffer.push(new_list_unit);
            break;

        default:
            throw runtime_error("Invaild frame pool type");
            break;
        }
    }

private:
    // buffer 应该作为一个 “缓冲区”，规定一个 pool depth 来说明该缓冲区最多可以同时存放多少帧
    int pool_max_depth;
    int frame_size;
    std::queue<vec3 *> image_buffer;
    std::queue<vec3 *> depth_buffer;
    std::queue<vec3 *> ray_compute_load_buffer;
};

enum class present_method
{
    LOCAL_FILE,        // 图片输出写如到本地文件
    LOCAL_SCREEN_FLOW, // 图片输出到屏幕空间进行展示
    REMOTE_FILE,       // 图片被传输到远程，写文件
    REMOTE_FLOW,       // 图片被传输到远程，展示到屏幕
    MERGE              // 预留一个可能进行多流输出的接口
};

// 暂定我们只允许发生一个图片流，该类一定是定义在主机端的
// 该类应该接收一个 frame_pool 中的某个buffer池，并对该流进行控流输出
class output_flow
{
public:
    __host__ output_flow() = default;
    __host__ output_flow(vec3 **pool, int width, int height, int p_depth, present_method p)
    {
        primary_frame_pool = pool;
        frame_width = width;
        frame_height = height;
        pool_depth = p_depth;
        present = p;
    }

    __host__ void flow_loop(std::queue<vec3 *> buffer_pool)
    {

        switch (present)
        {
        case present_method::LOCAL_FILE:
            throw std::runtime_error("present_method::LOCAL_FILE not support currently");
            /* code */
            break;

        // 就先测试这个将图片流输出到本地屏幕进行显示的接口
        case present_method::LOCAL_SCREEN_FLOW:
        {
            // 输出到屏幕空间的图片流和写文件的ofstream是否要定义成成员变量呢？
            // 留到明天考虑 2023-04-19 00：54
            cv::Mat img = cv::Mat(cv::Size(frame_width, frame_height), CV_8UC3);
            vec3 *current_buffer = buffer_pool.back();

            for (int row = 0; row < FRAME_HEIGHT; row++)
            {
                for (int col = 0; col < FRAME_WIDTH; col++)
                {
                    const int global_index = row * FRAME_WIDTH + col;
                    vec3 pixelVal = current_buffer[global_index];
                    int ir = int(255.99 * pixelVal[0]);
                    if (ir < 0)
                        ir = 0;
                    int ig = int(255.99 * pixelVal[1]);
                    if (ig < 0)
                        ig = 255;
                    int ib = int(255.99 * pixelVal[2]);
                    if (ib < 0)
                        ib = 0;

                    img.at<unsigned char>(row, col * 3 + 0) = ib;
                    img.at<unsigned char>(row, col * 3 + 1) = ig;
                    img.at<unsigned char>(row, col * 3 + 2) = ir;
                }
            }

            cv::imshow("Image Flow", img);
            buffer_pool.pop();
            break;
        }
        case present_method::REMOTE_FILE:
            throw std::runtime_error("present_method::REMOTE_FILE not support currently");
            /* code */
            break;
        case present_method::REMOTE_FLOW:
            throw std::runtime_error("present_method::REMOTE_FLOW not support currently");
            /* code */
            break;
        case present_method::MERGE:
            throw std::runtime_error("present_method::MERGE not support currently");
            /* code */
            break;

        default:
            throw std::runtime_error("Invaild present_method");
            break;
        }
    }

private:
    vec3 **primary_frame_pool;
    int frame_width;
    int frame_height;
    int pool_depth;
    int max_fps = 30;
    present_method present;
};

#endif