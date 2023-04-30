#include <stdio.h>
#include <iostream>
#include <sys/time.h>

#include <string>

#include <thread>

#include "render.h"
#include "server/onMessageRouter.h"

#include <vector>

extern "C" __host__ void init_and_render(void);

// broadcast_server b_server;

// void server_startup()
// {
// 	b_server.run(CURRENT_PORT);
// }

void access_global_variable()
{
	sleep(3);
	while (1)
	{
		// 无论GPU相隔多久计算完一帧，CPU稳定最快30帧向前端发数据
		// sleep(1);// Linux 下 sleep 是以 s 为单位的
		usleep(10000); // 这个应该是以 us 为单位的

		// if (global_queue.size() >= 30)
		// {
		// 	// 注意pop操作不会释放内存，需要手动释放
		// 	global_queue.pop();
		// }
		// printf("\ncurrent variable = %d\n", global_variable);
		// printf("current set length = %ld, queue push head = %d\n", global_queue.size(), global_queue.front());

		// 缓冲区最大深度为10
		if (frame_buffer_pool.size() < 10)
		{
			continue;
		}
		printf("current frame buffer pool depth = %ld\n", frame_buffer_pool.size());
		// 显示当前帧
		// cv::namedWindow("Image Flow");
		// showFrameFlow(FRAME_WIDTH, FRAME_HEIGHT, frame_buffer_pool.front());
		vec3 *header_frame = frame_buffer_pool.front();
		frame_buffer_pool.pop();
		// 在此处手动释放内存，因为pop操作并不会自动帮忙释放
		delete[] header_frame;
	}
}

void send_img_pack_to_client()
{

	cv::Mat img = cv::Mat(cv::Size(FRAME_WIDTH, FRAME_HEIGHT), CV_8UC3);
	clock_t start, end;

	while (true)
	{
		// usleep(10000);
		start = clock();

		// 缓冲区最大深度为10
		if (frame_buffer_pool.size() < 10)
		{
			continue;
		}
		printf("current frame buffer pool depth = %ld\n", frame_buffer_pool.size());
		// 显示当前帧
		// cv::namedWindow("Image Flow");
		// showFrameFlow(FRAME_WIDTH, FRAME_HEIGHT, frame_buffer_pool.front());
		vec3 *frame_buffer = frame_buffer_pool.front();

		for (int row = 0; row < FRAME_HEIGHT; row++)
		{
			for (int col = 0; col < FRAME_WIDTH; col++)
			{
				const int global_index = row * FRAME_WIDTH + col;
				vec3 pixelVal = frame_buffer[global_index];
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

		std::vector<uchar> buf;
		cv::imencode(".jpg", img, buf);
		// uchar *enc_msg = reinterpret_cast<unsigned char*>(buf.data());
		std::string img_data = base64_encode(buf.data(), buf.size(), false);

		// 以下使用 JSON 数据格式进行信息传递
		Json::FastWriter jsonWrite;
		Json::Value json_obj;

		// 写入一般数据
		json_obj["url"] = img_data;

		std::string json_str = jsonWrite.write(json_obj);

		// 加入这两句后便不会默认打印发送的消息
		b_server.m_server.clear_access_channels(websocketpp::log::alevel::all);
		b_server.m_server.clear_access_channels(websocketpp::log::alevel::frame_payload);

		// 向所有的 client 端广播信息
		websocketpp::frame::opcode::value sCode = websocketpp::frame::opcode::TEXT;
		for (auto it : b_server.m_connections)
		{
			b_server.m_server.send(it, json_str, sCode);
		}

		frame_buffer_pool.pop();
		// 在此处手动释放内存，因为pop操作并不会自动帮忙释放
		delete[] frame_buffer;

		end = clock();
		std::cout << "send to clint loop time = " << 1000 * double(end - start) / CLOCKS_PER_SEC << "ms" << std::endl;
	}
}

int main(void)
{

	// 首个线程开启服务器，并进行特定端口的监听
	std::thread server_boost_thread(server_startup);
	std::thread render_thread(init_and_render);

	// // 这里应该单开一个线程用于管理 output_flow
	// std::thread frame_pool_management(access_global_variable);

	// 这里应该单开一个线程用于向前端传送数据
	std::thread server_send_frame_to_client(send_img_pack_to_client);

	server_boost_thread.join();
	render_thread.join();
	// frame_pool_management.join();
	server_send_frame_to_client.join();

	// std::vector<int> i;
	// i.push_back(1);
	// i.push_back(2);
	// i.push_back(3);
	// std::cout << "vec end = " << *(i.end()-1) << std::endl;
	// i.pop_back();
	// std::cout << "vec end = " << *(i.end()-1) << std::endl;
	// i.pop_back();
	// std::cout << "vec end = " << *(i.end()-1) << std::endl;

	return 0;
}
