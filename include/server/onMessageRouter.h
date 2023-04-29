#ifndef MSG_ROUTER
#define MSG_ROUTER

#include <set>
#include <string>
#include <iostream>

#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>

#include <json/json.h>

// OpenCV 用于图片展示/解析
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

// base64 用于图片编码
#include "base64.h"


#define FRAME_WIDTH 1280
#define FRAME_HEIGHT 720

#define CURRENT_PORT 9002
// #define CURRENT_PORT 9003

static std::string rand_str(const int len) /*参数为字符串的长度*/
{
    /*初始化*/
    std::string str; /*声明用来保存随机字符串的str*/
    char c;          /*声明字符c，用来保存随机生成的字符*/
    int idx;         /*用来循环的变量*/
    /*循环向字符串中添加随机生成的字符*/
    for (idx = 0; idx < len; idx++)
    {
        /*rand()%26是取余，余数为0~25加上'a',就是字母a~z,详见asc码表*/
        c = 'a' + rand() % 26;
        str.push_back(c); /*push_back()是string类尾插函数。这里插入随机字符c*/
    }
    return str; /*返回生成的随机字符串*/
}

// void server_handler_close_server(server current_server);
// void server_handled_get_test_json(server current_server, con_list current_connections);
// void server_handler_get_test_frame_pack(server current_server, con_list current_connections);

typedef websocketpp::server<websocketpp::config::asio> server;

using websocketpp::connection_hdl;
using websocketpp::lib::bind;
using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;

class broadcast_server
{
public:
    broadcast_server()
    {
        m_server.init_asio();

        m_server.set_open_handler(bind(&broadcast_server::on_open, this, ::_1));
        m_server.set_close_handler(bind(&broadcast_server::on_close, this, ::_1));
        m_server.set_message_handler(bind(&broadcast_server::on_message, this, ::_1, ::_2));
    }

    void on_open(connection_hdl hdl)
    {
        m_connections.insert(hdl);
    }

    void on_close(connection_hdl hdl)
    {
        m_connections.erase(hdl);
    }

    void on_message(connection_hdl hdl, server::message_ptr json_msg_pack)
    {

        printf("\n\n\n");

        std::string message_str = json_msg_pack->get_payload();
        // 1.以字符串的形式将收到的数据进行后台打印输出
        std::cout << "message_str = " << message_str << std::endl;

        // 2.将收到的 json_pack 进行解析，得到 JSON obj
        Json::Reader reader;
        Json::Value parsed_json_obj;

        bool err = reader.parse(message_str, parsed_json_obj);

        if (!err) // 如果解析出问题则直接返回
        {
            std::cout << "parse json err has happened, please check!" << std::endl;
            // throw std::runtime_error("parse json err has happened, please check!\n");
            return;
        }
        if (!parsed_json_obj["cmd"].isString()) // 如果json没有cmd这个key，则直接返回
        {
            std::cout << "Invalid Json Pack with no cmd keyword, please check!" << std::endl;
            // throw std::runtime_error("Invalid Json Pack with no cmd keyword, please check!\n");
            return;
        }

        std::string cmd_str = parsed_json_obj["cmd"].asString();

        // 将 cmd 进行打印输出
        std::cout << "cmd = " << cmd_str << std::endl;

        if (cmd_str == "close")
        {
            m_server.stop();
            std::cout << "server_has_closed" << std::endl;
        }
        else if (cmd_str == "get_json")
        {
            std::cout << "'get_json' branch not supported" << std::endl;
        }
        else if (cmd_str == "get_frame_pack")
        {

            // 生成噪声图 raw data，数据类型是 OpenCV 的 Mat 类型
            cv::Mat img = cv::Mat(cv::Size(FRAME_WIDTH, FRAME_HEIGHT), CV_8UC3);
            for (int i = 0; i < 10; i++)
            {
                for (int row = 0; row < FRAME_HEIGHT; row++)
                {
                    for (int col = 0; col < FRAME_WIDTH; col++)
                    {
                        const int global_index = row * FRAME_WIDTH + col;

                        int ir = i * 20 + 20;
                        int ig = i * 20 + 20;
                        int ib = i * 20 + 20;

                        // int ir = rand() % 255;
                        // int ig = rand() % 255;
                        // int ib = rand() % 255;

                        img.at<unsigned char>(row, col * 3 + 0) = ib;
                        img.at<unsigned char>(row, col * 3 + 1) = ig;
                        img.at<unsigned char>(row, col * 3 + 2) = ir;
                    }
                }

                // std::string window_name = "windows temp";
                // cv::imshow(window_name, img);
                // cv::waitKey(30);

                // 将其转化为 base64 编码的str 默认转成 jpg 编码格式
                // Mat转base64

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
                m_server.clear_access_channels(websocketpp::log::alevel::all);
                m_server.clear_access_channels(websocketpp::log::alevel::frame_payload);

                // 向所有的 client 端广播信息
                websocketpp::frame::opcode::value sCode = websocketpp::frame::opcode::TEXT;
                for (auto it : m_connections)
                {
                    m_server.send(it, json_str, sCode);
                }
            }
        }
    }

    void run(uint16_t port)
    {
        m_server.listen(port);
        m_server.start_accept();
        m_server.run();
    }

public:
    typedef std::set<connection_hdl, std::owner_less<connection_hdl>> con_list;

    server m_server;
    con_list m_connections;
};

#endif

extern broadcast_server b_server;
void server_startup();