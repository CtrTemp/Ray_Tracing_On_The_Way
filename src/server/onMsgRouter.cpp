#include "server/onMessageRouter.h"

// 全局变量，整体的后台 websocket server
broadcast_server b_server;


void server_startup()
{
	b_server.run(CURRENT_PORT);
}
