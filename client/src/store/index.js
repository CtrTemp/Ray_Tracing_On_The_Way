import { createStore, useStore, mapState } from "vuex";
import footerPannel from "./footerPannel.js"
import siderPannel from "./siderPannel.js";
import controlBar from "./controlBar.js";


const store = createStore({

    namespaced: true,
    mutations: {
        update_running_time(state, current_time) {
            state.total_running_time = current_time - state.begin_time_mark;
        },

        update_pause_time(state, pause_time) {
            state.total_pause_time += (pause_time - state.pause_time_mark);
        },
    },
    actions: {
    },
    state: {
        ws: new WebSocket('ws://localhost:9002'),
        // ws: new WebSocket('ws://localhost:9003'),
        request_time: 0,
        get_time: 0,
        total_time_cost: 0, // 2023-05-08 弃用
        begin_time_mark: 0, // 程序开始时的时间起点
        pause_time_mark: 0, // 点击暂停demo时的时间起点
        total_running_time: 0, // 从页面建立开始，就一刻不停计时
        total_pause_time: 0,  // 点击停止按钮后开始累计计时
    },
    modules: {
        footerPannel_Related: footerPannel,
        siderPannel_Related: siderPannel,
        controlBar_Related: controlBar,
    }
})



export { store }



