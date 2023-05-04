import { createStore, useStore, mapState } from "vuex";
import footerPannel from "./footerPannel.js"
import siderPannel from "./siderPannel.js";


const store = createStore({

    namespaced: true,
    mutations: {
    },
    actions: {
    },
    state: {
        // ws: new WebSocket('ws://localhost:9002'),
        ws: new WebSocket('ws://localhost:9003'),
        request_time: 0,
        get_time: 0,
        total_time_cost: 0
    },
    modules: {
        footerPannel_Related: footerPannel,
        siderPannel_Related: siderPannel,
    }
})



export { store }



