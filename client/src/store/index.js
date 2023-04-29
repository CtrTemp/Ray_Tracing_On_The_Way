import { createStore, useStore, mapState } from "vuex";


const store = createStore({

    namespaced: true,
    mutations: {
    },
    actions: {
    },
    state: {
        ws: new WebSocket('ws://localhost:9002'),
        // ws: new WebSocket('ws://localhost:9003'),
        request_time: 0,
        get_time: 0
    },
})



export { store }



