

export default {

    namespaced: true,

    actions: {},
    mutations: {
        flip_pause_begin(state, pause_begin) {
            if (pause_begin == "pause") {
                state.pause = true;
            }
            else {
                state.pause = false;
            }
            console.log("flip_pause_begin committed !!! current pause state is ", state.pause);
        }
    },
    state() {
        return {
            pause: true
        }
    },
    getters: {}
}


