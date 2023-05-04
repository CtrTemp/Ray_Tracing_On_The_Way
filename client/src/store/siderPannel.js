

export default {

    namespaced: true,

    actions: {},
    mutations: {
        updateTimeCostArr(state, update_pack) {
            state.renderCost.arr[0].value = update_pack.rCost;
            state.renderCost.arr[1].value = update_pack.eCost;
            state.renderCost.arr[2].value = update_pack.dCost;
            state.renderCost.arr[3].value = update_pack.tCost;


            state.otherCost.arr[0].value = update_pack.rCost;
            state.otherCost.arr[1].value = update_pack.eCost;
            state.otherCost.arr[2].value = update_pack.dCost + Math.random() * 10 + 5;
            state.otherCost.arr[3].value = update_pack.tCost;
        }
    },
    state() {
        return {
            renderCost: {
                title: "renderCost",
                arr: [
                    { name: "rCost", value: 3 },
                    { name: "eCost", value: 2 },
                    { name: "dCost", value: 1 },
                    { name: "tCost", value: 7 },
                ]
            },
            otherCost: {
                title: "otherCost",
                arr: [
                    { name: "rCost", value: 4 },
                    { name: "eCost", value: 3 },
                    { name: "dCost", value: 1 },
                    { name: "tCost", value: 9 },
                ]
            },
        }
    },
    getters: {}
}


