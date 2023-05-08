

export default {


    namespaced: true,

    actions: {},
    mutations: {
        updateTimeCostArr(state, update_pack) {
            // console.log("you've committed updateTimeCostArr!!!", update_pack);
            state.timeCostArr[0].arr.push({
                cost: update_pack.rCost,
                time: update_pack.runningTime
            });
            state.timeCostArr[1].arr.push({
                cost: update_pack.eCost,
                time: update_pack.runningTime
            });
            state.timeCostArr[2].arr.push({
                cost: update_pack.dCost,
                time: update_pack.runningTime
            });
            state.timeCostArr[3].arr.push({
                cost: update_pack.tCost,
                time: update_pack.runningTime
            });

            // 同时也要对 axisRange 进行更新
            if (update_pack.runningTime > state.axisRange[1]) {
                state.axisRange[1] = update_pack.runningTime;
                state.axisRange[0] = update_pack.runningTime - state.maxAxisScale;
            }
            // console.log("state.timeCostArr[0].arr.length = ", state.axisRange[0], state.axisRange[1])


            state.timeCostArr[0].arr = state.timeCostArr[0].arr.filter(d => {
                // console.log(state.axisRange, "sss", d.time);
                return d.time >= state.axisRange[0] && d.time <= state.axisRange[1]
            });
            state.timeCostArr[1].arr = state.timeCostArr[1].arr.filter(d => {
                return d.time >= state.axisRange[0] && d.time <= state.axisRange[1]
            });
            state.timeCostArr[2].arr = state.timeCostArr[2].arr.filter(d => {
                return d.time >= state.axisRange[0] && d.time <= state.axisRange[1]
            });
            state.timeCostArr[3].arr = state.timeCostArr[3].arr.filter(d => {
                return d.time >= state.axisRange[0] && d.time <= state.axisRange[1]
            });

            // console.log("state.timeCostArr[0].arr.length = ", state.timeCostArr[0].arr.length)


        }
    },
    state() {
        return {
            maxAxisScale: 20000,
            // 横轴展现最大时长
            axisRange: [0, 20000],
            // 折线颜色表
            colorMap: [
                "yellowgreen",
                "steelblue",
                "pink",
                "white"
            ],
            // 整个渲染-展示完整流程的时耗表
            timeCostArr: [
                {
                    name: "rCurve",
                    arr: []
                },
                {
                    name: "eCurve",
                    arr: []
                },
                {
                    name: "dCurve",
                    arr: []
                },
                {
                    name: "tCurve",
                    arr: []
                },
            ]
        }
    },
    getters: {}
}


