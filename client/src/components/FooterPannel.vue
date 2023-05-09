<template>
    <div class="footer-svg-container">
        <svg class="footer-pannel-svg">
        </svg>
    </div>
</template>
 
<script setup>
import { reactive, ref } from "vue";
import { computed, watch } from "vue";
import { onMounted } from "vue";
import { inject } from "vue";
import { useStore } from "vuex";

import * as d3 from "d3"
import { upperCase } from "lodash";

const store = useStore();

const color_map = Object.values(inject("color_map").value);


// 整个渲染-展示完整流程的时耗表
const timeCostArr = store.state.footerPannel_Related.timeCostArr;

// 横轴展现最大时长
const axisRange = store.state.footerPannel_Related.axisRange;


// 以下变量在mounted后初始化
let width;
let height;
let margin;

// x 轴是时间轴（也可以理解为多少帧）
let xScale;

// y 轴是当前帧耗时轴，单位为ms
let yScale;


const render_init = function (data_arr) {

    width = document.getElementsByClassName("footer-pannel-svg")[0].clientWidth;
    height = document.getElementsByClassName("footer-pannel-svg")[0].clientHeight;

    margin = {
        top: height * 0.2,
        bottom: height * 0.1,
        left: width * 0.035,
        right: width * 0.025,
    };

    xScale = d3.scaleLinear()
        .domain(axisRange)
        .range([0, width - margin.left - margin.right]);
    yScale = d3.scaleLinear()
        .domain([0, 100])
        .range([height - margin.bottom - margin.top, 0]);


    const svg = d3.select(".footer-pannel-svg");


    // Add title
    svg.append("text")
        .text("Render Loop Time Cost / ms")
        .attr("text-anchor", "start")
        .attr("font-size", "3vh")
        .attr("x", margin.left)
        .attr("y", "3.5vh")
        .attr("fill", "white")

    // Adding axis
    const xAxis = d3.axisBottom(xScale)
        .ticks(10)
        .tickFormat(d3.format("d"))


    const yAxis = d3.axisLeft(yScale)
        .ticks(10)
        .tickFormat(d3.format("d"))


    //添加一个g用于放x轴
    const gxAxis = svg.append("g")
        .attr("class", "xAxis")
        .attr("transform", "translate(" + margin.left + "," + (height - margin.bottom) + ")")
        .call(xAxis);


    //添加一个g用于放y轴
    const gyAxis = svg.append("g")
        .attr("class", "yAxis")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .call(yAxis)
        // .call(g => g.select(".domain").remove())
        .call(g => g.selectAll(".tick line").clone()
            .attr("x2", width - margin.left - margin.right)
            .attr("stroke", "white")
            .attr("stroke-dasharray", "5,10")
            // .attr("stroke-dashoffset", 5)
            .attr("stroke-opacity", 0.35))

    gxAxis.selectAll('.tick text').attr('font-size', '1.5vh').attr("fill", "white");
    gyAxis.selectAll('.tick text').attr('font-size', '1.5vh').attr("fill", "white");

    gxAxis.selectAll("line").attr("stroke", "white");
    gxAxis.selectAll(".domain").attr("stroke", "white").attr("opacity", 0.4);
    gyAxis.selectAll("line").attr("stroke", "white");
    gyAxis.selectAll(".domain").attr("stroke", "white").attr("opacity", 0.4);




    // 初始化折线图
    var linePath = d3.line()//创建一个直线生成器
        .x(function (d) {
            // console.log("x = ", yScale(d["time"]));
            return xScale(d["time"]);
        })
        .y(function (d) {
            // console.log("y = ", yScale(d["cost"]));
            return yScale(d["cost"]);
        })
        .curve(d3.curveCardinal.tension(0.25)); //插值模式
    // .curve(d3.curveBasisClosed); //插值模式


    // console.log("data arr = ", data_arr);

    // svg.selectAll("path")
    //     .data(data_arr)
    //     .join()
    //     .append("path")
    //     .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
    //     .attr("d", function (d) {
    //         // console.log("d = ", linePath(d.arr));
    //         return linePath(d.arr);
    //         //返回线段生成器得到的路径
    //     })
    //     .attr("fill", "none")
    //     .attr("stroke-width", 3)
    //     .attr("stroke", "black");


    svg.append("path").attr("class", "rCurve")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .attr("d", () => linePath(data_arr[0].arr))
        .attr("fill", "none")
        .attr("stroke-width", "0.25vh")
        .attr("stroke", color_map[0]);
    svg.append("path").attr("class", "eCurve")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .attr("d", () => linePath(data_arr[1].arr))
        .attr("fill", "none")
        .attr("stroke-width", "0.25vh")
        .attr("stroke", color_map[1]);
    svg.append("path").attr("class", "dCurve")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .attr("d", () => linePath(data_arr[2].arr))
        .attr("fill", "none")
        .attr("stroke-width", "0.25vh")
        .attr("stroke", color_map[2]);
    svg.append("path").attr("class", "tCurve")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .attr("d", () => linePath(data_arr[3].arr))
        .attr("fill", "none")
        .attr("stroke-width", "0.25vh")
        .attr("stroke", color_map[3]);

    // hover line
    let hover_line = svg.append("line")
        .attr("id", "hover_line")
        .attr("stroke", "yellow")
        .attr("stroke-width", "0.25vh")
        .attr("y1", yScale(0))
        .attr("y2", yScale.range()[1])
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    let hover_point_r = svg.append("circle")
        .attr("id", "hover_point_r")
        .attr("r", "0.5vh")
        .attr("stroke", "gold")
        .attr("stroke-width", "0.25vh")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");


    let hover_point_e = svg.append("circle")
        .attr("id", "hover_point_e")
        .attr("r", "0.5vh")
        .attr("stroke", "gold")
        .attr("stroke-width", "0.25vh")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    let hover_point_d = svg.append("circle")
        .attr("id", "hover_point_d")
        .attr("r", "0.5vh")
        .attr("stroke", "gold")
        .attr("stroke-width", "0.25vh")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    let hover_point_t = svg.append("circle")
        .attr("id", "hover_point_t")
        .attr("r", "0.5vh")
        .attr("stroke", "gold")
        .attr("stroke-width", "0.25vh")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");


    // hover line interaction
    svg.on("mousemove", function (a, b, c) {
        // demo 运行阶段应该屏蔽掉 hover
        if (!store.state.controlBar_Related.pause) {
            return;
        }
        var location = d3.pointer(a);

        let x1 = location[0] - margin.left;
        if (x1 < 0) {
            x1 = 0;
        }
        if (x1 > width - margin.left - margin.right) {
            x1 = width - margin.left - margin.right;
        }
        // var y1 = location[1]

        // console.log(`[${x1},${y1}]`, width);

        var bisectLeft = d3.bisector(function (d) { return d.time; }).left;
        var idx = bisectLeft(timeCostArr[0].arr, xScale.invert(x1));


        if (idx >= timeCostArr[0].arr.length) { return; }

        var datum = timeCostArr[0].arr[idx];
        var time = datum.time;



        hover_line
            .attr("x1", xScale(time))
            .attr("x2", xScale(time));

        var r_cost = timeCostArr[0].arr[idx].cost;
        var e_cost = timeCostArr[1].arr[idx].cost;
        var d_cost = timeCostArr[2].arr[idx].cost;
        var t_cost = timeCostArr[3].arr[idx].cost;


        hover_point_r.attr("cx", xScale(time)).attr("cy", yScale(r_cost));
        hover_point_e.attr("cx", xScale(time)).attr("cy", yScale(e_cost));
        hover_point_d.attr("cx", xScale(time)).attr("cy", yScale(d_cost));
        hover_point_t.attr("cx", xScale(time)).attr("cy", yScale(t_cost));

        // hover 后要同时更新 Cost 相关信息，以便更新同区域内的 Dount-Chart
        const update_pack = {
            rCost: timeCostArr[0].arr[idx].cost,
            eCost: timeCostArr[1].arr[idx].cost,
            dCost: timeCostArr[2].arr[idx].cost,
            tCost: timeCostArr[3].arr[idx].cost,
        }

        const updateCostStr = "siderPannel_Related/updateTimeCostArr";
        store.commit(updateCostStr, update_pack);
        const updateCurrentFrameStr = "footerPannel_Related/updateCurrentFrame";
        store.commit(updateCurrentFrameStr, idx);

    })

}

const update_axis = function (xRange, data_arr) {

    let real_range = JSON.parse(JSON.stringify(xRange));
    real_range[1] = real_range[1] + store.state.footerPannel_Related.maxAxisScale * 0.01;
    // xScale 更新
    xScale = d3.scaleLinear()
        .domain(real_range)
        .range([0, width - margin.left - margin.right]);

    // Adding axis
    const xAxis = d3.axisBottom(xScale)
        .ticks(10)
        .tickFormat(d3.format("d"));

    // 这里最好不要加过渡动画效果
    d3.select(".xAxis")
        .call(xAxis)
        .selectAll(".tick text")
        .attr('font-size', '1.5vh')
        .attr("fill", "white");
    // gxAxis.selectAll('.tick text').attr('font-size', '1.5vh');
    // gyAxis.selectAll('.tick text').attr('font-size', '1.5vh');
    // const g = d3.select(".xAxis").transition().duration(1000).call(xAxis);


    // // yScale 更新
    // let max_time_cost = 130; // 设定 yScale 的最小值就是150
    // for (let i = 0; i < data_arr[3].arr.length; i++) {
    //     if (data_arr[3].arr[i].cost > max_time_cost) {
    //         max_time_cost = data_arr[3].arr[i].cost;
    //     }
    // }

    // yScale = d3.scaleLinear()
    //     .domain([0, max_time_cost * 1.2])
    //     .range([height - margin.bottom - margin.top, 0]);

    // // Adding axis
    // const yAxis = d3.axisLeft(yScale);

    // //添加一个g用于放y轴
    // d3.select(".yAxis").call(yAxis).selectAll(".tick text").attr('font-size', '1.5vh');
}

const update_line = function (data_arr) {

    const svg = d3.select(".footer-pannel-svg");

    var linePath = d3.line()//创建一个直线生成器
        .x(function (d) {
            return xScale(d["time"]);
        })
        .y(function (d) {
            return yScale(d["cost"]);
        })
    // 不要插值看起来效果更好
    // .curve(d3.curveCardinal.tension(0.25)); //插值模式

    svg.select(".rCurve")
        .datum(data_arr[0].arr)
        .transition().duration(0)
        .attr("d", linePath)
    svg.select(".eCurve")
        .datum(data_arr[1].arr)
        .transition().duration(0)
        .attr("d", linePath)
    svg.select(".dCurve")
        .datum(data_arr[2].arr)
        .transition().duration(0)
        .attr("d", linePath)
    svg.select(".tCurve")
        .datum(data_arr[3].arr)
        .transition().duration(0)
        .attr("d", linePath)

}

const update_hover_line = function (data_arr) {
    const update_data_pack = [
        { name: "rCost", value: 0 },
        { name: "eCost", value: 0 },
        { name: "dCost", value: 0 },
        { name: "tCost", value: 0 },
    ];
    for (let i = 0; i < data_arr.length; i++) {
        const arr_length = data_arr[i].arr.length;
        update_data_pack[i].value = data_arr[i].arr[arr_length - 1].cost;
    }

    // console.log("update_data_pack = ", update_data_pack);
    // const commit_str = "siderPannel_Related/updateTimeCostArr";
    // store.commit(commit_str, update_data_pack);

    // 对 hover_line 的显示进行更新
    // var bisectLeft = d3.bisector(function (d) { return d.time; }).left;
    // var idx = bisectLeft(data_arr[0].arr, xScale.invert(x1));


    // if (idx >= data_arr[0].arr.length) { return; }

    var time = data_arr[0].arr[data_arr[0].arr.length - 1].time;

    d3.select("#hover_line")
        .attr("x1", xScale(time))
        .attr("x2", xScale(time));

    var r_cost = update_data_pack[0].value;
    var e_cost = update_data_pack[1].value;
    var d_cost = update_data_pack[2].value;
    var t_cost = update_data_pack[3].value;


    d3.select("#hover_point_r").attr("cx", xScale(time)).attr("cy", yScale(r_cost));
    d3.select("#hover_point_e").attr("cx", xScale(time)).attr("cy", yScale(e_cost));
    d3.select("#hover_point_d").attr("cx", xScale(time)).attr("cy", yScale(d_cost));
    d3.select("#hover_point_t").attr("cx", xScale(time)).attr("cy", yScale(t_cost));


}

// 挂载完成后进行初始化
onMounted(() => {

    const arr_temp = [
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
    ];
    const arr_len = 300;
    for (let i = 0; i < 4; i++) {
        for (let j = 0; j < arr_len; j++) {
            arr_temp[i].arr.push({
                time: j * store.state.footerPannel_Related.maxAxisScale / arr_len,
                cost: Math.random() * 10 + 20 * (i + 1)
            })
        }
    }
    render_init(arr_temp);
})

// 每当数据有更新立，即进行刷新显示
watch(() => store.state.footerPannel_Related.timeCostArr, () => {
    let timer1 = new Date();
    const start = timer1.getMinutes() * 60000 + timer1.getSeconds() * 1000 + timer1.getMilliseconds();
    update_line(timeCostArr);
    update_axis(store.state.footerPannel_Related.axisRange, timeCostArr);
    update_hover_line(timeCostArr);
    let timer2 = new Date();
    const end = timer2.getMinutes() * 60000 + timer2.getSeconds() * 1000 + timer2.getMilliseconds();

    // console.log("update footer chart time cost = ", end - start);

}, { deep: true });

</script>
 
<style scoped>
.footer-svg-container {

    position: absolute;

    top: 75%;
    left: 0%;
    width: 100%;
    height: 25%;

    /* border: red solid 3px; */
    box-sizing: border-box;
    background-color: rgba(0, 0, 0, 0.25);

}

.footer-pannel-svg {
    width: 100%;
    height: 100%;
    /* border: 3px yellowgreen solid; */
    box-sizing: border-box;
}
</style>