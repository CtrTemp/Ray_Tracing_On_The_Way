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
        .call(g => g.select(".domain").remove())
        .call(g => g.selectAll(".tick line").clone()
            .attr("x2", width - margin.left - margin.right)
            .attr("stroke", "white")
            .attr("stroke-dasharray", "5,10")
            // .attr("stroke-dashoffset", 5)
            .attr("stroke-opacity", 0.35))

    gxAxis.selectAll('.tick text').attr('font-size', '1.5vh').attr("fill", "white");
    gyAxis.selectAll('.tick text').attr('font-size', '1.5vh').attr("fill", "white");


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
        .attr("stroke-width", "0.5vh")
        .attr("stroke", color_map[0]);
    svg.append("path").attr("class", "eCurve")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .attr("d", () => linePath(data_arr[1].arr))
        .attr("fill", "none")
        .attr("stroke-width", "0.5vh")
        .attr("stroke", color_map[1]);
    svg.append("path").attr("class", "dCurve")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .attr("d", () => linePath(data_arr[2].arr))
        .attr("fill", "none")
        .attr("stroke-width", "0.5vh")
        .attr("stroke", color_map[2]);
    svg.append("path").attr("class", "tCurve")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .attr("d", () => linePath(data_arr[3].arr))
        .attr("fill", "none")
        .attr("stroke-width", "0.5vh")
        .attr("stroke", color_map[3]);

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