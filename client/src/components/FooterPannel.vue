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
import { useStore } from "vuex";

import * as d3 from "d3"

const store = useStore();

// 整个渲染-展示完整流程的时耗表
const timeCostArr = store.state.footerPannel_Related.timeCostArr;

// // 横轴展现最大时长
const axisRange = store.state.footerPannel_Related.axisRange;

// 折线颜色表
const colorMap = store.state.footerPannel_Related.colorMap;


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
        top: height * 0.1,
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


    // Adding axis
    const xAxis = d3.axisBottom(xScale)
        .ticks(10)
        .tickFormat(d3.format("d"))



    const yAxis = d3.axisLeft(yScale)


    //添加一个g用于放x轴
    const gxAxis = svg.append("g")
        .attr("class", "xAxis")
        .attr("transform", "translate(" + margin.left + "," + (height - margin.bottom) + ")")
        .call(xAxis);


    //添加一个g用于放y轴
    const gyAxis = svg.append("g")
        .attr("class", "yAxis")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .call(yAxis);

    gxAxis.selectAll('.tick text').attr('font-size', '1.5vh');
    gyAxis.selectAll('.tick text').attr('font-size', '1.5vh');





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
        .attr("stroke-width", 3)
        .attr("stroke", colorMap[0]);
    svg.append("path").attr("class", "eCurve")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .attr("d", () => linePath(data_arr[1].arr))
        .attr("fill", "none")
        .attr("stroke-width", 3)
        .attr("stroke", colorMap[1]);
    svg.append("path").attr("class", "dCurve")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .attr("d", () => linePath(data_arr[2].arr))
        .attr("fill", "none")
        .attr("stroke-width", 3)
        .attr("stroke", colorMap[2]);
    svg.append("path").attr("class", "tCurve")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .attr("d", () => linePath(data_arr[3].arr))
        .attr("fill", "none")
        .attr("stroke-width", 3)
        .attr("stroke", colorMap[3]);



}

const update_axis = function (xRange, data_arr) {


    // xRange[1] += store.state.footerPannel_Related.maxAxisScale * 0.05;
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
    d3.select(".xAxis").call(xAxis).selectAll(".tick text").attr('font-size', '1.5vh');
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

onMounted(() => {


    render_init(timeCostArr);


    let time_offset = 0;




    // setInterval(() => {
    //     time_offset += 30;
    //     timeCostArr[0].arr.push({
    //         time: time_offset,
    //         cost: 30 + (Math.random() - 0.5) * 5
    //     })
    //     timeCostArr[1].arr.push({
    //         time: time_offset,
    //         cost: 15 + (Math.random() - 0.5) * 3
    //     })
    //     timeCostArr[2].arr.push({
    //         time: time_offset,
    //         cost: 65 + (Math.random() - 0.5) * 3
    //     })
    //     timeCostArr[3].arr.push({
    //         time: time_offset,
    //         cost: 120 + (Math.random() - 0.5) * 20
    //     })

    //     // 以下做法将过滤掉纵向坐标轴左边的不应被显示的曲线部分
    //     timeCostArr[0].arr = timeCostArr[0].arr.filter(d => {
    //         return d.time > scale_update[0];
    //     })
    //     timeCostArr[1].arr = timeCostArr[1].arr.filter(d => {
    //         return d.time > scale_update[0];
    //     })
    //     timeCostArr[2].arr = timeCostArr[2].arr.filter(d => {
    //         return d.time > scale_update[0];
    //     })
    //     timeCostArr[3].arr = timeCostArr[3].arr.filter(d => {
    //         return d.time > scale_update[0];
    //     })

    //     // 超出界限后开始同步更新坐标轴
    //     if (time_offset >= MAX_RANGE) {
    //         update_axis(scale_update, timeCostArr);
    //         scale_update[0] += 30;
    //         scale_update[1] += 30;
    //     }
    //     update_line(timeCostArr);
    //     // scale_update[0] += 30;
    //     // scale_update[1] += 30;
    //     // update_axis(scale_update, timeCostArr);
    // }, 30);







    // setInterval(() => {
    //     console.log("scale_update = ", scale_update);
    //     update_axis(scale_update);
    //     scale_update[0] += 10;
    //     scale_update[1] += 10;
    // }, 20);
})


watch(() => store.state.footerPannel_Related.timeCostArr, () => {
    // console.log("haha");


    // // 以下做法将过滤掉纵向坐标轴左边的不应被显示的曲线部分
    // timeCostArr[0].arr = timeCostArr[0].arr.filter(d => {
    //     return d.time > scale_update[0];
    // })
    // timeCostArr[1].arr = timeCostArr[1].arr.filter(d => {
    //     return d.time > scale_update[0];
    // })
    // timeCostArr[2].arr = timeCostArr[2].arr.filter(d => {
    //     return d.time > scale_update[0];
    // })
    // timeCostArr[3].arr = timeCostArr[3].arr.filter(d => {
    //     return d.time > scale_update[0];
    // })

    // // 超出界限后开始同步更新坐标轴
    // if (time_offset >= MAX_RANGE) {
    //     update_axis(scale_update, timeCostArr);
    //     scale_update[0] += 30;
    //     scale_update[1] += 30;
    // }
    update_line(timeCostArr);
    // scale_update[0] += 30;
    // scale_update[1] += 30;
    update_axis(store.state.footerPannel_Related.axisRange, timeCostArr);
}, { deep: true });

{
    // var width = document.getElementsByClassName("footer-pannel-svg")[0].clientWidth;
    // var height = document.getElementsByClassName("footer-pannel-svg")[0].clientHeight;

    // const margin = {
    //     top: height * 0.05,
    //     right: width * 0.025,
    //     bottom: height * 0.1,
    //     left: width * 0.05
    // }


    // var dataset = [
    //     {
    //         country: "china",
    //         gdp: [[2000, 11920], [2001, 13170], [2002, 14550],
    //         [2003, 16500], [2004, 19440], [2005, 22870],
    //         [2006, 27930], [2007, 35040], [2008, 45470],
    //         [2009, 51050], [2010, 59490], [2011, 73140],
    //         [2012, 83860], [2013, 103550],]
    //     },
    //     {
    //         country: "japan",
    //         gdp: [[2000, 47310], [2001, 41590], [2002, 39800],
    //         [2003, 43020], [2004, 46550], [2005, 45710],
    //         [2006, 43560], [2007, 43560], [2008, 48490],
    //         [2009, 50350], [2010, 54950], [2011, 59050],
    //         [2012, 59370], [2013, 48980],]
    //     }
    // ];

    // var gdpmax = 0;
    // for (var i = 0; i < dataset.length; i++) {
    //     var currGdp = d3.max(dataset[i].gdp, function (d) {
    //         return d[1];
    //     });
    //     if (currGdp > gdpmax)
    //         gdpmax = currGdp;
    // }
    // // console.log(gdpmax);

    // var xScale = d3.scaleLinear()
    //     .domain([2000, 2013])
    //     .range([0, width - margin.left - margin.right]);

    // var yScale = d3.scaleLinear()
    //     .domain([0, gdpmax * 1.1])
    //     .range([height - margin.bottom - margin.top, 0]);

    // var linePath = d3.line()//创建一个直线生成器
    //     .x(function (d) {
    //         return xScale(d[0]);
    //     })
    //     .y(function (d) {
    //         return yScale(d[1]);
    //     })
    //     .curve(d3.curveBasis); //插值模式

    // //定义两个颜色
    // var colors = [d3.rgb(0, 0, 255), d3.rgb(0, 255, 0)];

    // var svg = d3.select(".footer-pannel-svg")

    // svg.selectAll("path")
    //     .data(dataset)
    //     .enter()
    //     .append("path")
    //     .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
    //     .attr("d", function (d) {
    //         return linePath(d.gdp);
    //         //返回线段生成器得到的路径
    //     })
    //     .attr("fill", "none")
    //     .attr("stroke-width", 3)
    //     .attr("stroke", function (d, i) {
    //         return colors[i];
    //     });


    // // Adding axes
    // const xAxis = d3.axisBottom(xScale)
    //     .ticks(5)
    //     .tickFormat(d3.format("d"))

    // const yAxis = d3.axisLeft(yScale)


    // //添加一个g用于放x轴
    // svg.append("g")
    //     .attr("class", "axis")
    //     .attr("transform", "translate(" + margin.left + "," + (height - margin.bottom) + ")")
    //     .call(xAxis);

    // svg.append("g")
    //     .attr("class", "axis")
    //     .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
    //     .call(yAxis);

}
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

}

.footer-pannel-svg {
    width: 100%;
    height: 100%;
    /* border: 3px yellowgreen solid; */
    box-sizing: border-box;

    background-color: black;
    opacity: 40%;
}
</style>