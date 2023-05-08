<template>
    <div class="sider-bar-chart-container">
        <svg class="sider-bar-chart-svg">
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

// 以下变量在mounted后初始化
let width;
let height;
let margin;

// x 轴是时间轴（也可以理解为多少帧）
let xScale;

// y 轴是当前帧耗时轴，单位为ms
let yScale;


const bar_render_init = function () {


    width = document.getElementsByClassName("sider-bar-chart-svg")[0].clientWidth;
    height = document.getElementsByClassName("sider-bar-chart-svg")[0].clientHeight;

    margin = {
        top: height * 0.15,
        bottom: height * 0.1,
        left: width * 0.15,
        right: width * 0.05,
    };

    // x 轴是时耗轴
    xScale = d3.scaleLinear()
        .domain([0, 100])
        .range([0, width - margin.left - margin.right]);

    // y 轴是时耗分项

    const yScaleDomain = ["rCost", "eCost", "dCost", "tCost"];
    let yScaleRange = [];
    for (let i = 0; i < yScaleDomain.length; i++) {
        yScaleRange[i] = (height - margin.bottom - margin.top) * ((i + 0.5) / 4);
    }
    yScale = d3.scaleOrdinal()
        .domain(yScaleDomain)
        .range(yScaleRange);


    const svg = d3.select(".sider-bar-chart-svg");

    // Adding axis
    const xAxis = d3.axisTop(xScale)
        .ticks(10)
        .tickFormat(d3.format("d"))


    const yAxis = d3.axisLeft(yScale)


    //添加一个g用于放x轴
    const gxAxis = svg.append("g")
        .attr("class", "xAxis_SiderBar")
        .attr("transform", "translate(" + margin.left + "," + (margin.top) + ")")
        .call(xAxis)
        .call(g => g.select(".domain").remove())
        .call(g => g.selectAll(".tick line").clone()
            .attr("y1", height - margin.top - margin.bottom)
            .attr("y2", 0)
            // .attr("transform", `translate(${0},${-(height - margin.top - margin.bottom)})`)
            .attr("stroke", "white")
            .attr("stroke-dasharray", "3,3")
            .attr("stroke-opacity", 0.35));


    //添加一个g用于放y轴
    const gyAxis = svg.append("g")
        .attr("class", "yAxis_SiderBar")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
        .call(yAxis)
        .call(g => g.select(".domain").remove())

    gxAxis.selectAll('.tick text').attr('font-size', '1.5vh').attr("fill", "white");
    gyAxis.selectAll('.tick text').attr('font-size', '1.5vh').attr("fill", "white");

    gxAxis.selectAll("line").attr("stroke", "white");
    gxAxis.selectAll(".domain").attr("stroke", "white").attr("opacity", 0.4);
    gyAxis.selectAll("line").attr("stroke", "white");
    gyAxis.selectAll(".domain").attr("stroke", "white").attr("opacity", 0.4);


    // Bar Rect
    const data_arr = [
        { name: "rCost", value: 50 },
        { name: "eCost", value: 70 },
        { name: "dCost", value: 5 },
        { name: "tCost", value: 90 },
    ]

    const rect_group = svg.append("g")
        .attr("class", "bar_group")
        .selectAll("rect")
        .data(data_arr)
        .join("rect")
        .attr("transform", `translate(${margin.left},${margin.top})`)
        .attr("x", 0)
        .attr("y", d => { return yScale(d.name) - height * 0.04 })
        .attr("width", d => { return xScale(d.value); })
        .attr("height", "8%")
        .attr("fill", (d, i) => { return color_map[i] })


}

const bar_render_update = function () {
    const data_arr = store.state.siderPannel_Related.renderCost.arr;

    const rect_group = d3.select(".bar_group")
        .selectAll("rect")
        .data(data_arr)
        // .transition() 
        // .duration(30) 
        .attr("x", 0)
        .attr("y", d => { return yScale(d.name) - height * 0.04 })
        .attr("width", d => { return xScale(d.value); })
        .attr("height", "8%")
        .attr("fill", (d, i) => { return color_map[i] })

}


onMounted(() => {
    bar_render_init();
})

watch(() => store.state.siderPannel_Related.renderCost, () => {
    bar_render_update();
}, { deep: true });

</script>

 
<style scoped>
.sider-bar-chart-container {
    width: 100%;
    height: 30%;

    /* border: 3px yellowgreen solid; */
    box-sizing: border-box;
}

.sider-bar-chart-svg {
    width: 100%;
    height: 100%;
}
</style>