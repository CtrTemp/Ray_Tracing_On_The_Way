<template>
    <div class="donut-container">
        <svg :class="`donut-svg donut-svg${classIdx}`">
        </svg>
        <div id="tooltip" class="hidden">
            <p><span id="game"></span>: <strong><span id="value"></span>%</strong></p>
        </div>
    </div>
</template>

<script setup>
import { reactive, ref } from "vue";
import { computed, watch } from "vue";

import { onMounted } from "vue";


import { inject } from "vue";

import * as d3 from "d3"


const props = defineProps(["donutDeps", "classIdx"]);

const color_map = Object.values(inject("color_map").value);



// 以下变量在mounted后初始化
let width;
let height;
let margin;

function showToolTip(selection) {

    const dataset = [
        { game: "Dota 2", value: 68 },
        { game: "Skeletal Skism", value: 21 },
        { game: "Edifice of Fiends", value: 6 },
        { game: "Yu-Gi-Oh! MD", value: 2 },
        { game: "CS:GO", value: 1 },
        { game: "Other (14)", value: 2 }
    ];

    selection
        .on("mousemove", function (event, d) {
            const [x, y] = d3.pointer(event);

            d3.select("#tooltip").style("top", `${y + 280}px`).style("left", `${x + 330}px`);

            d3.select("#tooltip").select("#game").text(dataset[d.index].game);
            d3.select("#tooltip").select("#value").text(d.value);

            d3.select("#tooltip").classed("hidden", false);
        })
        .on("mouseout", function (event, d) {
            d3.select("#tooltip").classed("hidden", true);
        });
}


const render_init = function (data_arr) {

    width = document.getElementsByClassName(`donut-svg${props.classIdx}`)[0].clientWidth;
    height = document.getElementsByClassName(`donut-svg${props.classIdx}`)[0].clientHeight;


    // donut-chart 部分
    margin = {
        top: height * 0.15,
        bottom: height * 0.05,
        left: width * 0.05,
        right: width * 0.05,
    };

    // set the dimensions and margins of the graph
    const cScale = d3.scaleOrdinal(d3.schemeTableau10);

    const svg = d3.select(`.donut-svg${props.classIdx}`);




    // const radius = (width - margin.left - margin.right) / 2;
    const radius = (height - margin.top - margin.bottom) / 2;

    const pie = d3
        .pie()
        .sort(null)
        .value((d) => d.value)(data_arr);

    const arc = d3.arc().innerRadius(radius * 0.65).outerRadius(radius);


    // title

    const title = svg.append("text")
        .text(props.donutDeps.title)
        .attr("x", "55%")
        .attr("y", "15%")
        .attr("fill", "white")
        .attr("font-size", "3vh")

    // 圆环
    const sections = svg
        .selectAll("arc")
        .data(pie)
        .enter()
        .append("g")
        .attr("transform", `translate(${radius + margin.left}, ${height / 2})`);

    // hover 文字显示
    sections
        .append("path")
        .attr("d", arc)
        .attr("fill", (d, i) => color_map[i]).transition().delay(function (d, i) { return i * 800 }).duration(800)
    // 暂时先不要显示文字
    // .call(showToolTip); 

    // // 环上图元文字
    // sections
    //     .append("text")
    //     .text((d, i) => `${data_arr[i].game}`)
    //     .attr("transform", (d) => `translate(${arc.centroid(d)})`)
    //     .attr("text-anchor", "middle")
    //     .style("font-size", (d, i) => data_arr[i].value / 2.5 + 8)
    //     .style("fill", "white")
    //     .style("font-weight", 600);


    // 环形中央文字
    svg.append("text")
        .attr("class", "percentage")
        .attr("x", radius + margin.left)
        .attr("y", radius + margin.top)
        .attr("text-anchor", "middle")
        .text("percent")
        .attr("fill", "white")
        .attr("font-size", "2vh")

    // legend 部分

    const legend_gap = (height - margin.top - margin.bottom) / (data_arr.length + 1);

    const legend = svg.append('g')
        .attr('class', 'legend')
        .attr('transform', `translate(${radius + margin.left}, ${margin.top})`);

    const lg = legend.selectAll('g')
        .data(data_arr)
        .enter()
        .append('g')
        .attr('class', 'legendGroup')
        .attr('transform', (d, i) => {
            return `translate(${radius + margin.left}, ${legend_gap * (i + 0.5)})`;
        });

    lg.append('rect')
        .attr('fill', (d, i) => {
            return color_map[i]
        })
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', "1vw")
        .attr('height', "1vw");

    lg.append('text')
        .style('font-size', '2vh')
        .attr("fill", "white")
        .attr('x', "2vw")
        .attr('y', "1.5vh")
        .text(d => d.name);

}

const render_update = function (data_arr) {
    const svg = d3.select(`.donut-svg${props.classIdx}`);
    const cScale = d3.scaleOrdinal(d3.schemeTableau10);
    const pie = d3
        .pie()
        .sort(null)
        .value((d) => d.value)(data_arr);

    const radius = (height - margin.top - margin.bottom) / 2;
    const arc = d3.arc().innerRadius(radius * 0.65).outerRadius(radius);



    function tweenArc(b) {
        return function (a, i) {
            var d = b.call(this, a, i),
                i = d3.interpolate(d, a);
            return function (t) {
                return arc(i(t));
            };
        };
    }

    // 圆环
    const sections = svg
        .selectAll("arc")
        .data(pie)
        .enter()
        .append("g")
        .attr("transform", `translate(${radius + margin.left}, ${height / 2})`)


    sections
        .append("path")
        .attr("d", arc)
        .attr("fill", (d, i) => color_map[i])
    // 暂时先不使用过渡动画效果
    // .transition().duration(800)
    // .attrTween("d", tweenArc(function (d, i) {
    //     return {
    //         startAngle: d.laststartAngle,
    //         endAngle: d.lastendAngle,
    //     };
    // }))

    const total_time_cost = data_arr[0].value + data_arr[1].value + data_arr[2].value + data_arr[3].value;
    const rCostPercentage_num = 100 * (data_arr[0].value / total_time_cost);
    const rCostPercentage_str = rCostPercentage_num.toFixed(2) + "%";
    // 圆环中心文字
    const innerText = svg.select(".percentage")
        .text(rCostPercentage_str)
        .attr("font-size", "2vh");
}

onMounted(() => {

    const static_data = [
        { name: "Dota 2", value: 68 },
        { name: "Skeletal Skism", value: 21 },
        { name: "Edifice of Fiends", value: 6 },
        { name: "Yu-Gi-Oh! MD", value: 2 },
        { name: "CS:GO", value: 1 },
        { name: "Other (14)", value: 2 }
    ];

    const init_data_placeholder = [
        { name: "rCost", value: 3 },
        { name: "eCost", value: 2 },
        { name: "dCost", value: 1 },
        { name: "tCost", value: 7 },
    ];


    render_init(props.donutDeps.arr);
})


watch(() => props.donutDeps, () => {
    let timer1 = new Date();
    const start = timer1.getMinutes() * 60000 + timer1.getSeconds() * 1000 + timer1.getMilliseconds();
    render_update(props.donutDeps.arr);
    let timer2 = new Date();
    const end = timer2.getMinutes() * 60000 + timer2.getSeconds() * 1000 + timer2.getMilliseconds();
    // console.log("update footer chart time cost = ", end - start);
}, { deep: true })

// setInterval(() => {
//     const update_arr = [
//         { name: "rCost", value: 6 },
//         { name: "eCost", value: 3 },
//         { name: "dCost", value: 1 },
//         { name: "tCost", value: 5 },
//     ]
//     for (let i = 0; i < update_arr.length; i++) {
//         update_arr[i].value = update_arr[i].value + Math.random();
//     }
//     render_update(update_arr);
// }, 50);

</script>
 
<style scoped>
.donut-container {
    width: 100%;
    height: 35%;
    /* border: 3px rebeccapurple solid; */
}

.donut-svg {
    width: 100%;
    height: 100%;
    /* border: 3px yellowgreen solid; */
    box-sizing: border-box;

    background-color: black;
    opacity: 40%;
}


#tooltip {
    position: absolute;
    width: 100px;
    height: auto;
    padding: 10px;
    background-color: black;
    opacity: 0.7;
    color: white;
    -webkit-border-radius: 10px;
    -moz-border-radius: 10px;
    border-radius: 10px;
    -webkit-box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.4);
    -moz-box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.4);
    box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.4);
    pointer-events: none;
}

#tooltip.hidden {
    display: none;
}

#tooltip p {
    margin: 0;
    font-family: sans-serif;
    font-size: 16px;
    line-height: 20px;
}
</style>