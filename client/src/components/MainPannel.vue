<template>
    <div class="main-pannel-container">
        <div class="img-container">
            <canvas id="primiaryCanvas"></canvas>
            <canvas id="secondaryCanvas"></canvas>
            <!-- <img :src=balls_url alt="" class="basic_img img_balls"> -->
            <!-- <img :src=bunnies_url alt="" class="basic_img img_bunnies"> -->
        </div>
        <div class="canvas-aux-pannel">
            <div class="mouse-pos-bar">
                <div>x = {{ store.state.mainPannel_Related.mouseX.toFixed(3) }}</div>
                <div>y = {{ store.state.mainPannel_Related.mouseY.toFixed(3) }}</div>
            </div>
            <div class="sphere-angle-bar">
                <!-- <div>pitch({{ store.state.mainPannel_Related.pitch }})</div>
                <div>roll({{ store.state.mainPannel_Related.roll }})</div>
                <div>yaw({{ store.state.mainPannel_Related.yaw }})</div> -->
                <div>theta({{ store.state.mainPannel_Related.theta.toFixed(3) }})</div>
                <div>phi({{ store.state.mainPannel_Related.phi.toFixed(3) }})</div>
            </div>
            <div class="sphere-angle-bar">
                <div>d_Theta({{ store.state.mainPannel_Related.deltaTheta.toFixed(2) }})</div>
                <div>d_Phi({{ store.state.mainPannel_Related.deltaPhi.toFixed(2) }})</div>
            </div>
        </div>
    </div>
</template>
 
<script setup>
import { reactive, ref } from "vue";
import { computed, watch } from "vue";
import { balls_url, bunnies_url } from "@/assets/static_url"
import { useStore } from 'vuex';
import { onMounted } from 'vue';

import { pos_map_screen_to_sphere } from "@/js/sceneInteraction/math.js"

const store = useStore();

const ws = store.state.ws;





// 用于添加 placeholder
onMounted(() => {

    // canvas 画静态图
    let primiaryCanvas = document.getElementById("primiaryCanvas");
    let secondaryCanvas = document.getElementById("secondaryCanvas");


    // 获取到屏幕倒是是几倍屏。
    let getPixelRatio = function (context) {
        var backingStore = context.backingStorePixelRatio ||
            context.webkitBackingStorePixelRatio ||
            context.mozBackingStorePixelRatio ||
            context.msBackingStorePixelRatio ||
            context.oBackingStorePixelRatio ||
            context.backingStorePixelRatio || 1;
        return (window.devicePixelRatio || 1) / backingStore;
    };
    // 得到缩放倍率
    const pixelRatio = getPixelRatio(primiaryCanvas);
    // 设置canvas的真实宽高
    primiaryCanvas.width = pixelRatio * primiaryCanvas.offsetWidth;
    primiaryCanvas.height = pixelRatio * primiaryCanvas.offsetHeight;

    // 更新对应全局变量
    store.state.mainPannel_Related.primiaryCanvasWidth = primiaryCanvas.width;
    store.state.mainPannel_Related.primiaryCanvasHeight = primiaryCanvas.height;


    secondaryCanvas.width = pixelRatio * secondaryCanvas.offsetWidth;
    secondaryCanvas.height = pixelRatio * secondaryCanvas.offsetHeight;


    let p_ctx = primiaryCanvas.getContext("2d");
    let primiary_img = new Image();
    primiary_img.src = bunnies_url;
    primiary_img.onload = () => {
        p_ctx.drawImage(primiary_img, 0, 0, primiaryCanvas.width, primiaryCanvas.height);
    }

    let s_ctx = secondaryCanvas.getContext("2d");
    let secondary_img = new Image();
    secondary_img.src = balls_url;
    secondary_img.onload = () => {
        s_ctx.drawImage(secondary_img, 0, 0, secondaryCanvas.width, secondaryCanvas.height);
    }

    // 记录 时间起点（开始是demo的暂停状态）
    store.state.begin_time_mark = new Date().getTime();
    store.state.pause_time_mark = new Date().getTime();


    // 针对 canvas 的鼠标交互 startup 2023-05-09
    primiaryCanvas.addEventListener("mousemove", primiaryCanvasMouseMove, false);
    primiaryCanvas.addEventListener("mousedown", primiaryCanvasMouseDown, false);
    primiaryCanvas.addEventListener("mouseup", primiaryCanvasMouseUp, false);

})

// 鼠标移动交互事件
const primiaryCanvasMouseMove = function (e) {

    const primiaryCanvasHeight = document.getElementById("primiaryCanvas").height;
    const primiaryCanvasWidth = document.getElementById("primiaryCanvas").width;
    store.state.mainPannel_Related.mouseX = e.pageX / primiaryCanvasWidth - 0.5;
    store.state.mainPannel_Related.mouseY = 0.5 - e.pageY / primiaryCanvasHeight;

    let ret_sphere_angle = pos_map_screen_to_sphere([store.state.mainPannel_Related.mouseX, store.state.mainPannel_Related.mouseY], store.state.mainPannel_Related.camera);

    store.state.mainPannel_Related.theta = ret_sphere_angle.theta;
    store.state.mainPannel_Related.phi = ret_sphere_angle.phi;

    // 如果当前鼠标处于按下状态，则更新 deltaTheta 和 deltaPhi 属性
    if (store.state.mainPannel_Related.mouseDownMark) {
        store.state.mainPannel_Related.deltaTheta = store.state.mainPannel_Related.theta - store.state.mainPannel_Related.locTheta;
        store.state.mainPannel_Related.deltaPhi = store.state.mainPannel_Related.phi - store.state.mainPannel_Related.locPhi;

        // 向后台发送信息以支持交互
        const trackballControl_json_data_pack = {
            cmd: "trackball_control",
            deltaTheta: store.state.mainPannel_Related.deltaTheta,
            deltaPhi: store.state.mainPannel_Related.deltaPhi,
        };

        ws.send(JSON.stringify(trackballControl_json_data_pack));


        store.state.mainPannel_Related.locTheta = store.state.mainPannel_Related.theta;
        store.state.mainPannel_Related.locPhi = store.state.mainPannel_Related.phi;
    }


}



const primiaryCanvasMouseDown = function (e) {
    store.state.mainPannel_Related.mouseDownMark = true;
    store.state.mainPannel_Related.locTheta = store.state.mainPannel_Related.theta;
    store.state.mainPannel_Related.locPhi = store.state.mainPannel_Related.phi;
}



const primiaryCanvasMouseUp = function (e) {
    store.state.mainPannel_Related.mouseDownMark = false;
    // 将 deltaTheta 和 deltaPhi 清零
    store.state.mainPannel_Related.deltaTheta = 0;
    store.state.mainPannel_Related.deltaPhi = 0;

    // 向后台发送信息以支持交互
    const trackballControl_json_data_pack = {
        cmd: "trackball_control",
        deltaTheta: store.state.mainPannel_Related.deltaTheta,
        deltaPhi: store.state.mainPannel_Related.deltaPhi,
    };

    ws.send(JSON.stringify(trackballControl_json_data_pack));
}




// 在demo暂停状态下，可以根据已经缓存的信息序列进行查看
watch(() => store.state.footerPannel_Related.currentFrame, () => {
    // console.log("current hover frame = ", store.state.footerPannel_Related.currentFrame);
    // 对显示进行更新

    const frameIdx = store.state.footerPannel_Related.currentFrame;
    // 将显示的图像进行替换
    const frame_url_str = store.state.mainPannel_Related.frame_buffer_cache[frameIdx].buf;
    const frame_img = new Image();
    frame_img.src = frame_url_str;


    // 将显示的图像进行替换
    const depth_url_str = store.state.mainPannel_Related.depth_buffer_cache[frameIdx].buf;
    const depth_img = new Image();
    depth_img.src = depth_url_str;


    // depth buffer 刷新显示
    depth_img.onload = () => {
        let canvas = document.getElementById("secondaryCanvas");

        // 获取到屏幕倒是是几倍屏。
        let getPixelRatio = function (context) {
            var backingStore = context.backingStorePixelRatio ||
                context.webkitBackingStorePixelRatio ||
                context.mozBackingStorePixelRatio ||
                context.msBackingStorePixelRatio ||
                context.oBackingStorePixelRatio ||
                context.backingStorePixelRatio || 1;
            return (window.devicePixelRatio || 1) / backingStore;
        };
        // 得到缩放倍率
        const pixelRatio = getPixelRatio(canvas);
        // 设置canvas的真实宽高
        canvas.width = pixelRatio * canvas.offsetWidth; // 想当于 2 * 375 = 750 
        canvas.height = pixelRatio * canvas.offsetHeight;

        let ctx = canvas.getContext("2d");
        ctx.drawImage(depth_img, 0, 0, canvas.width, canvas.height);
    }

    // frame_buffer 刷新显示
    frame_img.onload = () => {
        let canvas = document.getElementById("primiaryCanvas");

        // 获取到屏幕倒是是几倍屏。
        let getPixelRatio = function (context) {
            let backingStore = context.backingStorePixelRatio ||
                context.webkitBackingStorePixelRatio ||
                context.mozBackingStorePixelRatio ||
                context.msBackingStorePixelRatio ||
                context.oBackingStorePixelRatio ||
                context.backingStorePixelRatio || 1;
            return (window.devicePixelRatio || 1) / backingStore;
        };
        // 得到缩放倍率
        const pixelRatio = getPixelRatio(canvas);
        // 设置canvas的真实宽高
        canvas.width = pixelRatio * canvas.offsetWidth;
        canvas.height = pixelRatio * canvas.offsetHeight;

        let ctx = canvas.getContext("2d");
        ctx.drawImage(frame_img, 0, 0, canvas.width, canvas.height);

    }

})

ws.onmessage = function (evt) {

    // self_encode_and_decode_test();

    let timer1 = new Date();
    const time1 = timer1.getMinutes() * 60000 + timer1.getSeconds() * 1000 + timer1.getMilliseconds();
    const json_data_pack = JSON.parse(evt.data);
    // console.log("json_data_pack from server = ", json_data_pack.frame_url);
    // console.log("json_data_pack from server = ", json_data_pack.depth_url);

    let timer2 = new Date();
    const time2 = timer2.getMinutes() * 60000 + timer2.getSeconds() * 1000 + timer2.getMilliseconds();
    // 数据包解码用时
    const decode_time_cost = time2 - time1;



    // 将显示的图像进行替换
    const frame_url_str = "data:image/jpg;base64," + json_data_pack.frame_url;
    // console.log("new = ", frame_url_str);
    const frame_img = new Image();
    frame_img.src = frame_url_str;


    // 将显示的图像进行替换
    const depth_url_str = "data:image/jpg;base64," + json_data_pack.depth_url;
    const depth_img = new Image();
    depth_img.src = depth_url_str;


    // depth buffer 刷新显示
    depth_img.onload = () => {
        let canvas = document.getElementById("secondaryCanvas");

        // 获取到屏幕倒是是几倍屏。
        let getPixelRatio = function (context) {
            var backingStore = context.backingStorePixelRatio ||
                context.webkitBackingStorePixelRatio ||
                context.mozBackingStorePixelRatio ||
                context.msBackingStorePixelRatio ||
                context.oBackingStorePixelRatio ||
                context.backingStorePixelRatio || 1;
            return (window.devicePixelRatio || 1) / backingStore;
        };
        // 得到缩放倍率
        const pixelRatio = getPixelRatio(canvas);
        // 设置canvas的真实宽高
        canvas.width = pixelRatio * canvas.offsetWidth; // 想当于 2 * 375 = 750 
        canvas.height = pixelRatio * canvas.offsetHeight;

        let ctx = canvas.getContext("2d");
        ctx.drawImage(depth_img, 0, 0, canvas.width, canvas.height);
    }

    // frame_buffer 刷新显示
    frame_img.onload = () => {
        let canvas = document.getElementById("primiaryCanvas");

        // 获取到屏幕倒是是几倍屏。
        let getPixelRatio = function (context) {
            let backingStore = context.backingStorePixelRatio ||
                context.webkitBackingStorePixelRatio ||
                context.mozBackingStorePixelRatio ||
                context.msBackingStorePixelRatio ||
                context.oBackingStorePixelRatio ||
                context.backingStorePixelRatio || 1;
            return (window.devicePixelRatio || 1) / backingStore;
        };
        // 得到缩放倍率
        const pixelRatio = getPixelRatio(canvas);
        // 设置canvas的真实宽高
        canvas.width = pixelRatio * canvas.offsetWidth;
        canvas.height = pixelRatio * canvas.offsetHeight;

        let ctx = canvas.getContext("2d");
        ctx.drawImage(frame_img, 0, 0, canvas.width, canvas.height);


        const last_time = store.state.total_running_time - store.state.total_pause_time;
        let myDate = new Date();
        store.state.get_time = myDate.getMinutes() * 60000 + myDate.getSeconds() * 1000 + myDate.getMilliseconds();

        // store.state.total_time_cost += (store.state.get_time - last_time);
        // store.state.total_time_cost += 100;

        // console.log("time cost = ", last_time);


        // 无论如何都会更新当前时耗
        const current_time = new Date().getTime();
        const commit_update_time_str = "update_running_time";
        store.commit(commit_update_time_str, current_time);

        // 当当前状态为 pause 但仍能接收来自后台发来的数据时（这是一种比较常见的延迟现象，并非bug）
        // 有同样需要更新 pause 时耗状态
        if (store.state.controlBar_Related.pause == true) {
            // 则需要更新 pause mark 作为新的计时起点
            store.state.pause_time_mark = current_time;
        }

        // 在这里刷新要展示的可视化数据
        const update_pack = {
            rCost: json_data_pack.rCost,
            eCost: json_data_pack.eCost,
            dCost: decode_time_cost,
            tCost: store.state.total_running_time - store.state.total_pause_time - last_time,
            runningTime: store.state.total_running_time - store.state.total_pause_time
        }
        const footer_pannel_commit_str = "footerPannel_Related/updateTimeCostArr";
        store.commit(footer_pannel_commit_str, update_pack);
        const sider_pannel_commit_str = "siderPannel_Related/updateTimeCostArr";
        store.commit(sider_pannel_commit_str, update_pack);

        // 在这里更新 图片缓存
        const update_buffer_cache_pack = {
            time: store.state.total_running_time - store.state.total_pause_time,
            frame_buffer: frame_url_str,
            depth_buffer: depth_url_str,
            time_range: store.state.footerPannel_Related.axisRange
        };

        const update_buffer_cache_commit_str = "mainPannel_Related/updateBuffers";
        store.commit(update_buffer_cache_commit_str, update_buffer_cache_pack);

    }



}





</script>
 
<style scoped>
.main-pannel-container {

    width: 100%;
    height: 100%;

    /* border: 10px pink solid; */
    box-sizing: border-box;

    display: flex;
    flex-direction: row;

    justify-content: center;
    align-items: center;

    gap: 10vw;
}

.img-container {

    width: 100%;
    height: 100%;

    /* border: 3px red solid; */
    box-sizing: border-box;
}


#primiaryCanvas {
    position: absolute;
    top: 0%;
    left: 0%;
    width: 100%;
    height: 100%;

    /* border: 3px solid yellowgreen; */
    box-sizing: border-box;
}

#secondaryCanvas {
    position: absolute;

    top: 0%;
    left: 0%;

    width: 20%;
    height: 20%;
}

.basic_img {
    width: 640px;
    height: 360px;
    /* height: 100px; */
}

.canvas-aux-pannel {
    position: absolute;
    top: 20%;
    left: 0%;
    width: 20%;
    height: 40%;

    background-color: rgba(0, 0, 0, 0.25);
    border: aquamarine solid 3px;
    box-sizing: border-box;

    color: aliceblue;

    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: center;

    padding: 5px;
    gap: 10px;
}

.mouse-pos-bar {
    width: 100%;

    border: gold solid 2px;
    box-sizing: border-box;

    display: flex;
    flex-direction: row;
    justify-content: space-around;
    align-items: center;

    gap: 1vw;
}

.sphere-angle-bar {

    width: 100%;

    border: gold solid 2px;
    box-sizing: border-box;

    display: flex;
    flex-direction: row;
    justify-content: space-around;
    align-items: center;

    gap: 1vw;
}
</style>