<template>
    <div class="main-pannel-container">
        <div class="img-container">
            <canvas id="primiaryCanvas"></canvas>
            <canvas id="secondaryCanvas"></canvas>
            <!-- <img :src=balls_url alt="" class="basic_img img_balls"> -->
            <!-- <img :src=bunnies_url alt="" class="basic_img img_bunnies"> -->
        </div>
    </div>
</template>
 
<script setup>
import { reactive, ref } from "vue";
import { computed, watch } from "vue";
import { balls_url, bunnies_url } from "@/assets/static_url"
import { useStore } from 'vuex';
import { onMounted } from 'vue';


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

})



// // 发送消息停掉后台
// const ws_client_close_server = function () {

//     const json_pack = {
//         cmd: "close"
//     }
//     ws.send(JSON.stringify(json_pack));
//     let myDate = new Date();
//     store.state.request_time = myDate.getSeconds() * 1000 + myDate.getMilliseconds();
// }


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
</style>