<template>
    <div class="main-pannel-container">
        <div class="img-container">
            <canvas id="myCanvas"></canvas>
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
    let canvas = document.getElementById("myCanvas");
    // let canvas_root_container = document.getElementsByClassName("img-container")[0];
    // console.log("canvas_root_container size = ", canvas_root_container.clientHeight)
    // canvas.width = "1000";
    // canvas.height = "500";


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


    // console.log("canvas = ", canvas);
    let ctx = canvas.getContext("2d");
    let img = new Image();
    img.src = balls_url;
    img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    }
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
    // console.log("json_data_pack from server = ", json_data_pack);

    let timer2 = new Date();
    const time2 = timer2.getMinutes() * 60000 + timer2.getSeconds() * 1000 + timer2.getMilliseconds();

    const decode_time_cost = time2 - time1;

    // 将显示的图像进行替换
    const url_str = "data:image/jpg;base64," + json_data_pack.url;
    // document.getElementsByClassName("img_balls")[0].setAttribute("src", url_str);
    const img = new Image();
    img.src = url_str;



    img.onload = () => {
        let canvas = document.getElementById("myCanvas");

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
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);


        const last_time = store.state.get_time;
        let myDate = new Date();
        store.state.get_time = myDate.getMinutes() * 60000 + myDate.getSeconds() * 1000 + myDate.getMilliseconds();

        store.state.total_time_cost += (store.state.get_time - last_time);

        // console.log("time cost = ", last_time);

        // 在这里刷新要展示的可视化数据
        const update_pack = {
            rCost: json_data_pack.rCost,
            eCost: json_data_pack.eCost,
            dCost: decode_time_cost,
            tCost: store.state.get_time - last_time,
            currentTime: store.state.total_time_cost
        }
        const commit_str = "footerPannel_Related/updateTimeCostArr";
        store.commit(commit_str, update_pack);

        // console.log("time cost between two frame = ", store.state.get_time - last_time);
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


#myCanvas {
    width: 100%;
    height: 100%;

    /* border: 3px solid yellowgreen; */
    box-sizing: border-box;
}

.basic_img {
    width: 640px;
    height: 360px;
    /* height: 100px; */
}
</style>