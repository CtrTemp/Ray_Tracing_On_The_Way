<template>
    <div class="control-bar-container">
        <el-button @click="flip_pause_begin_loop('pause')" type="primary" :icon="VideoPause" />
        <el-button @click="flip_pause_begin_loop('begin')" type="primary" :icon="VideoPlay" />
        <!-- <div>{{ total_running_time }}</div> -->
        <!-- <div>{{ total_pause_time }}</div> -->
    </div>
</template>
 
<script setup>
import { reactive, ref } from "vue";
import { computed, watch } from "vue";
import { useStore } from "vuex";
import { ElButton } from "element-plus";
import { VideoPlay, VideoPause } from '@element-plus/icons-vue'

const store = useStore();

const ws = store.state.ws;

const total_running_time = computed(() => {
    return store.state.total_running_time;
})
const total_pause_time = computed(() => {
    return store.state.total_pause_time;
})

const flip_pause_begin_loop = function (pause_begin) {

    // 无论如何都会更新当前时耗
    const current_time = new Date().getTime();
    const commit_update_time_str = "update_running_time";
    store.commit(commit_update_time_str, current_time);

    if (pause_begin == "pause") {
        // 当前状态为 demo正在运行状态 执行pause
        if (store.state.controlBar_Related.pause == false) {
            // 则需要更新 pause mark 作为新的计时起点
            store.state.pause_time_mark = current_time;
        }
        // 当前状态为 demo停止状态 执行pause
        else {
            // 则更新当前pause的持续总时长即可
            store.commit("update_pause_time", current_time);
            // 之后需要更新 pause mark 作为新的计时起点
            store.state.pause_time_mark = current_time;
        }
    }
    else {
        // 当前状态为 demo正在运行状态 执行begin
        if (store.state.controlBar_Related.pause == false) {
            // do nothing
        }
        // 当前状态为 demo停止状态 执行begin
        else {
            // 则更新当前pause的持续总时长即可
            store.commit("update_pause_time", current_time);
        }
    }


    const json_pack = {
        cmd: "pause_begin",
        value: pause_begin // 要的是反转后的值
    };
    ws.send(JSON.stringify(json_pack));

    // 更改全局变量
    const commit_str = "controlBar_Related/flip_pause_begin";
    store.commit(commit_str, pause_begin);


}


</script>
 
<style scoped>
.control-bar-container {
    position: absolute;

    top: 0%;
    left: 40%;
    width: 20%;
    /* height: 10%; */

    /* border: red solid 3px; */
    box-sizing: border-box;
    background-color: rgba(0, 0, 0, 0.25);
    color: white;


    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;

    padding-left: 10px;
    padding-right: 10px;

    gap: 10px;
}
</style>