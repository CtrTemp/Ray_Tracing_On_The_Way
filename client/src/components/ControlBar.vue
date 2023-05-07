<template>
    <div class="control-bar-container">
        <el-button @click="flip_pause_begin_loop('pause')" type="primary" :icon="VideoPause" />
        <el-button @click="flip_pause_begin_loop('begin')" type="primary" :icon="VideoPlay" />
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

const flip_pause_begin_loop = function (pause_begin) {


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
    left: 0%;
    /* width: 10%; */
    /* height: 10%; */

    /* border: red solid 3px; */
    box-sizing: border-box;
    background-color: rgba(0, 0, 0, 0.25);

    color: white;

}
</style>