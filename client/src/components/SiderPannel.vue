<template>
    <div class="sider-pannel-container">
        <div class="donut-container"></div>
        <SiderDount :donutDeps="renderCostData" :classIdx="1" />
        <SiderDount :donutDeps="otherCostData" :classIdx="2" />
    </div>
</template>
 
<script setup>

import SiderDount from "@/components/SiderPannel/SiderDonut.vue"
import { reactive, ref } from "vue";
import { computed, watch } from "vue";
import { useStore } from "vuex";

const store = useStore();

const renderCostData = store.state.siderPannel_Related.renderCost;
const otherCostData = store.state.siderPannel_Related.otherCost;

</script>
 
<style scoped>
.sider-pannel-container {
    position: absolute;

    top: 0%;
    right: 0%;
    width: 20%;
    height: 75%;

    /* border: red solid 3px; */
    box-sizing: border-box;

    display: flex;
    flex-direction: column;

    /* gap: 5px; */

    background-color: rgba(0, 0, 0, 0.25);
}
</style>