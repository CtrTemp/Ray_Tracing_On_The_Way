<template>
    <div class="sider-pannel-container">

        <transition class="thumbnail-animation" enter-active-class="animate__animated animate__slideInRight"
            leave-active-class="animate__animated animate__slideOutRight">

            <div class="sider-pannel" v-show="(show_flip) % 2">


                <SiderDount :donutDeps="renderCostData" :classIdx="1" />
                <SiderDount :donutDeps="otherCostData" :classIdx="2" />
                <SiderBarChart />

                <div @click="show_flip++" class="menu-hide">
                    <div>•</div>
                    <div>•</div>
                    <div>•</div>
                </div>

            </div>
        </transition>



        <transition class="thumbnail-animation animate__delay-1s" enter-active-class="animate__animated ">
            <div @click="show_flip++" v-show="(show_flip + 1) % 2" class="menu-show animate__fadeIn">
                <div>•</div>
                <div>•</div>
                <div>•</div>
            </div>
        </transition>


    </div>
</template>
 
<script setup>

import SiderDount from "@/components/SiderPannel/SiderDonut.vue"
import SiderBarChart from "@/components/SiderPannel/SiderBarChart.vue";


import { reactive, ref } from "vue";
import { computed, watch } from "vue";
import { useStore } from "vuex";

const store = useStore();

const renderCostData = store.state.siderPannel_Related.renderCost;
const otherCostData = store.state.siderPannel_Related.otherCost;


let show_flip = ref(1);

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

    /* background-color: rgba(0, 0, 0, 0.25); */

}


.sider-pannel {
    position: absolute;

    top: 0%;
    right: 0%;
    width: 100%;
    height: 100%;


    /* border: yellowgreen solid 3px; */
    box-sizing: border-box;

    background-color: rgba(0, 0, 0, 0.25);
    display: flex;
    flex-direction: column;

    padding-left: 1vh;
}


.menu-show {

    position: absolute;

    cursor: pointer;

    top: 40%;
    right: 0%;
    width: 5%;
    height: 20%;

    /* border: 2px steelblue solid; */
    box-sizing: border-box;
    border-radius: 3px;

    background-color: rgba(0, 0, 0, 0.5);

    font-size: 1.5vh;
    color: aliceblue;

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;

}



.menu-hide {

    position: absolute;

    cursor: pointer;


    top: 40%;
    left: 0%;
    width: 5%;
    height: 20%;

    /* border: 2px steelblue solid; */
    box-sizing: border-box;
    border-radius: 3px;

    background-color: rgba(0, 0, 0, 0.25);

    font-size: 1.5vh;
    color: aliceblue;

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;

}


.menu-animation {
    --animate-duration: 500ms;
}

.thumbnail-animation {
    --animate-duration: 500ms;
}
</style>