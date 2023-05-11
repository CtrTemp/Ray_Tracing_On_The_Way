<template>
    <div class="sider-menu-container">
        <!-- <div v-show="(show_flip) % 2" class="menu animate__animated animate__slideInLeft">{{ show_flip }}</div> -->

        <transition class="menu-animation" enter-active-class="animate__animated animate__slideInLeft"
            leave-active-class="animate__animated animate__slideOutLeft">
            <div v-show="(show_flip) % 2" class="menu">
                <MenuItem @mouseover="flip_to_show(index)" @mouseleave="flip_to_hide(index)" v-for="(item, index) in data"
                    :key="index" :dataDeps="item" :primiary_index="index" />
                <div @click="show_flip++" class="menu-hide">
                    <!-- 这个符号需要复制一下 -->
                    <div>•</div>
                    <div>•</div>
                    <div>•</div>
                </div>
            </div>
        </transition>

        <transition class="thumbnail-animation animate__delay-1s" enter-active-class="animate__animated ">
            <div @click="show_flip++" v-show="(show_flip + 1) % 2" class="menu-show animate__fadeIn">
                <div v-if="false" class="menu-show-icon"></div>
                <!-- <div>▶</div>
                <div>▶</div>
                <div>▶</div> -->
                <div>•</div>
                <div>•</div>
                <div>•</div>
            </div>
        </transition>

        <!-- <div @click="show_flip++" v-show="(show_flip) % 2" class="hover-bar animate__animated animate__slideInLeft">H</div>
        <div @click="show_flip++" v-show="(show_flip + 1) % 2" class="hover-bar animate__animated ">M</div> -->
    </div>
</template>

<script setup>
import { reactive, ref } from "vue";
import { computed, watch } from "vue";
import { useStore } from "vuex";

import MenuItem from "@/components/RenderOptionMenu/MenuItem.vue"

const store = useStore();

let show_flip = ref(0);

const data = store.state.siderMenu_Related.data_temp;

const flip_to_show = function (index) {
    const commit_str = "siderMenu_Related/flip_to_show";
    store.commit(commit_str, index);
}

const flip_to_hide = function (index) {
    const commit_str = "siderMenu_Related/flip_to_hide";
    store.commit(commit_str, index);
}

// watch(() => store.state.siderMenu_Related.data_temp, () => {
//     console.log(store.state.siderMenu_Related.data_temp);
// }, { deep: true });

</script>

<style scoped>
.sider-menu-container {
    position: absolute;
    top: 25%;
    left: 0%;
    width: 6%;
    height: 40%;

    /* border: gold 5px solid; */
    box-sizing: border-box;

    /* background-color: whitesmoke; */

    display: flex;
    flex-direction: row;

    justify-content: flex-start;
    align-items: center;

}


.menu-show {
    position: relative;

    cursor: pointer;
    width: 10%;
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

.menu-show-icon {
    position: absolute;

    top: 0%;
    left: 0%;
    height: 100%;
    width: 100%;
    background-image: url(@/assets/icons/sample_icon.svg);
    background-size: 100% 100%;
    background-repeat: no-repeat;
}

.menu {
    position: relative;
    width: 80%;
    height: 100%;

    /* background-color: rgba(0, 0, 0, 0.5); */
    border-radius: 1vh;

    display: flex;
    flex-direction: column;

    gap: 0.5vh;
    padding-left: 0.5vh;
    padding-top: 1vh;
    padding-bottom: 0.5vh;
    padding-right: 0.5vh;
}

.menu-hide {
    position: absolute;
    top: 40%;
    right: 0%;
    width: 10%;
    height: 20%;
    font-size: 1.5vh;
    background-color: rgba(0, 0, 0, 0.35);
    color: white;
    border-radius: 4px;

    cursor: pointer;

    z-index: 0;

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