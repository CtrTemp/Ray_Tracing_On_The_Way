<template>
    <div class="menu-item-container">{{ dataDeps.name }}
        <transition class="secondary-options-animation" enter-active-class="animate__animated animate__slideInLeft"
            leave-active-class="animate__animated animate__slideOutLeft">
            <div v-if="dataDeps.showable" class="secondary-option-container">
                <div @mouseover="show_secondary_tips(index)" @mouseleave="hide_secondary_tips(index)"
                    class="secondary-option-item" v-for="(item, index) in dataDeps.children" :key="index">
                    <!-- {{ item.name }} -->
                    <img class="secondary-option-img" :src="item.icon" alt="">
                </div>
                <div class="option-description-tips">{{ show_tips }}</div>
            </div>
        </transition>
    </div>
</template>
 
<script setup>
import { reactive, ref } from "vue";
import { computed, watch } from "vue";
import { useStore } from "vuex";

const store = useStore();
const deps = defineProps(["dataDeps", "primiary_index"]);

const show_secondary_tips = function (index) {
    const commit_str = "siderMenu_Related/show_secondary_tips";
    const index_pack = {
        primiary_index: deps.primiary_index,
        secondary_index: index
    }
    store.commit(commit_str, index_pack);
}

const hide_secondary_tips = function (index) {
    const commit_str = "siderMenu_Related/hide_secondary_tips";
    const index_pack = {
        primiary_index: deps.primiary_index,
        secondary_index: index
    }
    store.commit(commit_str, index_pack);
}


const icon = `url(${deps.dataDeps.icon})`;


const show_tips = computed(() => {
    for (let i = 0; i < store.state.siderMenu_Related.data_temp[deps.primiary_index].children.length; i++) {
        const unit = store.state.siderMenu_Related.data_temp[deps.primiary_index].children[i];
        if (unit.showable) {
            return unit.tips;
        }
    }
    return deps.dataDeps.show_tips;
})



</script>
 
<style scoped>
.menu-item-container {
    position: relative;
    width: 90%;
    /* 设置同等宽高 */
    padding: 40% 0 40% 0;

    cursor: pointer;

    background-color: rgba(0, 0, 0, 0.5);
    background-image: v-bind(icon);
    background-size: 100% 100%;
    color: white;
    border-radius: 0.5vh;
    /* border: red solid 3px; */
    box-sizing: border-box;


    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;

}

.menu-item-container:hover {
    box-shadow: inset 0 0 10px rgba(255, 255, 255, 0.8);
}

.secondary-option-container {

    position: absolute;
    left: 100%;
    width: 1000%;
    height: 100%;
    /* border: yellowgreen solid 3px; */
    /* background-color: rgba(0, 0, 0, 0.5); */
    box-sizing: border-box;

    display: flex;
    flex-direction: row;
    justify-content: flex-start;
    align-items: center;

    gap: 10px;

    padding-left: 50%;

    padding-top: 5%;
    padding-bottom: 5%;

    z-index: 1;
}

.secondary-option-item {
    position: relative;
    height: 100%;
    width: 9%;

    cursor: pointer;

    background-color: rgba(0, 0, 0, 0.5);
    color: white;
    border-radius: 0.5vh;

    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;

}

.secondary-option-item:hover {
    box-shadow: inset 0 0 10px rgba(255, 255, 255, 0.8);
    ;
}

.secondary-options-animation {
    --animate-duration: 250ms;
}


.option-description-tips {
    position: relative;
    height: 100%;
    width: 25%;

    cursor: pointer;

    background-color: rgba(0, 0, 0, 0.5);
    color: white;
    border-radius: 0.5vh;

    display: flex;
    flex-direction: row;
    justify-content: center;
    align-items: center;

    font-size: 1.5vh;
}

.secondary-option-img {
    width: 100%;
    height: 100%;
}
</style>