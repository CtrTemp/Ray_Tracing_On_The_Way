<template>
  <div class="root-container">

    <MainPannel />

    <ControlBar />

    <SiderPannel />

    <FooterPannel />

    <RenderOptionMenu />


  </div>
</template>

<script setup>

import MainPannel from "@/components/MainPannel.vue"
import FooterPannel from "@/components/FooterPannel.vue"
import SiderPannel from "@/components/SiderPannel.vue";
import ControlBar from "@/components/ControlBar.vue";
import RenderOptionMenu from "@/components/RenderOptionMenu.vue";

import { computed } from "vue";

import { provide } from "vue";

const color_map = computed(() => {
  const ret_map = {
    rCost: "yellowgreen",
    eCost: "pink",
    dCost: "steelblue",
    tCost: "white"
  }

  return ret_map;
})


provide("color_map", color_map);


</script>

<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}

.root-container {

  position: absolute;

  top: 0%;
  left: 0%;
  width: 100%;
  height: 100%;

  /* border: 10px gold solid; */
  box-sizing: border-box;

  display: flex;
  flex-direction: column;

  justify-content: center;
  align-items: center;

  gap: 10vh;

  overflow: hidden;
}
</style>
