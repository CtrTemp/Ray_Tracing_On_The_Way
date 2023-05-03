<template>
  <div class="root-container">

    <MainPannel />

    <FooterPannel />


  </div>
</template>

<script setup>

import MainPannel from "@/components/MainPannel.vue"
import FooterPannel from "@/components/FooterPannel.vue"

// import { proto } from "./proto/test_pack"

// const proto = require("./proto/test_pack");

// console.log("proto = ", proto);

// import * as proto from "@/proto/message_pb"




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
