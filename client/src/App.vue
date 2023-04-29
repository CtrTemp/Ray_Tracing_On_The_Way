<template>
  <div class="root-container">
    <div class="img-container">
      <img :src=balls_url alt="" class="basic_img img_balls">
      <!-- <img :src=bunnies_url alt="" class="basic_img img_bunnies"> -->
    </div>

  </div>
</template>

<script setup>

import { useStore } from 'vuex';
import { balls_url, bunnies_url } from "@/assets/static_url"

// import { proto } from "./proto/test_pack"

// const proto = require("./proto/test_pack");

// console.log("proto = ", proto);

import * as proto from "@/proto/message_pb"

const store = useStore();


const ws = store.state.ws;


// 发送消息停掉后台
const ws_client_close_server = function () {

  const json_pack = {
    cmd: "close"
  }
  ws.send(JSON.stringify(json_pack));
  let myDate = new Date();
  store.state.request_time = myDate.getSeconds() * 1000 + myDate.getMilliseconds();
}


const ws_request_protobuf_data_pack = function () {
  const json_pack = {
    cmd: "get_protobuf_pack"
  }

  ws.send(JSON.stringify(json_pack));
  let myDate = new Date();
  store.state.request_time = myDate.getSeconds() * 1000 + myDate.getMilliseconds();
}

const ws_request_protobuf_img_pack = function () {
  const json_pack = {
    cmd: "get_image_proto_pack"
  }

  ws.send(JSON.stringify(json_pack));
  let myDate = new Date();
  store.state.request_time = myDate.getSeconds() * 1000 + myDate.getMilliseconds();
}


// 这里再尝试以下 json 传 url 的速度

const ws_client_get_frame = function () {

  const json_pack = {
    cmd: "get_frame_pack"
  }

  ws.send(JSON.stringify(json_pack));
  let myDate = new Date();
  store.state.request_time = myDate.getSeconds() * 1000 + myDate.getMilliseconds();

}


ws.onmessage = function (evt) {

  // self_encode_and_decode_test();


  const json_data_pack = JSON.parse(evt.data);
  // console.log("json_data_pack from server = ", json_data_pack);

  // 将显示的图像进行替换
  const url_str = "data:image/jpg;base64," + json_data_pack.url;
  document.getElementsByClassName("img_balls")[0].setAttribute("src", url_str);


  const last_time = store.state.get_time;
  let myDate = new Date();
  store.state.get_time = myDate.getSeconds() * 1000 + myDate.getMilliseconds();

  console.log("time cost between two frame = ", store.state.get_time - last_time);

  // 这里好像收到的是一个空的blob





  // let reader = new FileReader();
  // reader.readAsArrayBuffer(evt.data);



  // reader.onload = function (e) {

  //   // 首先将读取到的数据转换成 uint8Arr
  //   var uint8_buf = new Uint8Array(reader.result);
  //   // // 发现这里解析出来的uint8arr和编码时得到的不一样，使用string发送就一样了
  //   // console.log("buf = ", uint8_buf);



  //   // 下一步就是送入解析器进行解析
  //   let parsed_pack = proto.test_pack.deserializeBinary(uint8_buf);


  //   // let date_record_stop = ownDate.getSeconds() * 1000 + ownDate.getMilliseconds();
  //   // console.log("parse time cost = ", date_record_stop - date_record_start);
  //   console.log("parsed_pack = ", parsed_pack);
  //   // console.log("parsed_pack_img_url = ", parsed_pack.getImgUrl());
  //   let myDate = new Date();
  //   store.state.get_time = myDate.getSeconds() * 1000 + myDate.getMilliseconds();

  //   // 将显示的图像进行替换
  //   const url_str = "data:image/jpeg;base64," + parsed_pack.getImgUrl();
  //   document.getElementsByClassName("img_balls")[0].setAttribute("src", url_str);

  // }


}

const self_encode_and_decode_test = function () {

  const proto_test_pack = proto.test_pack;

  const pack_instance = new proto_test_pack();

  // console.log("\n\n\n\n\n");

  pack_instance.setCmd("proto_frame_pack");
  pack_instance.setWidth(4);
  pack_instance.setHeight(2);
  pack_instance.addBuffer(0);
  pack_instance.addBuffer(1);
  pack_instance.addBuffer(2);
  pack_instance.addBuffer(3);
  pack_instance.addBuffer(4);
  pack_instance.addBuffer(5);
  pack_instance.addBuffer(6);
  pack_instance.addBuffer(7);
  // console.log("pack_instance = ", pack_instance);

  const pack_binary_uint8Arr = pack_instance.serializeBinary();

  console.log("pack_binary_str = ", pack_binary_uint8Arr);

  let self_parsed_pack = proto.test_pack.deserializeBinary(pack_binary_uint8Arr);
  // console.log("decoded pack = ", self_parsed_pack);

}




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

  top: 1%;
  left: 1%;
  width: 98%;
  height: 98%;

  /* border: 10px gold solid; */

  display: flex;
  flex-direction: column;

  justify-content: center;
  align-items: center;

  gap: 10vh;
}


.img-container {

  width: 95%;
  height: 95%;
  /* border: 10px pink solid; */

  display: flex;
  flex-direction: row;

  justify-content: center;

  gap: 10vw;

}

.basic_img {
  width: 90%;
  /* height: 100px; */
}
</style>
