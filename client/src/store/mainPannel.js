

export default {

    namespaced: true,

    actions: {},
    mutations: {
        updateBuffers(state, update_pack) {

            const frame_update_unit = { time: update_pack.time, buf: update_pack.frame_buffer };
            const depth_update_unit = { time: update_pack.time, buf: update_pack.depth_buffer };

            state.frame_buffer_cache.push(frame_update_unit);
            state.depth_buffer_cache.push(depth_update_unit);

            // cache 中保留信息
            state.frame_buffer_cache = state.frame_buffer_cache.filter((d) => {
                return d.time >= update_pack.time_range[0] && d.time <= update_pack.time_range[1]
            });
            state.depth_buffer_cache = state.depth_buffer_cache.filter((d) => {
                return d.time >= update_pack.time_range[0] && d.time <= update_pack.time_range[1]
            });

            // console.log("current buffer size = ", state.frame_buffer_cache.length);
        }
    },
    state() {
        return {
            frame_buffer_cache: [],
            depth_buffer_cache: [],


            // 以下是鼠标交互所依赖的相机参数
            camera: {
                fov: 40,            // 相机视角
                aspect: 16 / 9,     // 屏幕长宽比
                focus_dist: 10.0,   // 屏幕与视点距离
            },
            // 以下是鼠标交互所用参数
            mouseX: 0,  // 屏幕X坐标
            mouseY: 0,  // 屏幕Y坐标
            unitSphereX: 0,  // 单位球上X坐标
            unitSphereY: 0,  // 单位球上Y坐标
            pitch: 0,
            roll: 0,
            yaw: 0,
            theta: 0,  // 极角
            phi: 0,    // 方位角
            deltaTheta: 0,
            deltaPhi: 0,
            locTheta: 0,
            locPhi: 0,
            mouseDownMark: false,
        }
    },
    getters: {}
}


