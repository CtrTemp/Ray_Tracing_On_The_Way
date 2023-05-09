
/**
 *  给定屏幕上的某个坐标，计算其在单位球上的投影坐标
 * 实际上就是从当前视点向屏幕位置坐标处发射的射线与以视点为球心的单位球的交点
 *  假设我们在相机的相对坐标系中，相机朝向永远是x轴正向
 *  假设屏幕正中为原点（0,0）
 *  screenPos 以比例传入
 */
const pos_map_screen_to_sphere = function (screenPos, camera) {

    // 由于是单位球，这里就省略半径参数，默认为1
    const ret_sphere_pos = {
        theta: 0,   // 极角
        phi: 0      // 方位角
    }
    let half_angle = camera.fov * Math.PI / 180 / 2;

    // console.log("half_angle = ", half_angle);

    let screen_half_height = camera.focus_dist * Math.tan(half_angle);
    let screen_half_width = screen_half_height * camera.aspect;

    // console.log(`[${screen_half_height},${screen_half_width}]`);

    const real_pos_x = screenPos[0] * screen_half_width * 2;
    const real_pos_y = screenPos[1] * screen_half_height * 2;

    ret_sphere_pos.theta = Math.PI / 2 - Math.atan(real_pos_y / camera.focus_dist);
    ret_sphere_pos.phi = -Math.atan(real_pos_x / camera.focus_dist)

    // console.log(ret_sphere_pos);
    return ret_sphere_pos;
}




export {
    pos_map_screen_to_sphere
}
