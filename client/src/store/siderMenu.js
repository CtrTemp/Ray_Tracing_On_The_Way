export default {
    namespaced: true,

    actions: {},
    mutations: {
        flip_to_show(state, index) {
            state.data_temp[index].showable = true;
            state.data_temp[index].show_tips = state.data_temp[index].tips;
        },
        flip_to_hide(state, index) {
            state.data_temp[index].showable = false;
        },
        show_secondary_tips(state, index) {
            state.data_temp[index.primiary_index].children[index.secondary_index].showable = true;
        },
        hide_secondary_tips(state, index) {
            state.data_temp[index.primiary_index].children[index.secondary_index].showable = false;
        },
    },
    state() {
        return {
            data_temp: [
                {
                    name: "",
                    icon: require("../assets/icons/distribution_menu.svg"),
                    children: [
                        { name: "1", icon: require("../assets/icons/distribution_format_option.svg"), tips: "consistent distribution", showable: false },
                        { name: "2", icon: require("../assets/icons/distribution_random_option.svg"), tips: "random distribution", showable: false },
                    ],
                    showable: false,
                    tips: "Subpixel level ray distribution",
                    show_tips: "",
                },
                {
                    name: "",
                    icon: require("../assets/icons/max_bounce_menu.svg"),
                    children: [
                        { name: "4", icon: require("../assets/icons/bounce_option_1.svg"), tips: "Max bounce depth=1", showable: false },
                        { name: "5", icon: require("../assets/icons/bounce_option_2.svg"), tips: "Max bounce depth=2", showable: false },
                        { name: "6", icon: require("../assets/icons/bounce_option_3.svg"), tips: "Max bounce depth=3", showable: false },
                        { name: "7", icon: require("../assets/icons/bounce_option_n.svg"), tips: "Max bounce depth=n", showable: false },
                    ],
                    showable: false,
                    tips: "Max ray-object bounce depth / times",
                    show_tips: "",
                },
                {
                    name: "",
                    icon: require("../assets/icons/render_methods_menu.svg"),
                    children: [
                        { name: "8", icon: require("../assets/icons/render_methods_whitted_option.svg"), tips: "Whitted Styled Ray-Tracing", showable: false },
                        { name: "9", icon: require("../assets/icons/render_methods_BXDF_option.svg"), tips: "BXDF based on Rendering Equation", showable: false },
                    ],
                    showable: false,
                    tips: "Main Render Methods",
                    show_tips: "",
                },
                {
                    name: "",
                    icon: require("../assets/icons/sample_methods_menu.svg"),
                    children: [],
                    showable: false,
                    tips: "Screen level ray sparse sampling methods",
                    show_tips: "",
                },
                {
                    name: "",
                    icon: require("../assets/icons/denoise_methods_menu.svg"),
                    children: [],
                    showable: false,
                    tips: "Image denoising methods(digital image processing)",
                    show_tips: "",
                },

            ]
        }
    },
    getters: {

    }
}


