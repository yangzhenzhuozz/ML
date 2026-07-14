import DefaultTheme from 'vitepress/theme';
import 'katex/dist/katex.min.css';
import FunctionEcharts from './components/FunctionEcharts.vue';
/** @type {import('vitepress').Theme} */
export default {
    extends: DefaultTheme,
    enhanceApp({ app }) {
        // 注册自定义全局组件
        app.component('FunctionEcharts', FunctionEcharts);
    },
};
