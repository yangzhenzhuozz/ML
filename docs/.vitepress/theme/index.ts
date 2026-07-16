import DefaultTheme from 'vitepress/theme';
import 'katex/dist/katex.min.css';
import BiFunctionEcharts from './components/BiFunctionEcharts.vue';
import LineRegression from './components/LineRegression.vue';

/** @type {import('vitepress').Theme} */
export default {
    extends: DefaultTheme,
    enhanceApp({ app }) {
        // 注册自定义全局组件
        app.component('BiFunctionEcharts', BiFunctionEcharts);
        app.component('LineRegression', LineRegression);
    },
};
