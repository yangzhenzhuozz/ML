import DefaultTheme from 'vitepress/theme';
import 'katex/dist/katex.min.css';
import BiFunctionEcharts from './components/BiFunctionEcharts.vue';
import LeastSquaresMethod from './components/LeastSquaresMethod.vue';
import GradientDescent from './components/GradientDescent.vue';

/** @type {import('vitepress').Theme} */
export default {
    extends: DefaultTheme,
    enhanceApp({ app }) {
        // 注册自定义全局组件
        app.component('BiFunctionEcharts', BiFunctionEcharts);
        app.component('LeastSquaresMethod', LeastSquaresMethod);
        app.component('GradientDescent', GradientDescent);
    },
};
