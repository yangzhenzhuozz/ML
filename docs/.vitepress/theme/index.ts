import DefaultTheme from 'vitepress/theme';
//@ts-ignore
import 'katex/dist/katex.min.css';
import BiFunctionEcharts from './components/BiFunctionEcharts.vue';
import LeastSquaresMethod from './components/LeastSquaresMethod.vue';
import GradientDescent from './components/GradientDescent.vue';
import FunctionChart from './components/FunctionChart.vue';
import LogisticRegressionSigmoid from './components/LogisticRegressionSigmoid.vue';
import GraphDemo from './components/GraphDemo.vue';
import GPUMatrix from './components/GPUMatrix.vue';
import GPUMlp from './components/GPUMlp.vue';
import GPUMlpBatch from './components/GPUMlpBatch.vue';
/** @type {import('vitepress').Theme} */
export default {
    extends: DefaultTheme,
    //@ts-ignore
    enhanceApp({ app }) {
        // 注册自定义全局组件
        app.component('BiFunctionEcharts', BiFunctionEcharts);
        app.component('LeastSquaresMethod', LeastSquaresMethod);
        app.component('GradientDescent', GradientDescent);
        app.component('FunctionChart', FunctionChart);
        app.component('LogisticRegressionSigmoid', LogisticRegressionSigmoid);
        app.component('GraphDemo', GraphDemo);
        app.component('GPUMatrix', GPUMatrix);
        app.component('GPUMlp', GPUMlp);
        app.component('GPUMlpBatch', GPUMlpBatch);
    },
};
