<template>
    <div class="echarts-wrapper">
        <div ref="chartRef" class="chart-container"></div>
        <div
            style="
                display: flex;
                justify-content: center;
                align-items: baseline;
                gap: 10px;
            "
        >
            <math xmlns="http://www.w3.org/1998/Math/MathML">
                <semantics>
                    <mrow>
                        <mi>y</mi>
                        <mo>=</mo>
                        <mn>{{ props.weight }}</mn>
                        <mi>x</mi>
                        <mo>+</mo>
                        <mn>{{ props.bias }}</mn>
                    </mrow>
                    <annotation encoding="application/x-tex">
                        y={{ props.weight }}x+{{ props.bias }}
                    </annotation>
                </semantics>
            </math>
            <div>均方误差（MSE）：{{ mseDisplay }}</div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { CustomSeriesRenderItemParams } from 'echarts';
import { CustomSeriesRenderItemAPI } from 'echarts';
import { EChartsType } from 'echarts';
import { ECBasicOption } from 'echarts/types/dist/shared';
import { computed, onMounted, useTemplateRef, watch, type PropType } from 'vue';
const props = defineProps({
    scatter: {
        type: Array as PropType<[number, number][]>,
        required: true,
    },
    weight: {
        type: Number,
        required: true,
    },
    bias: {
        type: Number,
        required: true,
    },
    title: { type: String, default: '' }, // 标题
});

const chartRef = useTemplateRef('chartRef');
let myChart: EChartsType | null = null;
const mseDisplay = computed(() => {
    if (!props.scatter || props.scatter.length === 0) return '0';

    let sumSquaredError = 0;
    const w = props.weight;
    const b = props.bias;

    // 遍历所有离散点计算累加平方误差
    for (const [x, yActual] of props.scatter) {
        const yTheoretical = w * x + b;
        const error = yActual - yTheoretical;
        sumSquaredError += error * error;
    }

    // 计算均值
    const mse = sumSquaredError / props.scatter.length;

    // 格式化输出：保留最多2位小数，且自动干掉末尾无用的 .0
    return Number(mse.toFixed(2));
});
function createErrorLineChartOption(
    scatter: [number, number][],
    weight: number,
    bias: number,
): ECBasicOption {
    const xValues = scatter.map((p) => p[0]);
    const minX = Math.min(...xValues);
    const maxX = Math.max(...xValues);

    const xMargin = maxX - minX || 1;
    // 保持最大值有一点溢出边距，让最右侧的点不贴边
    const xAxisMax = Math.ceil(maxX + xMargin * 0.1);

    // 根据直线方程，计算直线在 X=0 和 X=xAxisMax 时的端点位置
    // 这样直线就能精准地从 Y 轴（X=0）一侧一直延伸到图表最右侧
    const lineStartPoint = [0, weight * 0 + bias];
    const lineEndPoint = [xAxisMax, weight * xAxisMax + bias];

    // 2. 返回动态生成的 ECharts Option
    return {
        title: {
            text: props.title,
            left: 'center',
            textStyle: { fontSize: 16 },
        },
        tooltip: { trigger: 'item' },
        grid: [{ left: 'left', top: 'top', width: '100%', height: '100%' }],
        xAxis: {
            type: 'value',
            min: 0, // 👈 核心修改：X 轴严格从 0 开始
            max: xAxisMax,
        },
        yAxis: {
            type: 'value',
            min: 0, // 👈 核心修改：Y 轴严格从 0 开始
            scale: false, // 关闭原先的自动缩放
        },
        series: [
            // ① 离散点系列
            {
                name: '离散点',
                type: 'scatter',
                data: scatter,
                itemStyle: {
                    color: '#ff4d4f',
                },
                tooltip: {
                    formatter: function (params: any) {
                        const x = params.data[0];
                        const y = params.data[1];
                        const yFormatted = Number(y.toFixed(3));
                        return `(${x} , ${yFormatted})`;
                    },
                },
                z: 10,
            },
            // ② 拟合直线系列
            {
                name: '拟合直线',
                type: 'line',
                data: [lineStartPoint, lineEndPoint],
                showSymbol: false,
                lineStyle: {
                    color: '#1890ff',
                    width: 2,
                },
                z: 5,
            },
            // ③ 核心：动态渲染垂直误差线
            {
                name: '误差',
                type: 'custom',
                renderItem: function (
                    _params: CustomSeriesRenderItemParams,
                    api: CustomSeriesRenderItemAPI,
                ) {
                    // 获取当前点的 X 和 实际 Y 的坐标值
                    const xValue = api.value(0) as number;
                    const yActual = api.value(1) as number;

                    // 根据传入的斜率和截距，计算理论上的 Y 轴值 (y = wx + b)
                    const yTheoretical = weight * xValue + bias;

                    // 将数值坐标转换为图表上的像素点坐标
                    const startCoord = api.coord([xValue, yActual]);
                    const endCoord = api.coord([xValue, yTheoretical]);

                    return {
                        type: 'line',
                        shape: {
                            x1: startCoord[0],
                            y1: startCoord[1],
                            x2: endCoord[0],
                            y2: endCoord[1],
                        },
                        style: {
                            stroke: '#999999',
                            lineWidth: 5,
                            // lineDash: [4, 4], // 虚线
                        },
                    };
                },
                tooltip: {
                    formatter: function (params: any) {
                        const x = params.data[0]; // 散点的 X 值
                        const yActual = params.data[1]; // 散点的实际 Y 值

                        // 1. 根据直线方程计算出理论上的 Y 值 (y = wx + b)
                        // 注意：请确保这里的 weight 和 bias 变量与你传入函数的变量一致
                        const yTheoretical = weight * x + bias;

                        // 2. 计算真实差值（实际值 - 理论值）
                        const diff = yActual - yTheoretical;

                        // 3. 格式化数字：去掉末尾无用的 .0
                        const diffFormatted = Number(diff.toFixed(1));

                        // 4. 返回你想要的显示格式（例如：显示误差值，或者同时显示点和误差）
                        return `误差: ${diffFormatted}`;
                    },
                },
                // 绑定传入的散点数据
                data: scatter,
                z: 1,
            },
        ],
    };
}
const renderChart = async () => {
    if (!chartRef.value) return;
    const echarts = await import('echarts');
    myChart = echarts.init(chartRef.value);
    myChart.setOption(
        createErrorLineChartOption(props.scatter, props.weight, props.bias),
    );
};

onMounted(() => {
    renderChart();
    window.addEventListener('resize', () => myChart?.resize());
});

watch(
    () => [props.scatter, props.weight, props.bias, props.title],
    () => renderChart(),
);
</script>

<style scoped>
.echarts-wrapper {
    margin: 2rem 0;
    border: 1px solid var(--vp-c-divider);
    border-radius: 8px;
    background-color: var(--vp-c-bg-soft);
    padding: 15px;
}
.chart-container {
    width: 100%;
    aspect-ratio: 16 / 10;
}
</style>
