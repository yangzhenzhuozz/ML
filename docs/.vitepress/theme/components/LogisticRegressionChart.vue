<template>
    <div class="echarts-wrapper">
        <div ref="chartRef" class="chart-container"></div>
    </div>
</template>

<script setup lang="ts">
import {
    CustomSeriesRenderItemParams,
    CustomSeriesRenderItemAPI,
    EChartsType,
} from 'echarts';
import { ECBasicOption } from 'echarts/types/dist/shared';
import {
    onMounted,
    onUnmounted,
    useTemplateRef,
    watch,
    type PropType,
} from 'vue';

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
    title: { type: String, default: '' },
});

const chartRef = useTemplateRef('chartRef');
let myChart: EChartsType | null = null;

// 核心：标准 Sigmoid 函数
function sigmoid(z: number): number {
    return 1 / (1 + Math.exp(-z));
}

function createLogisticErrorChartOption(
    scatter: [number, number][],
    weight: number,
    bias: number,
): ECBasicOption {
    // 1. 根据离散点动态计算 X 轴边界，同时确保有一个合理的显示视窗
    const xValues = scatter.map((p) => p[0]);
    const minX = xValues.length ? Math.min(...xValues) : -5;
    const maxX = xValues.length ? Math.max(...xValues) : 5;
    
    // 给左右两边留出 10% 的边距防止边缘点被裁切，同时保证范围不为 0
    const xMargin = maxX - minX || 2;
    const xAxisMin = Math.floor(minX - xMargin * 0.1);
    const xAxisMax = Math.ceil(maxX + xMargin * 0.1);

    // 2. 密集采样生成平滑的 Sigmoid 预测曲线数据
    const lineData: [number, number][] = [];
    const samplePoints = 150; // 采样点数
    const step = (xAxisMax - xAxisMin) / samplePoints;
    for (let i = 0; i <= samplePoints; i++) {
        const x = xAxisMin + i * step;
        const z = weight * x + bias;
        lineData.push([x, sigmoid(z)]);
    }

    return {
        animation: false,
        title: {
            text: props.title,
            left: 'center',
            textStyle: { fontSize: 12 },
        },
        tooltip: { trigger: 'item' },
        // 保持你原本的无 margins 撑满布局
        grid: [{ left: 'left', top: 'top', width: '100%', height: '100%' }],
        xAxis: {
            type: 'value',
            min: xAxisMin,
            max: xAxisMax,
            splitLine: { show: true }
        },
        yAxis: {
            type: 'value',
            // 逻辑回归的预测概率区间为 [0, 1]，稍微上下探 0.1 方便看清离散点在 0 和 1 上的聚集情况
            min: -0.1,
            max: 1.1,
            scale: false,
            axisLabel: {
                formatter: function (value: number) {
                    if (Math.abs(value) >= 1e6 || (value !== 0 && Math.abs(value) < 1e-4)) {
                        return value.toExponential(2);
                    }
                    return value.toString();
                },
            },
        },
        series: [
            {
                name: '二分类离散点',
                type: 'scatter',
                data: scatter,
                itemStyle: { color: '#ff4d4f' },
                tooltip: {
                    formatter: function (params: any) {
                        return `样本: (${params.data[0]} , 真实标签: ${params.data[1]})`;
                    },
                },
                z: 10,
            },
            {
                name: 'Sigmoid拟合曲线',
                type: 'line',
                data: lineData,
                showSymbol: false,
                smooth: true,
                lineStyle: { color: '#1890ff', width: 2 },
                z: 5,
            },
            {
                name: '分类误差',
                type: 'custom',
                renderItem: function (
                    _params: CustomSeriesRenderItemParams,
                    api: CustomSeriesRenderItemAPI,
                ) {
                    const xValue = api.value(0) as number;
                    const yActual = api.value(1) as number; // 实际值 (0 或 1)
                    
                    // 计算 Sigmoid 曲线在当前 x 处的预测概率值
                    const z = weight * xValue + bias;
                    const yTheoretical = sigmoid(z);

                    // 映射到 ECharts 画布的像素坐标
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
                            lineWidth: 4, // 略微调细一点点，在曲线上视觉观感更精致
                        },
                    };
                },
                tooltip: {
                    formatter: function (params: any) {
                        const x = params.data[0];
                        const yActual = params.data[1];
                        const z = weight * x + bias;
                        const yTheoretical = sigmoid(z);
                        return `预测概率: ${yTheoretical.toFixed(3)}<br/>绝对误差: ${Math.abs(yActual - yTheoretical).toFixed(3)}`;
                    },
                },
                data: scatter,
                z: 1,
            },
        ],
    };
}

const renderChart = async () => {
    if (!myChart) return;
    myChart.setOption(
        createLogisticErrorChartOption(props.scatter, props.weight, props.bias),
    );
};

const handleResize = () => myChart?.resize();

onUnmounted(() => {
    window.removeEventListener('resize', handleResize);
    myChart?.dispose();
});

onMounted(async () => {
    const echarts = await import('echarts');
    if (chartRef.value) {
        myChart = echarts.init(chartRef.value);
        window.addEventListener('resize', handleResize);
    }
    renderChart();
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
    aspect-ratio: 16 / 10;
}
.chart-container {
    width: 100%;
    height: 100%;
}
</style>