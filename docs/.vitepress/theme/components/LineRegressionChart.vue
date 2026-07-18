<template>
    <div class="echarts-wrapper">
        <div ref="chartRef" class="chart-container"></div>
    </div>
</template>

<script setup lang="ts">
import { CustomSeriesRenderItemParams, CustomSeriesRenderItemAPI, EChartsType } from 'echarts';
import { ECBasicOption } from 'echarts/types/dist/shared';
import { onMounted, useTemplateRef, watch, type PropType } from 'vue';

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

function createErrorLineChartOption(
    scatter: [number, number][],
    weight: number,
    bias: number,
): ECBasicOption {
    const xValues = scatter.map((p) => p[0]);
    const minX = Math.min(...xValues);
    const maxX = Math.max(...xValues);

    const xMargin = maxX - minX || 1;
    const xAxisMax = Math.ceil(maxX + xMargin * 0.1);

    const lineStartPoint = [0, weight * 0 + bias];
    const lineEndPoint = [xAxisMax, weight * xAxisMax + bias];

    const yValues = [
        ...scatter.map((p) => p[1]), 
        lineStartPoint[1], 
        lineEndPoint[1]
    ];
    const absoluteMinY = Math.min(...yValues);
    
    const yAxisMin = Math.min(0, absoluteMinY);

    return {
        animation: false,
        title: {
            text: props.title,
            left: 'center',
            textStyle: { fontSize: 12 },
        },
        tooltip: { trigger: 'item' },
        grid: [{ left: 'left', top: 'top', width: '100%', height: '100%' }],
        xAxis: {
            type: 'value',
            min: 0,
            max: xAxisMax,
        },
        yAxis: {
            type: 'value',
            min: yAxisMin,
            scale: false,
            axisLabel: {
                formatter: function (value: number) {
                    if (Math.abs(value) >= 1e6 || (value !== 0 && Math.abs(value) < 1e-4)) {
                        return value.toExponential(2);
                    }
                    return value.toString();
                }
            }
        },
        series: [
            {
                name: '离散点',
                type: 'scatter',
                data: scatter,
                itemStyle: { color: '#ff4d4f' },
                tooltip: {
                    formatter: function (params: any) {
                        return `(${params.data[0]} , ${Number(params.data[1].toFixed(3))})`;
                    },
                },
                z: 10,
            },
            {
                name: '拟合直线',
                type: 'line',
                data: [lineStartPoint, lineEndPoint],
                showSymbol: false,
                lineStyle: { color: '#1890ff', width: 2 },
                z: 5,
            },
            {
                name: '误差',
                type: 'custom',
                renderItem: function (
                    _params: CustomSeriesRenderItemParams,
                    api: CustomSeriesRenderItemAPI,
                ) {
                    const xValue = api.value(0) as number;
                    const yActual = api.value(1) as number;
                    const yTheoretical = weight * xValue + bias;

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
                        },
                    };
                },
                tooltip: {
                    formatter: function (params: any) {
                        const x = params.data[0];
                        const yActual = params.data[1];
                        const yTheoretical = weight * x + bias;
                        return `误差: ${Number((yActual - yTheoretical).toFixed(1))}`;
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
        createErrorLineChartOption(props.scatter, props.weight, props.bias),
    );
};

onMounted(async () => {
    const echarts = await import('echarts');
    if (chartRef.value) {
        myChart = echarts.init(chartRef.value);
        window.addEventListener('resize', () => myChart?.resize());
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
