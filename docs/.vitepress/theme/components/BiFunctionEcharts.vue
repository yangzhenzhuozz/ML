<template>
    <div class="echarts-wrapper">
        <div ref="chartRef" class="chart-container"></div>
    </div>
</template>

<script setup lang="ts">
import { EChartsType } from 'echarts';
import { ECBasicOption } from 'echarts/types/dist/shared';
import {
    onMounted,
    onUnmounted,
    useTemplateRef,
    watch,
    type PropType,
} from 'vue';
const props = defineProps({
    //都是参数方程
    exprs: {
        type: Array as PropType<
            {
                u: {
                    min: number;
                    max: number;
                    step: number;
                };
                v: {
                    min: number;
                    max: number;
                    step: number;
                };
                x: string;
                y: string;
                z: string;
            }[]
        >,
        required: true,
    },
    title: { type: String, default: '' },
});

const chartRef = useTemplateRef('chartRef');
let myChart: EChartsType | null = null;
const renderChart = async () => {
    if (!chartRef.value) return;
    const echarts = await import('echarts');
    //@ts-ignore
    await import('echarts-gl'); // 必须引入 echarts-gl 以支持 3D
    const { compile } = await import('mathjs');
    const equations = props.exprs.map((item) => {
        return {
            ...item,
            ...{
                ex: compile(item.x),
                ey: compile(item.y),
                ez: compile(item.z),
            },
        };
    });
    myChart = echarts.init(chartRef.value);
    const option: ECBasicOption = {
        title: {
            text: props.title,
            left: 'center',
            textStyle: { fontSize: 16 },
        },
        tooltip: {},
        xAxis3D: {
            type: 'value',
        },
        yAxis3D: {
            type: 'value',
        },
        zAxis3D: {
            type: 'value',
        },
        grid3D: {
            viewControl: {
                autoRotate: false, // 是否开启自动旋转
                beta: 25, // 初始视角角度
                alpha: 20,
            },
        },
        series: equations.map((equation) => {
            return {
                type: 'surface',
                wireframe: {
                    show: true, // 是否显示网格线（数学教材建议开启，更方便看空间结构）
                    lineStyle: {
                        color: 'rgba(255,255,255,0.2)',
                        width: 0.5,
                    },
                },
                parametric: true,
                parametricEquation: {
                    u: {
                        max: equation?.u?.max ?? 1,
                        min: equation?.u?.min ?? -1,
                        step: equation?.u?.step ?? 0.1,
                    },
                    v: {
                        max: equation?.v?.max ?? 1,
                        min: equation?.v?.min ?? -1,
                        step: equation?.v?.step ?? 0.1,
                    },
                    x: function (u: number, v: number) {
                        return equation.ex.evaluate({ u, v });
                    },
                    y: function (u: number, v: number) {
                        return equation.ey.evaluate({ u, v });
                    },
                    z: function (u: number, v: number) {
                        return equation.ez.evaluate({ u, v });
                    },
                },
            };
        }),
    };
    myChart.setOption(option);
};

const handleResize = () => myChart?.resize();

onMounted(() => {
    renderChart();
    window.addEventListener('resize', handleResize);
});

onUnmounted(() => {
    window.removeEventListener('resize', handleResize);
    myChart?.dispose();
});

watch(
    () => props.exprs,
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
