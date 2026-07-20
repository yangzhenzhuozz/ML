<template>
    <div class="echarts-wrapper">
        <div ref="chartRef" class="chart-container"></div>
    </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, useTemplateRef, watch, type PropType } from 'vue';
import type { EChartsType } from 'echarts';

const props = defineProps({
    // 参数方程数组
    exprs: {
        type: Array as PropType<
            {
                min?: number; // 默认-5
                max?: number; // 默认5
                x: string;    // 例如: "3" 或 "t"
                y: string;    // 例如: "t" 或 "1 / (1 + exp(-t))"
            }[]
        >,
        required: true,
    },
});

const chartRef = useTemplateRef('chartRef');
let myChart: EChartsType | null = null;

const renderChart = async () => {
    if (!chartRef.value) return;

    // 1. 异步按需加载依赖
    const echarts = await import('echarts');
    const { compile } = await import('mathjs');

    // 2. 销毁旧实例，防止内存泄漏和重影
    if (myChart) {
        myChart.dispose();
    }

    // 3. 解析方程并采样生成 [x, y] 数据点
    const seriesData = props.exprs.map((item) => {
        const tMin = item.min ?? -5;
        const tMax = item.max ?? 5;
        
        // 动态计算步长，确保总是有 200 个采样点，兼顾性能与平滑度
        const step = (tMax - tMin) / 200; 
        
        const exprX = compile(item.x);
        const exprY = compile(item.y);
        const coordinates: [number, number][] = [];

        for (let t = tMin; t <= tMax; t += step) {
            try {
                // 传入自变量 t 进行数学求值
                const context = { t };
                const xVal = exprX.evaluate(context);
                const yVal = exprY.evaluate(context);
                
                // 过滤掉无法计算的无效数学值 (如分母为0导致的NaN或Infinity)
                if (Number.isFinite(xVal) && Number.isFinite(yVal)) {
                    coordinates.push([xVal, yVal]);
                }
            } catch (e) {
                // 捕获公式输入不完整时的解析错误，避免页面白屏
                console.warn('数学公式解析错误:', e);
            }
        }

        return {
            type: 'line' as const,
            smooth: true,
            showSymbol: false, // 隐藏数据点的小圆圈，让线条更清爽
            data: coordinates,  // 二维数组格式: [[x1, y1], [x2, y2], ...]
        };
    });

    // 4. 初始化并配置 ECharts
    myChart = echarts.init(chartRef.value);
    
    const option = {
        // 开启十字准星提示框，方便看坐标
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'cross' }
        },
        // 坐标轴设为 value 轴，解锁任意方向和曲率的线
        xAxis: {
            type: 'value',
            name: 'X',
            splitLine: { show: true } // 显示网格线，更像数学坐标系
        },
        yAxis: {
            type: 'value',
            name: 'Y',
            splitLine: { show: true }
        },
        series: seriesData,
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

// 监听深度变化，防止数组内部属性改动未触发刷新
watch(
    () => props.exprs,
    () => renderChart(),
    { deep: true }
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