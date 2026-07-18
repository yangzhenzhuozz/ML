<template>
    <div
        style="
            border: 1px solid var(--vp-c-divider);
            background-color: var(--vp-c-bg-soft);
            padding: 20px;
            border-radius: 8px;
        "
        class="vp-raw"
    >
        <div style="display: flex">
            <LineRegressionChart
                style="width: 50%"
                :scatter="props.scatter"
                :weight="w"
                :bias="b"
                :title="props.title"
            />
            <div style="flex-grow: 1" class="echarts-wrapper">
                <div ref="chartRef" class="chart-container"></div>
            </div>
        </div>
        <div
            style="
                display: flex;
                align-items: center;
                flex-wrap: wrap;
                gap: 8px;
                margin: 16px 0;
            "
        >
            <span>学习率R=</span>
            <input v-model="R" type="number" style="width: 50px" />
            &nbsp;&nbsp;
            <span>模型方程：</span>
            <math xmlns="http://w3.org">
                <semantics>
                    <mrow>
                        <mi>y</mi>
                        <mo>=</mo>
                        <mn>{{ Number(w.toFixed(3)) }}</mn>
                        <mi>x</mi>
                        <mo>+</mo>
                        <mn>{{ Number(b.toFixed(3)) }}</mn>
                    </mrow>
                    <annotation encoding="application/x-tex">
                        y={{ Number(w.toFixed(3)) }}x+{{ Number(b.toFixed(3)) }}
                    </annotation>
                </semantics>
            </math>
        </div>
        <div>
            <span>步数：{{ step - 1 }}</span>
            &nbsp;&nbsp;
            <span>梯度：</span>
            <math xmlns="http://w3.org">
                <semantics>
                    <mrow>
                        <mover accent="true">
                            <mrow>
                                <mo stretchy="false">(</mo>
                                <mi>ω</mi>
                                <mo separator="true">,</mo>
                                <mi>b</mi>
                                <mo stretchy="false">)</mo>
                            </mrow>
                            <mo stretchy="true">→</mo>
                        </mover>
                        <mo>=</mo>
                        <mover accent="true">
                            <mrow>
                                <mo stretchy="false">(</mo>
                                <mn>{{ Number(c_dw.toFixed(3)) }}</mn>
                                <mo separator="true">,</mo>
                                <mn>{{ Number(c_db.toFixed(3)) }}</mn>
                                <mo stretchy="false">)</mo>
                            </mrow>
                            <mo stretchy="true">→</mo>
                        </mover>
                    </mrow>
                </semantics>
            </math>
        </div>

        <div class="btn-group">
            <button class="btn btn-secondary" @click="initParam()">
                <svg
                    xmlns="http://w3.org"
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                >
                    <path
                        d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"
                    />
                    <path d="M3 3v5h5" />
                </svg>
                初始化参数
            </button>
            <button class="btn btn-secondary" @click="initParam(true)">
                <svg
                    xmlns="http://w3.org"
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                >
                    <path d="M18 4h4v4" />
                    <path d="M18 16h4v4" />
                    <path
                        d="M2 18h4a4 4 0 0 0 3.13-1.51L14.87 7.51A4 4 0 0 1 18 6h4"
                    />
                    <path d="m22 4-4 4" />
                    <path d="M2 6h4a4 4 0 0 1 3.13 1.51l1.43 1.76" />
                    <path d="M15 14.26a4 4 0 0 0 3 1.74h4" />
                    <path d="m22 20-4-4" />
                </svg>
                随机参数
            </button>
            <button
                class="btn btn-primary"
                :class="{ 'is-running': isAutoRunning }"
                @mousedown="startPress"
                @mouseup="endPress"
                @mouseleave="endPress"
                @touchstart.prevent="startPress"
                @touchend="endPress"
            >
                <svg
                    xmlns="http://w3.org"
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                >
                    <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
                {{ isAutoRunning ? '自动运行中...' : '下一步 (长按自动运行)' }}
            </button>
        </div>
    </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, PropType, ref, useTemplateRef } from 'vue';
import LineRegressionChart from './LineRegressionChart.vue';
import { EChartsType } from 'echarts';

const props = defineProps({
    scatter: {
        type: Array as PropType<[number, number][]>,
        required: true,
    },
    title: { type: String, default: '' },
});

function evaluateModel(scatter: [number, number][], w: number, b: number) {
    const n = scatter.length;
    if (n === 0) return { loss: 0, dw: 0, db: 0 };

    let sumLoss = 0,
        sumDw = 0,
        sumDb = 0;
    for (let i = 0; i < n; i++) {
        const [xi, yi] = scatter[i];
        const error = yi - (w * xi + b);
        sumLoss += error * error;
        sumDw += -2 * error * xi;
        sumDb += -2 * error;
    }

    return {
        loss: sumLoss / n,
        dw: sumDw / n,
        db: sumDb / n,
    };
}

const w = ref(0);
const b = ref(0);
const R = ref(0.02);
const c_dw = ref(0);
const c_db = ref(0);
const isAutoRunning = ref(false);

let step = ref(0);
let mse = 0;
let delayTimer: any = null;
let intervalTimer: any = null;
let myChart: EChartsType | null = null;
const chartRef = useTemplateRef('chartRef');

const MSEOption = {
    animation: false,
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: [] as number[] },
    yAxis: { type: 'value' },
    title: {
        text: 'MSE（均方误差）',
        left: 'center',
        textStyle: { fontSize: 12 },
    },
    grid: [{ left: 'left', top: 'top', bottom: '20', right: '20' }],
    series: [
        {
            data: [] as number[],
            type: 'line',
            smooth: true,
            showSymbol: false,
        },
    ],
} as any;

const updateModel = () => {
    const { loss, dw, db } = evaluateModel(props.scatter, w.value, b.value);
    mse = loss;
    c_dw.value = dw;
    c_db.value = db;
    if (myChart) {
        const mse_length = 200;
        const data = MSEOption.series[0].data;
        const xAxis = MSEOption.xAxis.data;
        if (data.length > mse_length) {
            data.shift();
            xAxis.shift();
        }
        data.push(mse);
        xAxis.push(step.value);
        step.value++;
        myChart.setOption(MSEOption);
    }
};

const initParam = (random = false) => {
    if (random) {
        //把随机参数限制在[0,20]
        w.value = Math.random() * 20;
        b.value = Math.random() * 20;
    } else {
        w.value = 0;
        b.value = 0;
    }
    step.value = 0;
    MSEOption.series[0].data = [];
    MSEOption.xAxis.data = [];
    updateModel();
};

const next = () => {
    w.value = w.value - R.value * c_dw.value;
    b.value = b.value - R.value * c_db.value;
    updateModel();
};

const startPress = () => {
    if (delayTimer) clearTimeout(delayTimer);
    if (intervalTimer) clearInterval(intervalTimer);
    next();
    delayTimer = setTimeout(() => {
        isAutoRunning.value = true;
        intervalTimer = setInterval(() => {
            for (let i = 0; i < 10; i++) next();
        }, 60);
    }, 300);
};

const endPress = () => {
    if (delayTimer) {
        clearTimeout(delayTimer);
        delayTimer = null;
    }
    if (intervalTimer) {
        clearInterval(intervalTimer);
        intervalTimer = null;
    }
    isAutoRunning.value = false;
};

onUnmounted(() => {
    endPress();
});

onMounted(async () => {
    const echarts = await import('echarts');
    if (chartRef.value) {
        myChart = echarts.init(chartRef.value);
        window.addEventListener('resize', () => myChart?.resize());
    }
    updateModel();
});
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
.btn-group {
    display: flex;
    gap: 12px;
    margin-top: 16px;
    flex-wrap: wrap;
}
.btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    font-size: 14px;
    font-weight: 500;
    padding: 8px 16px;
    border-radius: 8px;
    cursor: pointer;
    user-select: none;
    transition: all 0.2s cubic-bezier(0.165, 0.84, 0.44, 1);
    outline: none;
}
.btn-secondary {
    background-color: var(--vp-c-bg);
    border: 1px solid var(--vp-c-divider);
    color: var(--vp-c-text-1);
}
.btn-secondary:hover {
    border-color: var(--vp-c-brand-1);
    color: var(--vp-c-brand-1);
    background-color: var(--vp-c-bg-soft);
}
.btn-secondary:active {
    background-color: var(--vp-c-bg-alt);
}
.btn-primary {
    background-color: var(--vp-c-brand-1);
    border: 1px solid transparent;
    color: #ffffff;
}
.btn-primary:hover {
    background-color: var(--vp-c-brand-2);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    transform: translateY(-1px);
}
.btn-primary:active {
    transform: translateY(0);
    background-color: var(--vp-c-brand-3);
}
.btn-primary.is-running {
    background-color: #e6a23c !important;
    box-shadow: 0 0 0 4px rgba(230, 162, 60, 0.3);
    animation: pulse 1.2s infinite ease-in-out;
}
@keyframes pulse {
    0% {
        opacity: 1;
    }
    50% {
        opacity: 0.75;
    }
    100% {
        opacity: 1;
    }
}
</style>
