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
        <div style="display: flex; gap: 15px">
            <LogisticRegressionChart
                style="width: 50%"
                :scatter="props.scatter"
                :weight="w"
                :bias="b"
                :title="props.title"
            />
            <div style="flex-grow: 1; margin: 0" class="echarts-wrapper">
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
            <span>学习率 R =</span>
            <input
                v-model="R"
                type="number"
                style="width: 60px; padding: 2px 5px"
                step="0.1"
            />
            &nbsp;&nbsp;
            <span>模型方程：</span>
            <math xmlns="http://www.w3.org/1998/Math/MathML">
                <semantics>
                    <mrow>
                        <mi>y</mi>
                        <mo>=</mo>
                        <mfrac>
                            <mn>1</mn>
                            <mrow>
                                <mn>1</mn>
                                <mo>+</mo>
                                <msup>
                                    <mi>e</mi>
                                    <mrow>
                                        <mo>−</mo>
                                        <mo stretchy="false">(</mo>
                                        <mn>{{ Number(w.toFixed(3)) }}</mn>
                                        <mi>x</mi>
                                        <mo>+</mo>
                                        <mn>{{ Number(b.toFixed(3)) }}</mn>
                                        <mo stretchy="false">)</mo>
                                    </mrow>
                                </msup>
                            </mrow>
                        </mfrac>
                    </mrow>
                    <annotation encoding="application/x-tex">
                        y = \frac{1}{1 + e^{-({{ Number(w.toFixed(3)) }}x +
                        {{ Number(b.toFixed(3)) }})}}
                    </annotation>
                </semantics>
            </math>
        </div>
        <div>
            <span>步数：{{ step - 1 }}</span>
            &nbsp;&nbsp;
            <span>当前梯度：</span>
            <math xmlns="http://www.w3.org/1998/Math/MathML">
                <semantics>
                    <mrow>
                        <mover accent="true">
                            <mrow>
                                <mo stretchy="false">(</mo>
                                <mi>w</mi>
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
                                <mn>{{ formatGradient(c_dw) }}</mn>
                                <mo separator="true">,</mo>
                                <mn>{{ formatGradient(c_db) }}</mn>
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
                    xmlns="http://www.w3.org/2000/svg"
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
                回归原点 (0,0)
            </button>
            <button class="btn btn-secondary" @click="initParam(true)" title="尽量生成梯度消失的随机数据">
                <svg
                    xmlns="http://www.w3.org/2000/svg"
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
                随机远点
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
                    xmlns="http://www.w3.org/2000/svg"
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
                {{
                    isAutoRunning ? '自动迭代中...' : '单步更新 (长按快速迭代)'
                }}
            </button>
        </div>
    </div>
</template>

<script setup lang="ts">
import {
    onMounted,
    onUnmounted,
    PropType,
    ref,
    useTemplateRef,
    watch,
} from 'vue';
import LogisticRegressionChart from './LogisticRegressionChart.vue'; // 确保替换为逻辑回归组件
import { EChartsType } from 'echarts';

const props = defineProps({
    // 二分类数据，y 标签只有 0 或 1
    scatter: {
        type: Array as PropType<[number, number][]>,
        required: true,
    },
    title: { type: String, default: '' },
});

// 辅助 Sigmoid 函数
function sigmoid(z: number): number {
    return 1 / (1 + Math.exp(-z));
}

// 核心：基于 MSE 损失函数的逻辑回归评估与求导
function evaluateModel(scatter: [number, number][], w: number, b: number) {
    const n = scatter.length;
    if (n === 0) return { loss: 0, dw: 0, db: 0 };

    let sumLoss = 0,
        sumDw = 0,
        sumDb = 0;

    for (let i = 0; i < n; i++) {
        const [xi, yi] = scatter[i];

        // 1. 前向传播预测值
        const z = w * xi + b;
        const yHat = sigmoid(z);

        // 2. 计算当前样本的 MSE 误差
        const error = yi - yHat;
        sumLoss += error * error;

        // 3. 计算含 Sigmoid 导数的 MSE 梯度
        // dL/dyHat = -2 * error
        // dyHat/dz = yHat * (1 - yHat)  <-- 导致梯度消失的元凶
        // dz/dw = xi,  dz/db = 1
        const d_z = -2 * error * yHat * (1 - yHat);

        sumDw += d_z * xi;
        sumDb += d_z;
    }

    return {
        loss: sumLoss / n,
        dw: sumDw / n,
        db: sumDb / n,
    };
}

const w = ref(0);
const b = ref(0);
const R = ref(0.5); // 逻辑回归收敛慢，初始学习率可适当调高（例如 0.5 或 1.0）
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
        text: 'MSE (均方误差) 历史曲线',
        left: 'center',
        textStyle: { fontSize: 12 },
    },
    grid: [{ left: '10%', top: '20%', bottom: '15%', right: '5%' }],
    series: [
        {
            data: [] as number[],
            type: 'line',
            smooth: true,
            showSymbol: false,
            lineStyle: { color: '#67c23a' },
        },
    ],
} as any;

const updateModel = () => {
    const { loss, dw, db } = evaluateModel(props.scatter, w.value, b.value);
    mse = loss;
    c_dw.value = dw;
    c_db.value = db;

    if (myChart) {
        const mse_length = 500; // 调大存储空间以容纳长按快速迭代
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

// 格式化梯度输出，防止过小时变成 0 导致读者误解，引入科学计数法
const formatGradient = (val: number) => {
    if (Math.abs(val) < 0.001 && val !== 0) {
        return val.toExponential(2);
    }
    return Number(val.toFixed(3));
};

const initParam = (random = false) => {
    if (random && props.scatter.length > 0) {
        // 1. 获取输入数据在特征 X 方向的统计特性
        const xValues = props.scatter.map((pt) => pt[0]);
        const minX = Math.min(...xValues);
        const maxX = Math.max(...xValues);
        const rangeX = maxX - minX || 1.0; // 防止分母为 0
        const centerX = (minX + maxX) / 2;

        // 2. 动态计算缩放边界
        // 为了强行让 Sigmoid 图像的“突变过渡带”偏离数据中心，且能把数据赶进饱和平原：
        // 权重 w 的模需要使得 w * rangeX 显著大于激活函数的线性过渡区（大约在 -6 到 6 之间）
        const wBase = 12 / rangeX;

        // 随机给予正负大系数 w，使其饱和
        w.value =
            (Math.random() > 0.5 ? 1 : -1) * (wBase * (1 + Math.random() * 2));

        // 3. 动态平移偏置 b
        // 使得决策边界 x = -b/w 不在数据中心点，而是被狠狠地往左或往右拉偏 1~2 倍的数据跨度
        const targetXShift = centerX + (Math.random() - 0.5) * 3 * rangeX;
        b.value = -w.value * targetXShift;
    } else {
        w.value = 0;
        b.value = 0;
    }
    step.value = 0;
    if (MSEOption.series && MSEOption.series[0]) {
        MSEOption.series[0].data = [];
        MSEOption.xAxis.data = [];
    }
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
            // 每帧多迭代几步，让长按时动画更流畅，并迅速画出 MSE 曲线
            for (let i = 0; i < 5; i++) next();
        }, 50);
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

watch(
    () => props.scatter,
    () => {
        initParam();
    },
    { deep: true },
);
</script>

<style scoped>
/* 保持你原本优秀的样式体系 */
.echarts-wrapper {
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
