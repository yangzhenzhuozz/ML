<template>
    <div>
        <LineRegressionChart
            :scatter="props.scatter"
            :weight="optimalSolution.w"
            :bias="optimalSolution.b"
            :title="props.title"
        />
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
                        <mn>{{ optimalSolution.w }}</mn>
                        <mi>x</mi>
                        <mo>+</mo>
                        <mn>{{ optimalSolution.b }}</mn>
                    </mrow>
                    <annotation encoding="application/x-tex">
                        y={{ optimalSolution.w }}x+{{ optimalSolution.b }}
                    </annotation>
                </semantics>
            </math>
            <div>均方误差（MSE）：{{ optimalSolution.mse }}</div>
        </div>
    </div>
</template>
<script setup lang="ts">
import { computed, PropType } from 'vue';
import LineRegressionChart from './LineRegressionChart.vue';
const props = defineProps({
    scatter: {
        type: Array as PropType<[number, number][]>,
        required: true,
    },
    title: { type: String, default: '' },
});
const optimalSolution = computed(() => {
    const data = props.scatter;
    if (!data || data.length === 0) return { w: 0, b: 0, mse: 0 };

    // 对应教材中的累加公式
    let A = 0; // ∑ Y_i * X_i
    let B = 0; // ∑ X_i^2
    let C = 0; // ∑ X_i
    let D = 0; // ∑ Y_i
    let E = 0; // ∑ X_i (等于 C)
    let F = data.length; // ∑ 1 (即样本数量 n)

    for (const [x, y] of data) {
        A += y * x;
        B += x * x;
        C += x;
        D += y;
    }
    E = C;

    // 分母: BF - CE
    const denominator = B * F - C * E;

    // 防止分母为 0 导致崩溃（例如所有点都在同一条垂直线上）
    if (Math.abs(denominator) < 1e-6) return { w: 0, b: 0, mse: 0 };

    // 完美对应教材推导出的最终公式
    const wOptimal = (A * F - C * D) / denominator;
    const bOptimal = (B * D - A * E) / denominator;

    // 计算最优解下的理论最小 MSE
    let optimalSumError = 0;
    for (const [x, yActual] of data) {
        const yOpt = wOptimal * x + bOptimal;
        optimalSumError += (yActual - yOpt) * (yActual - yOpt);
    }
    const minMse = optimalSumError / data.length;

    return {
        w: Number(wOptimal.toFixed(2)),
        b: Number(bOptimal.toFixed(2)),
        mse: Number(minMse.toFixed(2)),
    };
});
</script>
