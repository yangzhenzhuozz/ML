<template>
    <div class="root_container">
        <div class="MLP_graph">
            <!-- 1. 输入阶段 -->
            <div class="stage">
                <div class="title-bar-container">
                    <div class="title">输入</div>
                </div>
                <div class="layer">
                    <div class="layer-column-layout">
                        <div class="neurone-vertical-container">
                            <div class="neurone" v-for="(value, idx) in input" :key="`in-${idx}`">
                                {{ value.toFixed(2) }}
                            </div>
                        </div>
                        <div class="node-controls">
                            <button class="node-btn" :disabled="input.length >= MAX_NODES" @click="input.push(0.0)">+</button>
                            <button class="node-btn" :disabled="input.length <= 1" @click="input.pop()">-</button>
                        </div>
                    </div>
                    <svg class="link-container" :width="gapHorizontal" :height="maxWidth * stepY">
                        <template v-for="s_idx in input.length" :key="`in-l-${s_idx}`">
                            <template v-for="e_idx in nextLayerSize(0)" :key="`in-l-${s_idx}-${e_idx}`">
                                <path fill="none" stroke="rgb(253, 214, 99)" :d="calcBezier(s_idx - 1, e_idx - 1)" />
                            </template>
                        </template>
                    </svg>
                </div>
            </div>

            <!-- 2. 隐藏层阶段 -->
            <div class="stage">
                <div class="title-bar-container">
                    <div class="title">隐藏层</div>
                    <div>{{ layerCfg.length }}/{{ MAX_LAYERS }}</div>
                    <div class="layer-controls">
                        <button class="ctrl-btn add" :disabled="layerCfg.length >= MAX_LAYERS" @click="addLayer">+</button>
                        <button class="ctrl-btn sub" :disabled="layerCfg.length <= 0" @click="removeLayer">-</button>
                    </div>
                </div>

                <div class="layer-container">
                    <template v-if="layerCfg.length > 0">
                        <div class="layer" v-for="(layer, layer_idx) in layerCfg" :key="`layer-${layer_idx}`">
                            <div class="layer-column-layout">
                                <div class="neurone-vertical-container">
                                    <div class="neurone" v-for="(neurone, n_idx) in layer" :key="`neu-${layer_idx}-${n_idx}`">
                                        {{ neurone.v.toFixed(2) }}
                                    </div>
                                </div>
                                <div class="node-controls">
                                    <button class="node-btn" :disabled="layer.length >= MAX_NODES" @click="addNode(layer_idx)">+</button>
                                    <button class="node-btn" :disabled="layer.length <= 1" @click="removeNode(layer_idx)">-</button>
                                </div>
                            </div>
                            <svg class="link-container" :width="gapHorizontal" :height="maxWidth * stepY">
                                <template v-for="s_idx in layer.length" :key="`hid-l-${layer_idx}-${s_idx}`">
                                    <template v-for="e_idx in nextLayerSize(layer_idx + 1)" :key="`hid-l-${layer_idx}-${s_idx}-${e_idx}`">
                                        <path fill="none" stroke="rgb(253, 214, 99)" :d="calcBezier(s_idx - 1, e_idx - 1)" />
                                    </template>
                                </template>
                            </svg>
                        </div>
                    </template>

                    <div v-else class="empty-layer-placeholder">
                        <span>直连输出</span>
                    </div>
                </div>
            </div>

            <!-- 3. 输出阶段 -->
            <div class="stage">
                <div class="title-bar-container">
                    <div class="title">输出</div>
                </div>
                <div class="neurone-vertical-container">
                    <div class="neurone">{{ output.toFixed(2) }}</div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';

const MAX_LAYERS = 10;
const MAX_NODES = 10;

const neuronSize = ref(50);
const gapVertical = ref(20);
const gapHorizontal = ref(60);

const stepY = computed(() => neuronSize.value + gapVertical.value);
const cssNeUniformWidth = computed(() => `${neuronSize.value}px`);
const cssVerticalGap = computed(() => `${gapVertical.value}px`);

const input = ref([0.5, 0.2, -0.1]);

const layerCfg = ref<{ w: number[]; b: number; v: number }[][]>([
    [
        { w: [], b: 1, v: 0.12 },
        { w: [], b: 1, v: 0.88 },
    ],
]);
const output = ref(0.99);

const maxWidth = computed(() => {
    let max = input.value.length;
    for (const layer of layerCfg.value) {
        max = Math.max(max, layer.length);
    }
    return max;
});

const nextLayerSize = (currentLayerIdx: number) => {
    if (layerCfg.value.length === 0 || currentLayerIdx >= layerCfg.value.length) {
        return 1;
    }
    return layerCfg.value[currentLayerIdx].length;
};

const calcBezier = (sIdx: number, eIdx: number) => {
    const startX = 0;
    const startY = neuronSize.value / 2 + stepY.value * sIdx;
    const endX = gapHorizontal.value;
    const endY = neuronSize.value / 2 + stepY.value * eIdx;
    const ctrlX = gapHorizontal.value / 2;
    return `M ${startX},${startY} C ${ctrlX},${startY} ${ctrlX},${endY} ${endX},${endY}`;
};

// 🌟 重构：智能判断层宽的新增隐藏层方法
const addLayer = () => {
    if (layerCfg.value.length >= MAX_LAYERS) return;

    // 1. 获取新层的目标宽度（如果当前数组为空则默认为 2，否则读取最后一层的长度）
    const targetNodeCount = layerCfg.value.length === 0 ? 2 : layerCfg.value[layerCfg.value.length - 1].length;

    // 2. 动态生成对应节点数量的数组
    const newLayer = Array.from({ length: targetNodeCount }, () => ({
        w: [],
        b: 1,
        v: 0.0,
    }));

    layerCfg.value.push(newLayer);
};

const removeLayer = () => {
    if (layerCfg.value.length <= 0) return;
    layerCfg.value.pop();
};

const addNode = (layerIdx: number) => {
    if (layerCfg.value[layerIdx].length >= MAX_NODES) return;
    layerCfg.value[layerIdx].push({ w: [], b: 1, v: 0.0 });
};

const removeNode = (layerIdx: number) => {
    if (layerCfg.value[layerIdx].length <= 1) return;
    layerCfg.value[layerIdx].pop();
};
</script>

<style scoped>
.root_container {
    padding: 20px;
    background: #fafafa;
}
.MLP_graph {
    --neurone-width: v-bind(cssNeUniformWidth);
    --vertical-gap: v-bind(cssVerticalGap);
    display: flex;
    gap: 0;
    user-select: none;
    align-items: stretch;
    overflow: auto;
}
.stage {
    display: flex;
    flex-direction: column;
    align-items: center;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 15px 0;
    background: #ffffff;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
}
.stage:first-child .layer-column-layout {
    padding-left: 15px;
}
.stage:last-child .neurone-vertical-container {
    padding-right: 15px;
}

.title-bar-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    height: 80px;
    margin-bottom: 15px;
    padding: 0 15px;
}
.title {
    font-weight: bold;
    color: #4b5563;
    font-size: 14px;
}

.layer-container {
    display: flex;
    align-items: flex-start;
}
.layer {
    display: flex;
}
.layer-column-layout {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 15px;
}
.neurone-vertical-container {
    display: flex;
    flex-direction: column;
    gap: var(--vertical-gap);
}
.link-container {
    display: block;
    overflow: visible;
}
.neurone {
    width: var(--neurone-width);
    height: var(--neurone-width);
    border-radius: 50%;
    border: solid 2px #3b82f6;
    background: #ffffff;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: monospace;
    font-size: 13px;
    color: #1e3a8a;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    overflow: hidden;
    contain: layout style paint;
}

.empty-layer-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 0;
    height: var(--neurone-width);
    color: #9ca3af;
    font-size: 13px;
    font-style: italic;
}

.layer-controls {
    display: flex;
    gap: 4px;
}
.ctrl-btn {
    width: 20px;
    height: 20px;
    border-radius: 4px;
    border: 1px solid #d1d5db;
    background: #ffffff;
    color: #374151;
    font-size: 12px;
    font-weight: bold;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s ease;
}
.ctrl-btn:hover:not(:disabled) {
    background: #f3f4f6;
    border-color: #9ca3af;
}
.ctrl-btn:disabled {
    background: #f9fafb;
    color: #d1d5db;
    border-color: #e5e7eb;
    cursor: not-allowed;
}

.node-controls {
    display: flex;
    gap: 6px;
}
.node-btn {
    width: 22px;
    height: 22px;
    border-radius: 4px;
    border: 1px solid #e5e7eb;
    background: #f9fafb;
    color: #6b7280;
    font-size: 12px;
    font-weight: bold;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s ease;
}
.node-btn:hover:not(:disabled) {
    background: #eff6ff;
    border-color: #bfdbfe;
    color: #2563eb;
}
.node-btn:disabled {
    background: #ffffff;
    color: #f3f4f6;
    border-color: #f3f4f6;
    cursor: not-allowed;
}
</style>
