<template>
  <div class="gpu-mlp">
    <div class="panel">
      <div class="title">🚀 GPU 加速版 MLP · 圆环分类（单样本 SGD）</div>
      <div class="fields">
        <label>隐藏层宽度
          <input type="number" v-model.number="cfg.width" min="1" max="64" />
        </label>
        <label>隐藏层深度
          <input type="number" v-model.number="cfg.depth" min="1" max="8" />
        </label>
        <label>样本数
          <input type="number" v-model.number="cfg.samples" min="100" step="100" />
        </label>
        <label>学习率 lr
          <input type="number" v-model.number="cfg.lr" min="0.01" max="1" step="0.05" />
        </label>
        <label>训练轮数 epochs
          <input type="number" v-model.number="cfg.epochs" min="10" step="50" />
        </label>
        <label>内半径 r1
          <input type="number" v-model.number="cfg.r1" min="0.1" step="0.1" />
        </label>
        <label>外半径 r2
          <input type="number" v-model.number="cfg.r2" min="0.5" step="0.1" />
        </label>
        <button class="run" :disabled="running" @click="run">
          {{ running ? '训练中…' : '🚀 开始训练' }}
        </button>
      </div>
    </div>

    <pre class="log">{{ log || '（训练期间不从 GPU 读数据，完成后这里显示结果……）' }}</pre>

    <div class="result" v-if="heatmap !== null">
      <div class="heat-wrap">
        <canvas ref="heat" width="480" height="480"></canvas>
        <div class="legend">
          <span class="sw in"></span> 圆内（预测概率低）
          <span class="sw mid"></span> 边界（≈0.5）
          <span class="sw out"></span> 圆外（预测概率高）
        </div>
      </div>
      <p class="meta">训练耗时：<strong>{{ formatMs(trainMs) }}</strong></p>
    </div>

    <p class="hint">💡 前向 / 反向 / 更新全部在 GPU 纹理间流转，训练过程零 readPixels；只在完成后生成热力图时读回一次。</p>
  </div>
</template>

<script setup lang="ts">
import { ref, onUnmounted } from 'vue'
import MlpWorkerCtor from './gpuMlpWorker.ts?worker'

/* 训练循环在 Web Worker（OffscreenCanvas + WebGL2）中执行，主线程只负责：
   ① 发送训练配置；② 收集每个 epoch 的平均损失；③ 绘制损失曲线与耗时。 */

const cfg = ref({ width: 8, depth: 3, samples: 800, lr: 0.3, epochs: 300, r1: 0.5, r2: 1.5 });
const running = ref(false);
const log = ref('');
const heatmap = ref<number[] | null>(null);
const trainMs = ref<number | null>(null);
const heat = ref<HTMLCanvasElement | null>(null);

let worker: Worker | null = null;

onUnmounted(() => {
    worker?.terminate();
    worker = null;
});

function run() {
    worker?.terminate(); // 若上一次还在跑，先停掉
    running.value = true;
    heatmap.value = null;
    trainMs.value = null;
    log.value = '';

    const w = new MlpWorkerCtor();
    worker = w;

    w.onmessage = (e: MessageEvent) => {
        const msg = e.data as { type: string; text?: string; trainMs?: number; heatmap?: number[]; grid?: number; message?: string };
        if (msg.type === 'info' && msg.text) {
            log.value += msg.text + '\n';
        } else if (msg.type === 'done' && msg.heatmap) {
            heatmap.value = msg.heatmap;
            trainMs.value = msg.trainMs ?? null;
            running.value = false;
            log.value += `\n✅ 训练完成，耗时 ${formatMs(msg.trainMs ?? null)}，正在绘制热力图…`;
            // 等 v-if 把 canvas 挂载出来后再绘制
            requestAnimationFrame(() => drawHeatmap(msg.heatmap as number[]));
        } else if (msg.type === 'error') {
            log.value += `❌ ${msg.message ?? '未知错误'}\n`;
            running.value = false;
        }
    };
    w.onerror = (e: ErrorEvent) => {
        log.value += `❌ Worker 出错：${e.message}\n`;
        running.value = false;
    };

    w.postMessage({ type: 'train', value: { ...cfg.value } });
}

/** 用 Worker 返回的概率数组绘制热力图（蓝=圆内 → 红=圆外） */
function drawHeatmap(probs: number[]) {
    const canvas = heat.value;
    if (!canvas) return;
    const ctx = canvas.getContext('2d')!;
    const grid = Math.round(Math.sqrt(probs.length)) || 60;
    const cell = canvas.width / grid;
    const span = 3.2;
    for (let row = 0; row < grid; row++) {
        for (let col = 0; col < grid; col++) {
            ctx.fillStyle = heatColor(probs[row * grid + col]);
            ctx.fillRect(col * cell, row * cell, cell, cell);
        }
    }
    // 画两个参考圆（r1 / r2）
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const px = (r: number) => (r / (span / 2)) * (canvas.width / 2);
    ctx.strokeStyle = 'rgba(255,255,255,0.55)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, px(cfg.value.r1), 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, px(cfg.value.r2), 0, Math.PI * 2);
    ctx.stroke();
}

function heatColor(p: number): string {
    const hue = 230 * (1 - p); // p=0 → 蓝(230)，p=1 → 红(0)
    return `hsl(${hue}, 80%, 55%)`;
}

/** 格式化训练耗时：小于 1 秒显示毫秒，否则显示秒 */
function formatMs(ms: number | null) {
    if (ms === null || ms === undefined) return '';
    return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${ms.toFixed(0)} ms`;
}
</script>

<style scoped>
.gpu-mlp {
    width: 100%;
    box-sizing: border-box;
    border: 1px solid var(--vp-c-divider);
    border-radius: 10px;
    margin: 16px 0;
    padding: 14px;
    background: var(--vp-c-bg-soft);
}
.panel .title {
    font-weight: 600;
    margin-bottom: 10px;
}
.fields {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 10px;
}
.fields label {
    display: flex;
    flex-direction: column;
    font-size: 12px;
    gap: 4px;
    color: var(--vp-c-text-2);
}
.fields input {
    padding: 4px 6px;
    border: 1px solid var(--vp-c-divider);
    border-radius: 6px;
    background: var(--vp-c-bg);
    color: var(--vp-c-text-1);
}
.run {
    grid-column: 1 / -1;
    padding: 8px 12px;
    border: none;
    border-radius: 8px;
    background: var(--vp-c-brand);
    color: #fff;
    cursor: pointer;
    font-weight: 600;
}
.run:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}
.log {
    margin: 12px 0 0;
    height: 160px;
    overflow: auto;
    padding: 10px;
    border-radius: 8px;
    background: var(--vp-c-bg);
    border: 1px solid var(--vp-c-divider);
    font-size: 12px;
    line-height: 1.6;
    white-space: pre-wrap;
}
.result {
    margin-top: 12px;
}
.result canvas {
    width: 100%;
    max-width: 480px;
    height: auto;
    border: 1px solid var(--vp-c-divider);
    border-radius: 8px;
    background: var(--vp-c-bg);
}
.meta {
    margin-top: 8px;
    font-size: 14px;
}
.meta strong {
    color: var(--vp-c-brand);
}
.heat-wrap {
    display: flex;
    gap: 14px;
    align-items: center;
}
.heat-wrap canvas {
    width: min(100%, 420px);
    height: auto;
    border: 1px solid var(--vp-c-divider);
    border-radius: 8px;
    background: var(--vp-c-bg);
}
.legend {
    display: flex;
    flex-direction: column;
    gap: 6px;
    font-size: 12px;
    color: var(--vp-c-text-2);
}
.legend .sw {
    display: inline-block;
    width: 14px;
    height: 14px;
    border-radius: 3px;
    vertical-align: -2px;
    margin-right: 4px;
}
.sw.in { background: hsl(230, 80%, 55%); }
.sw.mid { background: hsl(115, 80%, 55%); }
.sw.out { background: hsl(0, 80%, 55%); }
.hint {
    margin-top: 10px;
    font-size: 12px;
    color: var(--vp-c-text-3);
}
</style>