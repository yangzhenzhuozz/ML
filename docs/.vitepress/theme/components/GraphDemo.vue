<template>
  <div class="graph-demo">
    <div class="panel">
      <div class="title">⚙️ 计算图 · 圆环分类演示（可调参数自动构造网络）</div>
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

    <pre class="log">{{ log || '（训练日志会实时显示在这里……）' }}</pre>

    <div class="result">
      <p class="acc" v-if="acc !== null">测试集准确率：<strong>{{ acc.toFixed(2) }}%</strong></p>
      <div v-if="acc !== null" class="heat-wrap">
        <canvas ref="heat" width="480" height="480"></canvas>
        <div class="legend">
          <span class="sw in"></span> 圆内（预测概率低）
          <span class="sw mid"></span> 边界（≈0.5）
          <span class="sw out"></span> 圆外（预测概率高）
        </div>
      </div>
      <div v-else class="heat-placeholder">🚀 训练完成后，这里会绘制「预测圆外概率」热力图</div>
      <div class="colorbar">
        <canvas ref="colorbar" width="480" height="16"></canvas>
        <div class="scale">
          <span>0 · 圆内</span>
          <span>0.5 · 边界</span>
          <span>1 · 圆外</span>
        </div>
      </div>
    </div>
    <p class="hint">💡 每次点击「开始训练」，都会重新随机初始化网络参数并重新生成训练数据，因此每次结果会有小幅波动。</p>
  </div>
</template>

<script setup lang="ts">
import { ref, onUnmounted, onMounted } from 'vue'
import GraphWorkerCtor from './graphWorker.ts?worker' // Vite 会把 TS worker 打包成独立 .js chunk

/* 计算图核心与训练循环已移到 Web Worker（./graphWorker.ts）中执行，
   避免训练重计算阻塞主线程（UI 卡死）。主线程只负责：
   ① 发送训练配置；② 接收进度日志；③ 用返回的热力图数据绘制 canvas。 */

const cfg = ref({ width: 8, depth: 2, samples: 800, lr: 0.3, epochs: 300, r1: 0.5, r2: 1.5 });
const running = ref(false);
const log = ref('');
const acc = ref<number | null>(null);
const heatmap = ref<number[] | null>(null);
const heat = ref<HTMLCanvasElement | null>(null);
const colorbar = ref<HTMLCanvasElement | null>(null);

// 训练放在 Web Worker 中执行，避免阻塞主线程（UI 卡死）
let worker: Worker | null = null;

onUnmounted(() => {
    worker?.terminate();
    worker = null;
});

function run() {
    worker?.terminate(); // 若上一次还在跑，先停掉
    running.value = true;
    acc.value = null;
    heatmap.value = null;
    log.value = '';

    const w = new GraphWorkerCtor();
    worker = w;

    w.onmessage = (e: MessageEvent) => {
        const msg = e.data as { type: string; text?: string; acc?: number; heatmap?: number[]; message?: string };
        if (msg.type === 'progress' && msg.text) {
            log.value += msg.text; // 训练日志实时追加
        } else if (msg.type === 'done' && msg.acc !== undefined && msg.heatmap) {
            acc.value = msg.acc;
            heatmap.value = msg.heatmap;
            log.value += `\n测试集准确率：${msg.acc.toFixed(2)}%`;
            running.value = false;
            // 等 v-if 把 canvas 挂载出来后再绘制（否则 heat.value 还是 null）
            requestAnimationFrame(() => drawHeatmap(msg.heatmap as number[]));
        } else if (msg.type === 'error') {
            log.value += `❌ ${msg.message ?? '未知错误'}\n`;
            running.value = false;
        }
    };
    w.onerror = (e) => {
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
    const span = 3.2; // 展示范围 [-1.6, 1.6]
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

/** 画连续色卡：展示「预测圆外概率 p」0→1 对应的颜色（与热力图着色一致） */
function drawColorbar() {
    const c = colorbar.value;
    if (!c) return;
    const ctx = c.getContext('2d')!;
    for (let x = 0; x < c.width; x++) {
        const p = x / (c.width - 1);
        ctx.fillStyle = heatColor(p);
        ctx.fillRect(x, 0, 1, c.height);
    }
}

// 色卡常驻渲染（不依赖训练结果）
onMounted(() => drawColorbar());
</script>

<style scoped>
.graph-demo {
    width: 100%; /* 始终占满父容器，随浏览器宽度自适应 */
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
    height: 200px; /* 固定高度：训练时内容增长也不跳动，改为内部滚动 */
    overflow: auto;
    padding: 10px;
    border-radius: 8px;
    background: var(--vp-c-bg);
    border: 1px solid var(--vp-c-divider);
    font-size: 12px;
    line-height: 1.6;
    white-space: pre-wrap;
}
.acc {
    font-size: 15px;
    margin: 12px 0 8px;
}
.heat-wrap {
    display: flex;
    gap: 14px;
    align-items: center;
    min-height: 420px; /* 与下方占位同高，结果区出现/消失时高度稳定 */
}
.heat-placeholder {
    min-height: 420px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px dashed var(--vp-c-divider);
    border-radius: 8px;
    margin-top: 8px;
    color: var(--vp-c-text-2);
    font-size: 13px;
    text-align: center;
    padding: 12px;
}
.hint {
    margin-top: 10px;
    font-size: 12px;
    color: var(--vp-c-text-3);
}
.colorbar {
    margin-top: 10px;
}
.colorbar canvas {
    width: 100%;
    height: auto;
    border-radius: 6px;
    display: block;
}
.scale {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--vp-c-text-3);
    margin-top: 3px;
}
.heat-wrap canvas {
    border-radius: 8px;
    width: min(100%, 420px);
    height: auto;
    border: 1px solid var(--vp-c-divider);
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
</style>