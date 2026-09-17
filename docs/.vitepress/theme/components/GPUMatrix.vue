<template>
    <div class="gpu-matrix">
        <div class="panel">
            <div class="title">CPU / GPU 矩阵乘法耗时对比</div>
            <div class="fields">
                <label
                    >行数 m
                    <input type="number" v-model.number="mRef" min="1" />
                </label>
                <label
                    >收缩维 n
                    <input type="number" v-model.number="nRef" min="1" />
                </label>
                <label
                    >列数 p
                    <input type="number" v-model.number="pRef" min="1" />
                </label>
                <label
                    >迭代轮数
                    <input type="number" v-model.number="roundsRef" min="1" />
                </label>
                <button class="run" :disabled="computing" @click="run">
                    {{ computing ? '计算中…' : '🚀 开始计算' }}
                </button>
            </div>
        </div>

        <pre class="log">{{ summary || '（计算结果会显示在这里……）' }}</pre>
        <p class="hint">💡 CPU 在 Web Worker 中计算；GPU 用 WebGL2 的 fragment shader 计算，计时含 draw（GPU 计算）+ readPixels（显存→内存复制）。</p>
    </div>
</template>
<script lang="ts" setup>
import { ref } from 'vue';

const shader_source = {
    vertex: `#version 300 es
in vec4 a_position;
void main() {
  gl_Position = a_position;
}`,
    fragment: `#version 300 es
precision highp float;

uniform sampler2D u_A;
uniform sampler2D u_B;
uniform int u_n;
out vec4 outColor;

void main() {
  // result[row][col] = Σ_k A[row][k] * B[k][col]
  int row = int(gl_FragCoord.y);
  int col = int(gl_FragCoord.x);
  float sum = 0.0;
  for (int k = 0; k < u_n; k++) {
    sum += texelFetch(u_A, ivec2(k, row), 0).r
         * texelFetch(u_B, ivec2(col, k), 0).r;
  }
  outColor = vec4(sum, 0.0, 0.0, 1.0);
}`,
};

function compileShader(gl: WebGL2RenderingContext, type: number, source: string) {
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(shader);
        gl.deleteShader(shader);
        throw `着色器编译失败: ${info ?? ''}`;
    }
    return shader;
}

function createProgram(gl: WebGL2RenderingContext, vsSource: string, fsSource: string) {
    const vs = compileShader(gl, gl.VERTEX_SHADER, vsSource);
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, fsSource);
    const program = gl.createProgram()!;
    gl.attachShader(program, vs);
    gl.attachShader(program, fs);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const info = gl.getProgramInfoLog(program);
        throw `程序链接失败: ${info ?? ''}`;
    }
    return program;
}

// 把行优先矩阵 data[rows*cols] 打包成 RGBA32F 纹理（一元素一像素，值放 R 通道）
function uploadMatrixTexture(gl: WebGL2RenderingContext, data: Float32Array, rows: number, cols: number) {
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);

    // RGBA32F：每像素 4 个 32 位浮点通道，这里只用 R 通道存数值
    const pixels = new Float32Array(rows * cols * 4);
    for (let i = 0; i < rows * cols; i++) {
        pixels[i * 4] = data[i];
    }

    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, cols, rows, 0, gl.RGBA, gl.FLOAT, pixels);
    // 必须用 NEAREST：默认的线性过滤会把相邻像素的数值混合掉
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return texture;
}

// CPU 版矩阵乘法，放到 Web Worker 里执行，避免阻塞主线程 UI
const cpuWorkerCode = `self.onmessage = function (e) {
  const A = e.data.A;
  const B = e.data.B;
  const m = e.data.m;
  const n = e.data.n;
  const p = e.data.p;
  const rounds = e.data.rounds;
  const C = new Float32Array(m * p);

  const t0 = performance.now();
  for (let r = 0; r < rounds; r++) {
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < p; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += A[i * n + k] * B[k * p + j];
        }
        C[i * p + j] = sum;
      }
    }
  }
  const cpuMs = performance.now() - t0;

  self.postMessage({ cpuMs: cpuMs, C: C }, [C.buffer]);
};`;

// 在 Worker 中运行 CPU 版矩阵乘法，A、B 用 Transferable 零拷贝传入
function runCpuInWorker(data: { A: Float32Array; B: Float32Array; m: number; n: number; p: number; rounds: number }) {
    return new Promise<{ cpuMs: number; C: Float32Array }>((resolve, reject) => {
        const blob = new Blob([cpuWorkerCode], { type: 'text/javascript' });
        const url = URL.createObjectURL(blob);
        const worker = new Worker(url);

        worker.onmessage = (e: MessageEvent) => {
            resolve(e.data as { cpuMs: number; C: Float32Array });
            worker.terminate();
            URL.revokeObjectURL(url);
        };
        worker.onerror = (e: ErrorEvent) => {
            reject(new Error(e.message || 'Web Worker 运行出错'));
            worker.terminate();
            URL.revokeObjectURL(url);
        };

        worker.postMessage(data, [data.A.buffer, data.B.buffer]);
    });
}

// GPU 版矩阵乘法：C[m×p] = A[m×n] × B[n×p]
// 返回 { step }，step() 执行一次“draw + readPixels”并返回结果矩阵，
// shader/纹理/FBO 的初始化只做一次，便于在计时循环里反复调用。
function createGPUComputer(A: Float32Array, B: Float32Array, m: number, n: number, p: number) {
    const offscreen = new OffscreenCanvas(1, 1);
    const context = offscreen.getContext('webgl2');
    if (context === null) {
        throw `获取WebGL上下文失败，请检查浏览器`;
    }
    // 收窄为非空类型后，闭包 step() 里也能安全访问
    const gl: WebGL2RenderingContext = context;

    // WebGL2 中要把 RGBA32F 浮点纹理当作渲染目标（颜色附件），必须启用该扩展
    if (!gl.getExtension('EXT_color_buffer_float')) {
        throw `当前浏览器不支持 EXT_color_buffer_float，无法渲染到浮点纹理`;
    }

    const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    if (Math.max(m, n, p) > maxTex) {
        throw `矩阵尺寸 ${Math.max(m, n, p)} 超过纹理尺寸上限 ${maxTex}`;
    }

    const program = createProgram(gl, shader_source.vertex, shader_source.fragment);
    gl.useProgram(program);

    // 铺满裁剪空间的四边形（两个三角形、6 个顶点 [x, y, z, w]）
    // prettier-ignore
    const positions = new Float32Array([
        -1, -1, 0, 1,
         1, -1, 0, 1,
         1,  1, 0, 1,
         1,  1, 0, 1,
        -1,  1, 0, 1,
        -1, -1, 0, 1,
    ]);
    const buffer = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const positionLoc = gl.getAttribLocation(program, 'a_position');
    gl.enableVertexAttribArray(positionLoc);
    gl.vertexAttribPointer(positionLoc, 4, gl.FLOAT, false, 0, 0);

    // 上传矩阵 A、B（只做一次）
    const texA = uploadMatrixTexture(gl, A, m, n);
    const texB = uploadMatrixTexture(gl, B, n, p);

    // 输出纹理 C（宽 p、高 m），挂到 framebuffer 上作为渲染目标
    const texC = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texC);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, p, m, 0, gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    const fbo = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texC, 0);
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        throw `Framebuffer 不完整`;
    }

    // 绑定输入纹理与 uniform 参数（只做一次，后续 draw 沿用这些状态）
    const locA = gl.getUniformLocation(program, 'u_A');
    const locB = gl.getUniformLocation(program, 'u_B');
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texA);
    gl.uniform1i(locA, 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, texB);
    gl.uniform1i(locB, 1);
    gl.uniform1i(gl.getUniformLocation(program, 'u_n'), n);

    // 渲染目标：把 C 铺满 viewport
    gl.viewport(0, 0, p, m);

    // 复用读回缓冲区，避免每轮重复分配
    const pixelBuffer = new Float32Array(m * p * 4);

    function step(): Float32Array {
        // readPixels 是同步的，会把之前排队的 draw 都执行完
        gl.drawArrays(gl.TRIANGLES, 0, 6);
        gl.readPixels(0, 0, p, m, gl.RGBA, gl.FLOAT, pixelBuffer);

        const C = new Float32Array(m * p);
        for (let i = 0; i < m * p; i++) {
            C[i] = pixelBuffer[i * 4];
        }
        return C;
    }

    return { step };
}

// ---------------------------------------------------------------------
// 演示数据与 UI
// ---------------------------------------------------------------------
const mRef = ref(200);
const nRef = ref(500);
const pRef = ref(200);
const roundsRef = ref(100);
const summary = ref('');
const computing = ref(false);

function makeRandomMatrix(len: number) {
    const M = new Float32Array(len);
    for (let i = 0; i < len; i++) {
        M[i] = Math.random();
    }
    return M;
}

async function run() {
    const m = mRef.value;
    const n = nRef.value;
    const p = pRef.value;
    const rounds = roundsRef.value;

    if (!Number.isInteger(m) || !Number.isInteger(n) || !Number.isInteger(p) || !Number.isInteger(rounds) || m <= 0 || n <= 0 || p <= 0 || rounds <= 0) {
        summary.value = '请输入正整数。';
        return;
    }

    // 防止计算量过大（Worker 不卡 UI，但仍会吃满 CPU）
    if (m * n * p * rounds > 3e9) {
        summary.value = '计算量过大（m×n×p×轮数 超过 30 亿），请调小参数。';
        return;
    }

    // A、B 将 transfer 给 Worker，GPU 用一份副本
    const A = makeRandomMatrix(m * n);
    const B = makeRandomMatrix(n * p);
    const gpuA = A.slice();
    const gpuB = B.slice();

    computing.value = true;
    summary.value = '计算中……';

    try {
        // 初始化 GPU 资源（shader、纹理上传、FBO），不算入计时
        const gpu = createGPUComputer(gpuA, gpuB, m, n, p);

        // CPU 在 Worker 里异步计算，与 GPU 并行
        const cpuPromise = runCpuInWorker({ A, B, m, n, p, rounds });

        // GPU 计时（每轮含 draw（GPU 计算）+ readPixels（显存→内存复制））
        let gpuResult: Float32Array = new Float32Array(0);
        const gpuStart = performance.now();
        for (let r = 0; r < rounds; r++) {
            gpuResult = gpu.step();
        }
        const gpuMs = performance.now() - gpuStart;

        // 等 Worker 返回 CPU 结果
        const { cpuMs, C: cpuResult } = await cpuPromise;

        // 正确性校验（比较最后一轮结果，用相对误差）
        let maxErr = 0;
        for (let i = 0; i < gpuResult.length; i++) {
            const err = Math.abs(gpuResult[i] - cpuResult[i]) / (1 + Math.abs(cpuResult[i]));
            if (err > maxErr) maxErr = err;
        }
        const match = maxErr < 1e-4;
        const speedup = cpuMs / gpuMs;

        summary.value =
            `矩阵规模: ${m}×${n} × ${n}×${p}   迭代轮数: ${rounds}\n` +
            `结果一致性: ${match ? '一致 ✓' : '不一致 ✗'}（最大相对误差 ${maxErr.toExponential(2)}）\n\n` +
            `CPU 总耗时: ${cpuMs.toFixed(2)} ms（每轮 ${(cpuMs / rounds).toFixed(3)} ms，Web Worker）\n` +
            `GPU 总耗时: ${gpuMs.toFixed(2)} ms（每轮 ${(gpuMs / rounds).toFixed(3)} ms，含 draw（GPU 计算）+ readPixels（显存→内存复制））\n\n` +
            `加速比 CPU/GPU: ${speedup.toFixed(2)}` +
            (speedup >= 1 ? '（GPU 更快）' : '（CPU 反而更快，矩阵再调大些 GPU 优势更明显）');
        computing.value = false;
    } catch (e) {
        summary.value = `计算出错：${e}`;
        computing.value = false;
    }
}
</script>

<style scoped>
.gpu-matrix {
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
    padding: 10px;
    border-radius: 8px;
    background: var(--vp-c-bg);
    border: 1px solid var(--vp-c-divider);
    font-size: 12px;
    line-height: 1.6;
    white-space: pre-wrap;
}
.hint {
    margin-top: 10px;
    font-size: 12px;
    color: var(--vp-c-text-3);
}
</style>
