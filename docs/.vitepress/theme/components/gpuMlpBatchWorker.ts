/* GPU 版 MLP 训练（mini-batch）—— 运行在 Web Worker 中。
 *
 * 与单样本版（gpuMlpWorker.ts）的区别：
 *   把 batch 内样本拼成矩阵 H(n×B)，每层前向/反向/更新都变成矩阵乘，
 *   一次 draw 处理 B 个样本，draw 次数除以 B，单次 draw 计算量乘以 B。
 *
 * 数据布局：矩阵按“一元素一像素”存成 RGBA32F 纹理，数值放 R 通道；
 *   像素 (col, row) ↔ 矩阵元素 [row][col]。
 *   批量数据（激活/敏感度）尺寸为 (B, n)：col=b 是样本下标，row=i 是神经元下标。
 */
export {};

// =====================================================================
// 着色器源码
// =====================================================================
const MAX_N = 64; // 神经元维度上限（对应宽度上限）
const MAX_B = 128; // batch 大小上限

const vertexSrc = `#version 300 es
in vec4 a_position;
void main() {
  gl_Position = a_position;
}`;

// 前向：对每个样本 b、每个输出神经元 i，h[i][b] = sigmoid( Σ_k W[i][k]·h[k][b] + b[i] )
//   输出纹理 (B, nOut)，fragment (col=b, row=i)
const forwardSrc = `#version 300 es
precision highp float;
uniform sampler2D u_W;
uniform sampler2D u_h;
uniform int u_nIn;
out vec4 outColor;
void main() {
  int b = int(gl_FragCoord.x);
  int i = int(gl_FragCoord.y);
  float s = 0.0;
  for (int k = 0; k < ${MAX_N}; k++) {
    if (k >= u_nIn) break;
    s += texelFetch(u_W, ivec2(k, i), 0).r * texelFetch(u_h, ivec2(b, k), 0).r;
  }
  s += texelFetch(u_W, ivec2(u_nIn, i), 0).r;
  float a = 1.0 / (1.0 + exp(-s));
  outColor = vec4(a, 0.0, 0.0, 1.0);
}`;

// 反向敏感度：δ[i][b] = ( Σ_j W[j][i]·δ_next[j][b] ) · σ'(h[i][b])
//   输出纹理 (B, nIn)，fragment (col=b, row=i)
const sensitivitySrc = `#version 300 es
precision highp float;
uniform sampler2D u_W;
uniform sampler2D u_dNext;
uniform sampler2D u_h;
uniform int u_nOut;
out vec4 outColor;
void main() {
  int b = int(gl_FragCoord.x);
  int i = int(gl_FragCoord.y);
  float s = 0.0;
  for (int j = 0; j < ${MAX_N}; j++) {
    if (j >= u_nOut) break;
    s += texelFetch(u_W, ivec2(i, j), 0).r * texelFetch(u_dNext, ivec2(b, j), 0).r;
  }
  float h = texelFetch(u_h, ivec2(b, i), 0).r;
  outColor = vec4(s * h * (1.0 - h), 0.0, 0.0, 1.0);
}`;

// 梯度 + 更新（按 batch 平均）：W_new[j][i] = W_old[j][i] - lr · (1/B)·Σ_b δ[j][b]·h[i][b]
//   输出纹理 (nIn+1, nOut)，fragment (col=i, row=j)
const updateSrc = `#version 300 es
precision highp float;
uniform sampler2D u_Wold;
uniform sampler2D u_d;
uniform sampler2D u_h;
uniform float u_lr;
uniform int u_nIn;
uniform int u_B;
out vec4 outColor;
void main() {
  int i = int(gl_FragCoord.x);
  int j = int(gl_FragCoord.y);
  float s = 0.0;
  for (int b = 0; b < ${MAX_B}; b++) {
    if (b >= u_B) break;
    float h = (i == u_nIn) ? 1.0 : texelFetch(u_h, ivec2(b, i), 0).r;
    s += texelFetch(u_d, ivec2(b, j), 0).r * h;
  }
  float grad = s / float(u_B);
  float w = texelFetch(u_Wold, ivec2(i, j), 0).r - u_lr * grad;
  outColor = vec4(w, 0.0, 0.0, 1.0);
}`;

// 输出层敏感度（GPU 内计算）：δ_out[b] = a[b] - y[b]
//   输出纹理 (B, 1)
const outputDeltaSrc = `#version 300 es
precision highp float;
uniform sampler2D u_a;
uniform sampler2D u_y;
out vec4 outColor;
void main() {
  int b = int(gl_FragCoord.x);
  float a = texelFetch(u_a, ivec2(b, 0), 0).r;
  float y = texelFetch(u_y, ivec2(b, 0), 0).r;
  outColor = vec4(a - y, 0.0, 0.0, 1.0);
}`;

// =====================================================================
// WebGL 工具
// =====================================================================
function compileShader(gl: WebGL2RenderingContext, type: number, source: string) {
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(shader);
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

function createTex(gl: WebGL2RenderingContext, w: number, h: number, data?: Float32Array) {
    const tex = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, data ?? null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
}

/** 把 B 个样本、每个 n 维的数据打包成 (B, n) 纹理的 RGBA 数据（值放 R 通道） */
function packBatch(values: Float32Array, B: number, n: number) {
    const data = new Float32Array(B * n * 4);
    for (let i = 0; i < n; i++) {
        for (let b = 0; b < B; b++) {
            data[(i * B + b) * 4] = values[b * n + i];
        }
    }
    return data;
}

/** 把 n_out × (n_in+1) 的权重矩阵（含最后一列偏置）打包成纹理数据 */
function packWeights(nIn: number, nOut: number, random: () => number) {
    const cols = nIn + 1;
    const data = new Float32Array(nOut * cols * 4);
    for (let j = 0; j < nOut; j++) {
        for (let i = 0; i <= nIn; i++) {
            const v = i === nIn ? 0 : random();
            data[(j * cols + i) * 4] = v;
        }
    }
    return data;
}

type EngineProgram = { forward: WebGLProgram; sensitivity: WebGLProgram; update: WebGLProgram; outputDelta: WebGLProgram };

interface WeightLayer {
    nIn: number;
    nOut: number;
    a: WebGLTexture;
    b: WebGLTexture;
    cur: 0 | 1;
}

// =====================================================================
// 训练引擎（mini-batch）
// =====================================================================
function buildEngine(
    gl: WebGL2RenderingContext,
    fbo: WebGLFramebuffer,
    quadVBO: WebGLBuffer,
    size: number[],
    maxBatch: number,
) {
    const programs: EngineProgram = {
        forward: createProgram(gl, vertexSrc, forwardSrc),
        sensitivity: createProgram(gl, vertexSrc, sensitivitySrc),
        update: createProgram(gl, vertexSrc, updateSrc),
        outputDelta: createProgram(gl, vertexSrc, outputDeltaSrc),
    };

    const attrReady = new Set<WebGLProgram>();

    function useProgram(program: WebGLProgram) {
        gl.useProgram(program);
        if (!attrReady.has(program)) {
            const loc = gl.getAttribLocation(program, 'a_position');
            gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
            gl.enableVertexAttribArray(loc);
            gl.vertexAttribPointer(loc, 4, gl.FLOAT, false, 0, 0);
            attrReady.add(program);
        }
    }

    function renderTo(program: WebGLProgram, outTex: WebGLTexture, outW: number, outH: number, setup: () => void) {
        useProgram(program);
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outTex, 0);
        gl.viewport(0, 0, outW, outH);
        setup();
        gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    const L = size.length - 1;

    // 权重（含偏置列），双缓冲
    const weights: WeightLayer[] = [];
    for (let l = 0; l < L; l++) {
        const nIn = size[l];
        const nOut = size[l + 1];
        const data = packWeights(nIn, nOut, () => Math.random() * 0.4 - 0.2);
        weights.push({ nIn, nOut, a: createTex(gl, nIn + 1, nOut, data), b: createTex(gl, nIn + 1, nOut, data), cur: 0 });
    }

    // 激活 H[l]：(maxBatch, size[l])，敏感度 Δ[l] 同型；标签 y：(maxBatch, 1)
    const acts: WebGLTexture[] = size.map((n) => createTex(gl, maxBatch, n));
    const deltas: WebGLTexture[] = size.map((n) => createTex(gl, maxBatch, n));
    const yTex = createTex(gl, maxBatch, 1);

    // 上传一个 batch 的输入（B 个样本 × 2 维）
    function setInput(values: Float32Array, B: number) {
        const data = packBatch(values, B, size[0]);
        gl.bindTexture(gl.TEXTURE_2D, acts[0]);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, B, size[0], gl.RGBA, gl.FLOAT, data);
        gl.bindTexture(gl.TEXTURE_2D, null);
    }

    // 上传标签（B 个）
    function setLabels(values: Float32Array, B: number) {
        const data = packBatch(values, B, 1);
        gl.bindTexture(gl.TEXTURE_2D, yTex);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, B, 1, gl.RGBA, gl.FLOAT, data);
        gl.bindTexture(gl.TEXTURE_2D, null);
    }

    function curTex(l: number) {
        const w = weights[l];
        return w.cur === 0 ? w.a : w.b;
    }

    function forwardLayer(l: number, B: number) {
        const w = weights[l];
        const locW = gl.getUniformLocation(programs.forward, 'u_W');
        const locH = gl.getUniformLocation(programs.forward, 'u_h');
        const locN = gl.getUniformLocation(programs.forward, 'u_nIn');
        renderTo(programs.forward, acts[l + 1], B, size[l + 1], () => {
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, curTex(l));
            gl.uniform1i(locW, 0);
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, acts[l]);
            gl.uniform1i(locH, 1);
            gl.uniform1i(locN, w.nIn);
        });
    }

    function sensitivityLayer(l: number, B: number) {
        const w = weights[l];
        const locW = gl.getUniformLocation(programs.sensitivity, 'u_W');
        const locD = gl.getUniformLocation(programs.sensitivity, 'u_dNext');
        const locH = gl.getUniformLocation(programs.sensitivity, 'u_h');
        const locN = gl.getUniformLocation(programs.sensitivity, 'u_nOut');
        renderTo(programs.sensitivity, deltas[l], B, size[l], () => {
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, curTex(l));
            gl.uniform1i(locW, 0);
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, deltas[l + 1]);
            gl.uniform1i(locD, 1);
            gl.activeTexture(gl.TEXTURE2);
            gl.bindTexture(gl.TEXTURE_2D, acts[l]);
            gl.uniform1i(locH, 2);
            gl.uniform1i(locN, w.nOut);
        });
    }

    function updateLayer(l: number, lr: number, B: number) {
        const w = weights[l];
        const src = w.cur === 0 ? w.a : w.b;
        const dst = w.cur === 0 ? w.b : w.a;
        const locW = gl.getUniformLocation(programs.update, 'u_Wold');
        const locD = gl.getUniformLocation(programs.update, 'u_d');
        const locH = gl.getUniformLocation(programs.update, 'u_h');
        const locLr = gl.getUniformLocation(programs.update, 'u_lr');
        const locN = gl.getUniformLocation(programs.update, 'u_nIn');
        const locB = gl.getUniformLocation(programs.update, 'u_B');
        renderTo(programs.update, dst, w.nIn + 1, w.nOut, () => {
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, src);
            gl.uniform1i(locW, 0);
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, deltas[l + 1]);
            gl.uniform1i(locD, 1);
            gl.activeTexture(gl.TEXTURE2);
            gl.bindTexture(gl.TEXTURE_2D, acts[l]);
            gl.uniform1i(locH, 2);
            gl.uniform1f(locLr, lr);
            gl.uniform1i(locN, w.nIn);
            gl.uniform1i(locB, B);
        });
        w.cur = w.cur === 0 ? 1 : 0;
    }

    function computeOutputDelta(B: number) {
        const locA = gl.getUniformLocation(programs.outputDelta, 'u_a');
        const locY = gl.getUniformLocation(programs.outputDelta, 'u_y');
        renderTo(programs.outputDelta, deltas[L], B, 1, () => {
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, acts[L]);
            gl.uniform1i(locA, 0);
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, yTex);
            gl.uniform1i(locY, 1);
        });
    }

    // 读回输出层激活（B 个标量）
    function readOutputs(B: number): Float32Array {
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, acts[L], 0);
        const buf = new Float32Array(B * 4);
        gl.readPixels(0, 0, B, 1, gl.RGBA, gl.FLOAT, buf);
        const out = new Float32Array(B);
        for (let b = 0; b < B; b++) out[b] = buf[b * 4];
        return out;
    }

    return { setInput, setLabels, computeOutputDelta, readOutputs, forwardLayer, sensitivityLayer, updateLayer };
}

// =====================================================================
// 数据生成（与第五章一致）
// =====================================================================
function ringSample(minR: number, maxR: number): { x: number; y: number; r: number } {
    const r = minR + Math.random() * (maxR - minR);
    const theta = Math.random() * 2 * Math.PI;
    return { x: r * Math.cos(theta), y: r * Math.sin(theta), r };
}

function dataGen(r1: number, r2: number, count: number): { x: number; y: number; label: number }[] {
    const range = r2 + 0.5;
    const data: { x: number; y: number; label: number }[] = [];
    const perClass = Math.ceil(count / 3);
    for (let i = 0; i < perClass; i++) {
        const inner = ringSample(0, r1);
        data.push({ x: inner.x, y: inner.y, label: 0 });
        const mid = ringSample(r1, r2);
        const p = (mid.r - r1) / (r2 - r1);
        data.push({ x: mid.x, y: mid.y, label: Math.random() < p ? 1 : 0 });
        const outer = ringSample(r2, range);
        data.push({ x: outer.x, y: outer.y, label: 1 });
    }
    return data.slice(0, count);
}

// =====================================================================
// Worker 入口
// =====================================================================
interface TrainRequest {
    type: 'train';
    value?: { width: number; depth: number; samples: number; lr: number; epochs: number; r1: number; r2: number; batch: number };
}

self.addEventListener('message', (e: MessageEvent<TrainRequest>) => {
    const req = e.data;
    if (!req || req.type !== 'train' || !req.value) return;
    const { width, depth, samples, lr, epochs, r1, r2, batch } = req.value;
    const post = (msg: unknown) => (self as unknown as { postMessage(m: unknown): void }).postMessage(msg);

    try {
        const offscreen = new OffscreenCanvas(1, 1);
        const gl = offscreen.getContext('webgl2');
        if (gl === null) throw `获取WebGL上下文失败`;
        if (!gl.getExtension('EXT_color_buffer_float')) {
            throw `当前浏览器不支持 EXT_color_buffer_float，无法渲染到浮点纹理`;
        }

        // prettier-ignore
        const positions = new Float32Array([
            -1, -1, 0, 1,
             1, -1, 0, 1,
             1,  1, 0, 1,
             1,  1, 0, 1,
            -1,  1, 0, 1,
            -1, -1, 0, 1,
        ]);
        const quadVBO = gl.createBuffer()!;
        gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
        gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

        const fbo = gl.createFramebuffer()!;
        const size = [2, ...Array.from({ length: depth }, () => width), 1];
        const engine = buildEngine(gl, fbo, quadVBO, size, batch);

        const data = dataGen(r1, r2, samples);
        const L = size.length - 1;
        post({ type: 'info', text: `GPU dense(${width}, ${depth})  batch ${batch}  样本 ${samples}  lr ${lr}  epochs ${epochs}` });

        const xs = new Float32Array(batch * 2);
        const ys = new Float32Array(batch);

        const t0 = performance.now();
        for (let epoch = 0; epoch < epochs; epoch++) {
            for (let i = data.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [data[i], data[j]] = [data[j], data[i]];
            }

            for (let start = 0; start < data.length; start += batch) {
                const B = Math.min(batch, data.length - start); // 最后一个 batch 可能不满
                for (let b = 0; b < B; b++) {
                    const s = data[start + b];
                    xs[b * 2] = s.x;
                    xs[b * 2 + 1] = s.y;
                    ys[b] = s.label;
                }

                engine.setInput(xs, B);
                engine.setLabels(ys, B);
                for (let l = 0; l < L; l++) engine.forwardLayer(l, B);
                engine.computeOutputDelta(B);
                for (let l = L - 1; l >= 1; l--) engine.sensitivityLayer(l, B);
                for (let l = 0; l < L; l++) engine.updateLayer(l, lr, B);
            }
            // 训练过程零 readPixels
        }
        const trainMs = performance.now() - t0;

        // 训练结束，生成热力图（网格点也按 batch 批量前向，减少 readPixels 次数）
        const grid = 60;
        const span = 3.2;
        const total = grid * grid;
        const heatmap: number[] = new Array(total);
        for (let bi = 0; bi * batch < total; bi++) {
            const start = bi * batch;
            const B = Math.min(batch, total - start);
            for (let b = 0; b < B; b++) {
                const idx = start + b;
                const col = idx % grid;
                const row = Math.floor(idx / grid);
                xs[b * 2] = -span / 2 + (col + 0.5) * (span / grid);
                xs[b * 2 + 1] = span / 2 - (row + 0.5) * (span / grid);
            }
            engine.setInput(xs, B);
            for (let l = 0; l < L; l++) engine.forwardLayer(l, B);
            const outs = engine.readOutputs(B);
            for (let b = 0; b < B; b++) heatmap[start + b] = outs[b];
        }

        post({ type: 'done', trainMs, heatmap, grid });
    } catch (err) {
        post({ type: 'error', message: String(err) });
    }
});