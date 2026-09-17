/* GPU 版 MLP 训练 —— 运行在 Web Worker 中。
 *
 * 训练信号：严格单样本 SGD（和第五章 GraphDemo 的语义完全一致），
 *   逐样本：前向(逐层) → 反向(逐层) → 逐样本更新权重。
 * 区别：每一层的矩阵运算/激活/梯度更新都交给 WebGL2 的 fragment shader 完成，
 *   中间结果（激活 h、敏感度 δ）全部留在 GPU 纹理里，只有每个 epoch 的损失
 *   这类标量才 readPixels 读回。
 *
 * 数据布局（沿用 GPUMatrix 的约定）：矩阵按“一元素一像素”存成 RGBA32F 纹理，
 *   数值放 R 通道；像素 (col, row) ↔ 矩阵元素 [row][col]。
 */
export {};

// =====================================================================
// 着色器源码（固定循环上界 64，对应 GraphDemo 的宽度上限）
// =====================================================================
const MAX_N = 64;

const vertexSrc = `#version 300 es
in vec4 a_position;
void main() {
  gl_Position = a_position;
}`;

// 前向：h_next[i] = sigmoid( Σ_k W[i][k]·h[k] + b[i] )
//   u_W : 权重（含偏置列），尺寸 (nIn+1) × nOut，W[j][i] 在像素 (i, j)
//   u_h : 输入激活，尺寸 1 × nIn
//   输出纹理：1 × nOut
const forwardSrc = `#version 300 es
precision highp float;
uniform sampler2D u_W;
uniform sampler2D u_h;
uniform int u_nIn;
out vec4 outColor;
void main() {
  int i = int(gl_FragCoord.y); // 输出神经元下标
  float s = 0.0;
  for (int k = 0; k < ${MAX_N}; k++) {
    if (k >= u_nIn) break;
    s += texelFetch(u_W, ivec2(k, i), 0).r * texelFetch(u_h, ivec2(0, k), 0).r;
  }
  s += texelFetch(u_W, ivec2(u_nIn, i), 0).r; // 偏置列（最后一列）× 1
  float a = 1.0 / (1.0 + exp(-s));
  outColor = vec4(a, 0.0, 0.0, 1.0);
}`;

// 反向敏感度：δ[i] = ( Σ_j W[j][i]·δ_next[j] ) · σ'(h[i])
//   u_W     : 本层权重（含偏置列），尺寸 (nIn+1) × nOut
//   u_dNext : 下一层敏感度，尺寸 1 × nOut
//   u_h     : 本层激活，尺寸 1 × nIn（σ'(h[i]) = h[i]·(1-h[i])）
//   输出纹理：1 × nIn
const sensitivitySrc = `#version 300 es
precision highp float;
uniform sampler2D u_W;
uniform sampler2D u_dNext;
uniform sampler2D u_h;
uniform int u_nOut;
out vec4 outColor;
void main() {
  int i = int(gl_FragCoord.y); // 本层节点下标
  float s = 0.0;
  for (int j = 0; j < ${MAX_N}; j++) {
    if (j >= u_nOut) break;
    s += texelFetch(u_W, ivec2(i, j), 0).r * texelFetch(u_dNext, ivec2(0, j), 0).r;
  }
  float h = texelFetch(u_h, ivec2(0, i), 0).r;
  s *= h * (1.0 - h);
  outColor = vec4(s, 0.0, 0.0, 1.0);
}`;

// 梯度计算 + 参数更新：W_new[j][i] = W_old[j][i] - lr · δ[j] · h[i]
//   u_Wold : 旧权重（含偏置列），尺寸 (nIn+1) × nOut
//   u_d    : 本层输出侧敏感度，尺寸 1 × nOut
//   u_h    : 本层输入侧激活，尺寸 1 × nIn（偏置列 i==nIn 时取 1）
//   输出纹理：(nIn+1) × nOut
const updateSrc = `#version 300 es
precision highp float;
uniform sampler2D u_Wold;
uniform sampler2D u_d;
uniform sampler2D u_h;
uniform float u_lr;
uniform int u_nIn;
out vec4 outColor;
void main() {
  int i = int(gl_FragCoord.x); // 权重列下标（0..nIn，nIn 为偏置列）
  int j = int(gl_FragCoord.y); // 输出神经元下标
  float h = (i == u_nIn) ? 1.0 : texelFetch(u_h, ivec2(0, i), 0).r;
  float grad = texelFetch(u_d, ivec2(0, j), 0).r * h;
  float w = texelFetch(u_Wold, ivec2(i, j), 0).r - u_lr * grad;
  outColor = vec4(w, 0.0, 0.0, 1.0);
}`;

// 输出层敏感度：δ_out = a - y（在 GPU 内计算，不回读 a）
//   u_a : 输出激活，尺寸 1 × 1
//   输出纹理：1 × 1
const outputDeltaSrc = `#version 300 es
precision highp float;
uniform sampler2D u_a;
uniform float u_y;
out vec4 outColor;
void main() {
  float a = texelFetch(u_a, ivec2(0, 0), 0).r;
  outColor = vec4(a - u_y, 0.0, 0.0, 1.0);
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

/** 创建 RGBA32F 纹理（data 为整套 RGBA 打包数据，可空表示只分配） */
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

/** 把 n 个标量打包成 1×n 纹理的 RGBA 数据（值放 R 通道） */
function packVec(values: Float32Array) {
    const data = new Float32Array(values.length * 4);
    for (let i = 0; i < values.length; i++) data[i * 4] = values[i];
    return data;
}

/** 把 n_out × (n_in+1) 的权重矩阵（行优先，含最后一列偏置）打包成纹理数据 */
function packWeights(nIn: number, nOut: number, random: () => number) {
    const cols = nIn + 1;
    const data = new Float32Array(nOut * cols * 4);
    for (let j = 0; j < nOut; j++) {
        for (let i = 0; i <= nIn; i++) {
            // 偏置列初始化为 0，其余 [-0.2, 0.2)
            const v = i === nIn ? 0 : random();
            const idx = (j * cols + i) * 4;
            data[idx] = v;
        }
    }
    return data;
}

type EngineProgram = { forward: WebGLProgram; sensitivity: WebGLProgram; update: WebGLProgram; outputDelta: WebGLProgram };

interface WeightLayer {
    nIn: number;
    nOut: number;
    a: WebGLTexture; // 双缓冲 A
    b: WebGLTexture; // 双缓冲 B
    cur: 0 | 1;
}

// =====================================================================
// 训练引擎
// =====================================================================
function buildEngine(
    gl: WebGL2RenderingContext,
    fbo: WebGLFramebuffer,
    quadVBO: WebGLBuffer,
    size: number[],
) {
    const programs: EngineProgram = {
        forward: createProgram(gl, vertexSrc, forwardSrc),
        sensitivity: createProgram(gl, vertexSrc, sensitivitySrc),
        update: createProgram(gl, vertexSrc, updateSrc),
        outputDelta: createProgram(gl, vertexSrc, outputDeltaSrc),
    };

    const attrReady = new Set<WebGLProgram>();

    /** 让 program 生效，并（首次）把全屏四边形 VBO 接上 a_position */
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

    /** 渲染到一个纹理：绑定 FBO → viewport → 设置输入 → draw */
    function renderTo(program: WebGLProgram, outTex: WebGLTexture, outW: number, outH: number, setup: () => void) {
        useProgram(program);
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, outTex, 0);
        gl.viewport(0, 0, outW, outH);
        setup();
        gl.drawArrays(gl.TRIANGLES, 0, 6);
    }

    // ---- 构建网络结构 ----
    // size = [2, width, ..., width, 1]，共 depth+2 个节点层
    const L = size.length - 1; // 权重层数 = depth + 1

    // 每层权重（含偏置列），双缓冲，初始相同随机器
    const weights: WeightLayer[] = [];
    for (let l = 0; l < L; l++) {
        const nIn = size[l];
        const nOut = size[l + 1];
        const data = packWeights(nIn, nOut, () => Math.random() * 0.4 - 0.2);
        weights.push({
            nIn,
            nOut,
            a: createTex(gl, nIn + 1, nOut, data),
            b: createTex(gl, nIn + 1, nOut, data),
            cur: 0,
        });
    }

    // 激活 h[l]：节点层 l 的输出，尺寸 1 × size[l]
    const acts: WebGLTexture[] = size.map((n) => createTex(gl, 1, n));
    // 敏感度 δ[l]：节点层 l 的敏感度，尺寸 1 × size[l]
    const deltas: WebGLTexture[] = size.map((n) => createTex(gl, 1, n));

    // 上传 h[0]（输入，2 维），由训练循环每样本更新
    function setInput(x: number, y: number) {
        const data = packVec(new Float32Array([x, y]));
        gl.bindTexture(gl.TEXTURE_2D, acts[0]);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 1, size[0], gl.RGBA, gl.FLOAT, data);
        gl.bindTexture(gl.TEXTURE_2D, null);
    }

    // 输出层敏感度直接在 GPU 内计算：δ_out = a - y（不读回 a）
    function computeOutputDelta(y: number) {
        const locA = gl.getUniformLocation(programs.outputDelta, 'u_a');
        const locY = gl.getUniformLocation(programs.outputDelta, 'u_y');
        renderTo(programs.outputDelta, deltas[L], 1, 1, () => {
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, acts[L]);
            gl.uniform1i(locA, 0);
            gl.uniform1f(locY, y);
        });
    }

    // 读回输出层激活 h[L]（1×1 纹理的 R 通道）
    const readBuf = new Float32Array(4);
    function readOutput(): number {
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, acts[L], 0);
        gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, readBuf);
        return readBuf[0];
    }

    function curTex(l: number) {
        const w = weights[l];
        return w.cur === 0 ? w.a : w.b;
    }

    // 前向：h[l+1] = sigmoid(W[l]·h_aug[l])
    function forwardLayer(l: number) {
        const w = weights[l];
        const locW = gl.getUniformLocation(programs.forward, 'u_W');
        const locH = gl.getUniformLocation(programs.forward, 'u_h');
        const locN = gl.getUniformLocation(programs.forward, 'u_nIn');
        renderTo(programs.forward, acts[l + 1], 1, size[l + 1], () => {
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, curTex(l));
            gl.uniform1i(locW, 0);
            gl.activeTexture(gl.TEXTURE1);
            gl.bindTexture(gl.TEXTURE_2D, acts[l]);
            gl.uniform1i(locH, 1);
            gl.uniform1i(locN, w.nIn);
        });
    }

    // 反向敏感度：δ[l] = (W[l]ᵀ·δ[l+1]) ⊙ σ'(h[l])
    function sensitivityLayer(l: number) {
        const w = weights[l];
        const locW = gl.getUniformLocation(programs.sensitivity, 'u_W');
        const locD = gl.getUniformLocation(programs.sensitivity, 'u_dNext');
        const locH = gl.getUniformLocation(programs.sensitivity, 'u_h');
        const locN = gl.getUniformLocation(programs.sensitivity, 'u_nOut');
        renderTo(programs.sensitivity, deltas[l], 1, size[l], () => {
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

    // 更新：W[l] -= lr · δ[l+1]·h[l]ᵀ（含偏置列），读旧写新，之后交换双缓冲
    function updateLayer(l: number, lr: number) {
        const w = weights[l];
        const src = w.cur === 0 ? w.a : w.b;
        const dst = w.cur === 0 ? w.b : w.a;
        const locW = gl.getUniformLocation(programs.update, 'u_Wold');
        const locD = gl.getUniformLocation(programs.update, 'u_d');
        const locH = gl.getUniformLocation(programs.update, 'u_h');
        const locLr = gl.getUniformLocation(programs.update, 'u_lr');
        const locN = gl.getUniformLocation(programs.update, 'u_nIn');
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
        });
        w.cur = w.cur === 0 ? 1 : 0;
    }

    return { setInput, computeOutputDelta, readOutput, forwardLayer, sensitivityLayer, updateLayer };
}

// =====================================================================
// 数据生成（与第五章 GraphDemo 一致：圆内 / 模糊带 / 圆外 各约 1/3）
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
    value?: { width: number; depth: number; samples: number; lr: number; epochs: number; r1: number; r2: number };
}

self.addEventListener('message', (e: MessageEvent<TrainRequest>) => {
    const req = e.data;
    if (!req || req.type !== 'train' || !req.value) return;
    const { width, depth, samples, lr, epochs, r1, r2 } = req.value;
    const post = (msg: unknown) => (self as unknown as { postMessage(m: unknown): void }).postMessage(msg);

    try {
        const offscreen = new OffscreenCanvas(1, 1);
        const gl = offscreen.getContext('webgl2');
        if (gl === null) throw `获取WebGL上下文失败`;

        // RGBA32F 渲染到浮点纹理需要该扩展
        if (!gl.getExtension('EXT_color_buffer_float')) {
            throw `当前浏览器不支持 EXT_color_buffer_float，无法渲染到浮点纹理`;
        }

        // 全屏四边形（两个三角形）
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

        // 节点层尺寸：[2, width, ..., width, 1]
        const size = [2, ...Array.from({ length: depth }, () => width), 1];
        const engine = buildEngine(gl, fbo, quadVBO, size);

        const data = dataGen(r1, r2, samples);
        post({ type: 'info', text: `GPU dense(${width}, ${depth})  样本 ${samples}  lr ${lr}  epochs ${epochs}` });

        const L = size.length - 1;
        const t0 = performance.now();
        for (let epoch = 0; epoch < epochs; epoch++) {
            // 打乱样本顺序
            for (let i = data.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [data[i], data[j]] = [data[j], data[i]];
            }

            for (const s of data) {
                engine.setInput(s.x, s.y); // ① 输入（唯一一次 CPU→GPU 传数据）

                for (let l = 0; l < L; l++) engine.forwardLayer(l); // ② 前向逐层

                engine.computeOutputDelta(s.label); // 输出层敏感度 δ = a - y（GPU 内计算）

                for (let l = L - 1; l >= 1; l--) engine.sensitivityLayer(l); // ③ 反向逐层求敏感度
                for (let l = 0; l < L; l++) engine.updateLayer(l, lr); // ④ 逐层更新
            }
            // 训练过程完全不从 GPU 读数据，不打印中间损失
        }
        const trainMs = performance.now() - t0;

        // 训练结束，生成热力图（逐点前向 + 读回输出概率）
        const grid = 60;
        const span = 3.2;
        const heatmap: number[] = [];
        for (let row = 0; row < grid; row++) {
            for (let col = 0; col < grid; col++) {
                const x = -span / 2 + (col + 0.5) * (span / grid);
                const y = span / 2 - (row + 0.5) * (span / grid);
                engine.setInput(x, y);
                for (let l = 0; l < L; l++) engine.forwardLayer(l);
                heatmap.push(engine.readOutput());
            }
        }

        post({ type: 'done', trainMs, heatmap, grid });
    } catch (err) {
        post({ type: 'error', message: String(err) });
    }
});