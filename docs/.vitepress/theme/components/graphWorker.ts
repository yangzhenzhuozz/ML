/* 计算图核心 + 训练循环 —— 运行在 Web Worker 中，避免训练重计算阻塞主线程（UI 卡死） */
export {}; // 标记为 ES 模块：把顶层声明隔离在模块作用域，避免它们被当作全局脚本污染全局

interface Parameter {
    value: number; // 参数值
    grad: number; // 参数梯度
    partial: (node: GNode) => number; // 局部偏导 ∂z/∂w
}

let global_batch = 0;

class GNode {
    public next: { partial: () => number; node: GNode }[] = [];
    public previous: GNode[] = [];
    public parameters: Parameter[] = [];
    public partial_h = () => 0; // 激活函数导数 ∂h/∂z

    public z: (node: GNode) => number = () => 0;
    public active: (node: GNode) => number = () => 0;

    public active_cache_batch = -1;
    public active_cache: number | undefined = undefined;
    public delta_cache: number | undefined = undefined;
    public delta_cache_batch = -1; // 敏感度的独立批次标记

    // 敏感度 δ = (Σ δ_i · ∂z_i/∂h) · ∂h/∂z
    public delta() {
        if (this.delta_cache_batch == global_batch && this.delta_cache != undefined) {
            return this.delta_cache;
        }
        let sum_partial = 0;
        for (const next of this.next) {
            sum_partial += next.node.delta() * next.partial();
        }
        this.delta_cache = sum_partial * this.partial_h();
        this.delta_cache_batch = global_batch;
        return this.delta_cache;
    }

    // 反向传播：更新本节点梯度，再沿 previous 向上倒推
    public backPropagation() {
        for (const p of this.parameters) {
            p.grad = this.delta() * p.partial(this);
        }
        for (const p of this.previous) {
            p.backPropagation();
        }
    }
}

class Graph {
    public inputNodes: GNode[] = [];
    public outputNodes: GNode[] = [];
    private trainableNodes: GNode[] = [];

    // 由邻接矩阵自动构造：A[i][j] = 1 ⇔ 边 i → j（入度 0 → 输入；出度 0 → 输出）
    public constructor(adj: number[][]) {
        const n = adj.length;
        const nodes: GNode[] = Array.from({ length: n }, () => new GNode());

        const prevOf: GNode[][] = nodes.map((_, j) => {
            const prev: GNode[] = [];
            for (let i = 0; i < n; i++) if (adj[i][j]) prev.push(nodes[i]);
            return prev;
        });

        this.inputNodes = nodes.filter((_, j) => prevOf[j].length === 0);
        this.outputNodes = nodes.filter((_, j) => adj[j].every((v) => v === 0));
        if (this.outputNodes.length === 0) throw '没有输出节点';

        nodes.forEach((_, j) => {
            if (prevOf[j].length > 0) this.setupNode(nodes[j], prevOf[j]);
        });
        if (this.hasCycle(nodes)) throw '当前计算图存在环';
        this.trainableNodes = nodes.filter((node) => node.parameters.length > 0);
    }

    // “2输入 → depth 层全连接隐藏(width) → 1输出”的邻接矩阵
    public static dense(width: number, depth: number): number[][] {
        const layers = [2, ...Array.from({ length: depth }, () => width), 1];
        const total = layers.reduce((a, b) => a + b, 0);
        const adj = Array.from({ length: total }, () => Array<number>(total).fill(0));
        let offset = 0;
        for (let L = 0; L < layers.length - 1; L++) {
            const from = offset;
            const fromCount = layers[L];
            offset += fromCount;
            for (let a = 0; a < fromCount; a++) {
                for (let b = 0; b < layers[L + 1]; b++) {
                    adj[from + a][offset + b] = 1;
                }
            }
        }
        return adj;
    }

    private z(node: GNode): number {
        let ret = 0;
        for (let i = 0; i < node.previous.length; i++) {
            const prev = node.previous[i];
            ret += node.parameters[i].value * prev.active(prev);
        }
        ret += node.parameters[node.parameters.length - 1].value;
        return ret;
    }

    private active(node: GNode): number {
        if (node.active_cache_batch == global_batch && node.active_cache != undefined) {
            return node.active_cache;
        }
        node.active_cache_batch = global_batch;
        let ret = 1 / (1 + Math.pow(Math.E, 0 - node.z(node)));
        node.active_cache = ret;
        return node.active_cache;
    }

    private setupNode(node: GNode, prev: GNode[]): void {
        const weights: Parameter[] = prev.map((src) => ({
            value: Math.random() * 0.4 - 0.2,
            grad: 0,
            partial: () => src.active(src),
        }));
        const bias: Parameter = { value: 0, grad: 0, partial: () => 1 };

        node.parameters = [...weights, bias];
        node.previous = prev;
        node.z = (n) => this.z(n);
        node.active = (n) => this.active(n);

        // 输出节点的 δ 由损失梯度注入（δ_O = a - y），不需再设置激活导数
        if (!this.outputNodes.includes(node)) {
            node.partial_h = () => node.active_cache! * (1 - node.active_cache!);
        }
        prev.forEach((src, i) => {
            src.next.push({ node, partial: () => weights[i].value });
        });
    }

    public forwardPropagation(input: [number, number]) {
        global_batch++;
        this.inputNodes[0].active = () => input[0];
        this.inputNodes[1].active = () => input[1];
        this.outputNodes[0].active(this.outputNodes[0]);
    }

    public backPropagation(target: number) {
        const out = this.outputNodes[0];
        out.delta_cache = out.active_cache! - target;
        out.delta_cache_batch = global_batch;
        out.backPropagation();
    }

    public updateParameters(lr: number) {
        for (const node of this.trainableNodes) {
            for (const p of node.parameters) {
                p.value -= lr * p.grad;
            }
        }
    }

    public hasCycle(nodes: GNode[]): boolean {
        const state = new Map<GNode, number>();
        for (const node of nodes) state.set(node, 0);
        const dfs = (node: GNode): boolean => {
            const s = state.get(node)!;
            if (s === 2) return false;
            if (s === 1) return true;
            state.set(node, 1);
            for (const edge of node.next) if (dfs(edge.node)) return true;
            state.set(node, 2);
            return false;
        };
        for (const node of nodes) if (dfs(node)) return true;
        return false;
    }
}

/** 极坐标分层采样：圆内 / 模糊带 / 圆外 各取约 1/3 */
function ringSample(minR: number, maxR: number): { x: number; y: number; r: number } {
    const r = minR + Math.random() * (maxR - minR);
    const theta = Math.random() * 2 * Math.PI;
    return { x: r * Math.cos(theta), y: r * Math.sin(theta), r };
}

function dataGen(r1: number, r2: number, count = 800): { x: number; y: number; label: number }[] {
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

/* ===================== Worker 入口 ===================== */

interface WorkerMessage {
    type: 'train';
    value?: { width: number; depth: number; samples: number; lr: number; epochs: number; r1: number; r2: number };
}

self.addEventListener('message', (e: MessageEvent<WorkerMessage>) => {
    const req = e.data;
    if (!req || req.type !== 'train' || !req.value) return;
    const { width, depth, samples, lr, epochs, r1, r2 } = req.value;
    const post = (msg: unknown) => (self as unknown as { postMessage(m: unknown): void }).postMessage(msg);

    try {
        const graph = new Graph(Graph.dense(width, depth));
        const data = dataGen(r1, r2, samples);
        post({ type: 'progress', text: `网络 dense(${width}, ${depth})  样本 ${samples}  lr ${lr}  epochs ${epochs}\n` });

        const chunk = Math.max(1, Math.floor(epochs / 40)); // 约 40 条进度日志
        const t0 = performance.now(); // 训练计时起点
        for (let epoch = 0; epoch < epochs; epoch++) {
            // 打乱样本顺序
            for (let i = data.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [data[i], data[j]] = [data[j], data[i]];
            }
            let epochLoss = 0;
            for (const s of data) {
                graph.forwardPropagation([s.x, s.y]); // ① 前向
                const h = graph.outputNodes[0].active_cache!;
                epochLoss += -(s.label * Math.log(h + 1e-9) + (1 - s.label) * Math.log(1 - h + 1e-9));
                graph.backPropagation(s.label); // ② 反向求梯度
                graph.updateParameters(lr); // ③ 梯度下降更新
            }
            if (epoch % chunk === chunk - 1 || epoch === epochs - 1) {
                post({ type: 'progress', text: `epoch ${String(epoch).padStart(4)}  平均损失 = ${(epochLoss / data.length).toFixed(4)}\n` });
            }
        }
        const trainMs = performance.now() - t0; // 纯训练循环耗时

        // 用一组新样本评估准确率
        const test = dataGen(r1, r2, samples);
        let correct = 0;
        for (const s of test) {
            graph.forwardPropagation([s.x, s.y]);
            if ((graph.outputNodes[0].active_cache! >= 0.5 ? 1 : 0) === s.label) correct++;
        }
        const acc = (correct / test.length) * 100;

        // 计算热力图概率数组（60×60，行优先），主线程只负责上色
        const grid = 60;
        const span = 3.2;
        const heatmap: number[] = [];
        for (let row = 0; row < grid; row++) {
            for (let col = 0; col < grid; col++) {
                const x = -span / 2 + (col + 0.5) * (span / grid);
                const y = span / 2 - (row + 0.5) * (span / grid);
                graph.forwardPropagation([x, y]);
                heatmap.push(graph.outputNodes[0].active_cache!);
            }
        }

        post({ type: 'done', acc, heatmap, trainMs });
    } catch (err) {
        post({ type: 'error', message: String(err) });
    }
});