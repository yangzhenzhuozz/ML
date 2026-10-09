function asserts(condition: boolean, msg?: string): asserts condition {
    if (!condition) {
        throw new Error(msg ?? 'error');
    }
}

class Tensor {
    data: Float64Array;
    grad: Float64Array | undefined;
    shape: number[];
    sourceTensor: Tensor | undefined;
    gradFn: (() => any) | undefined; //梯度分发函数
    inputs: Tensor[] | undefined;
    constructor(data: Float64Array, shape: number[]) {
        this.data = data;
        this.shape = shape;
    }

    /**
     * 统一给本张量累积一段梯度：
     *  - 若我是广播张量（有 sourceTensor），先把梯度按形状差异归约，折回给源张量；
     *  - 否则累加到自己的 grad（节点被复用几次 → 梯度累加，第 8 章规则）
     */
    public addGrad(g: Float64Array | number[]): void {
        if (this.sourceTensor) {
            this.sourceTensor.addGrad(Tensor.foldBack(g, this.shape, this.sourceTensor.shape));
            return;
        }
        if (!this.grad) this.grad = new Float64Array(g.length);
        for (let i = 0; i < g.length; i++) this.grad[i] += g[i];
    }

    //把两个Tensor调整为同形（广播对齐），返回 [A', B']；无法对齐则抛错
    public static broadCast(A: Tensor, B: Tensor): [Tensor, Tensor] {
        const maxDimension = Math.max(A.shape.length, B.shape.length);
        const targetShape: number[] = new Array<number>(maxDimension).fill(0);
        // 从最右维向左对齐：相等或一方为 1 才可广播（第 8 章广播规则）
        for (let i = 0; i < maxDimension; i++) {
            const dA = i < maxDimension - A.shape.length ? 1 : A.shape[i - (maxDimension - A.shape.length)];
            const dB = i < maxDimension - B.shape.length ? 1 : B.shape[i - (maxDimension - B.shape.length)];
            if (dA === dB) {
                targetShape[i] = dA;
            } else if (dA === 1) {
                targetShape[i] = dB;
            } else if (dB === 1) {
                targetShape[i] = dA;
            } else {
                asserts(false, `广播失败：[${A.shape}] 与 [${B.shape}] 无法对齐`);
            }
        }
        return [Tensor.broadCastToShape(A, targetShape), Tensor.broadCastToShape(B, targetShape)];
    }

    public static size(shape: number[]): number {
        let ret = 1;
        for (let i = 0; i < shape.length; i++) {
            ret = ret * shape[i];
        }
        return ret;
    }

    public static stride(shape: number[]): number[] {
        const ret = new Array<number>(shape.length);
        ret[ret.length - 1] = 1;
        let subStride = 1;
        for (let i = ret.length - 2; i >= 0; i--) {
            subStride = subStride * shape[i + 1];
            ret[i] = subStride;
        }
        return ret;
    }

    public static index(idx: number[], stride: number[]): number {
        asserts(idx.length == stride.length, `idx长度:${idx.length}和stride长度${stride.length}不一致`);
        let ret = 0;
        for (let i = stride.length - 1; i >= 0; i--) {
            ret += idx[i] * stride[i];
        }
        return ret;
    }

    public static index2idx(idx: number, stride: number[]): number[] {
        const ret = new Array<number>(stride.length).fill(0);
        for (let i = 0; i < stride.length; i++) {
            ret[i] = Math.floor(idx / stride[i]);
            idx = idx - ret[i] * stride[i];
        }
        return ret;
    }

    /**
     * 把一个Tensor调整到目标shape，这里假设前置检查都已经通过
     * targetShape必定比t的shape阶数大，且所有需要扩展的维度宽度都是1
     * @param t
     * @param targetShape
     * @returns
     */
    public static broadCastToShape(t: Tensor, shape: number[]): Tensor {
        const srcShape = Array<number>(shape.length).fill(1);
        const targetShape = Array.from(shape);
        for (let i = 0; i < srcShape.length; i++) {
            const ti = t.shape.length - 1 - i;
            srcShape[srcShape.length - 1 - i] = ti >= 0 ? t.shape[ti] : 1; // 阶数不足的高位补 1
        }
        let hasNewTensor = false;
        for (let i = 0; i < targetShape.length; i++) {
            if (srcShape[i] != targetShape[i]) {
                hasNewTensor = true;
                break;
            }
        }
        if (!hasNewTensor) {
            return t;
        } else {
            const data = new Float64Array(Tensor.size(targetShape));
            // 逐元素复制：目标下标 → 源下标；源宽度为 1 的维 → 源下标固定为 0
            const srcStride = Tensor.stride(srcShape); // 用补 1 后的 srcShape，长度与 shapeIdx 匹配
            const destStride = Tensor.stride(targetShape);
            const size = Tensor.size(targetShape);
            for (let idx = 0; idx < size; idx++) {
                const shapeIdx = Tensor.index2idx(idx, destStride);
                for (let i = 0; i < srcShape.length; i++) {
                    shapeIdx[i] = srcShape[i] === 1 ? 0 : shapeIdx[i];
                }
                const srcIndex = Tensor.index(shapeIdx, srcStride);
                data[idx] = t.data[srcIndex];
            }
            const ret = new Tensor(data, targetShape);
            ret.sourceTensor = t; // 记录广播来源，反向传播时据此把梯度折回
            return ret;
        }
    }

    /**
     * broadCastToShape 的逆：把“广播形状(toShape)”的梯度归约回“源形状(fromShape)”
     * 元素被复制了几份，梯度就累加几份（第 8 章广播反向）
     */
    public static foldBack(g: Float64Array | number[], toShape: number[], fromShape: number[]): Float64Array {
        const srcShape = Array<number>(toShape.length).fill(1);
        for (let i = 0; i < srcShape.length; i++) {
            const fi = fromShape.length - 1 - i;
            srcShape[srcShape.length - 1 - i] = fi >= 0 ? fromShape[fi] : 1;
        }
        const out = new Float64Array(Tensor.size(fromShape));
        const srcStride = Tensor.stride(srcShape); // 长度与 toShape 对齐
        const destStride = Tensor.stride(toShape);
        const len = Tensor.size(toShape);
        for (let idx = 0; idx < len; idx++) {
            const shapeIdx = Tensor.index2idx(idx, destStride);
            for (let i = 0; i < srcShape.length; i++) {
                shapeIdx[i] = srcShape[i] === 1 ? 0 : shapeIdx[i];
            }
            out[Tensor.index(shapeIdx, srcStride)] += g[idx];
        }
        return out;
    }

    public static add(a: Tensor, b: Tensor): Tensor {
        const ts = Tensor.broadCast(a, b);
        const data = new Float64Array(ts[0].data.length);
        const shape = ts[0].shape;
        for (let i = 0; i < data.length; i++) {
            data[i] = ts[0].data[i] + ts[1].data[i];
        }
        const ret = new Tensor(data, shape);
        ret.inputs = ts;
        ret.gradFn = () => {
            if (!ret.grad) return;
            ret.inputs![0].addGrad(ret.grad); // ∂(a+b)/∂a = 1
            ret.inputs![1].addGrad(ret.grad); // ∂(a+b)/∂b = 1
        };
        return ret;
    }
    public static sub(a: Tensor, b: Tensor): Tensor {
        const ts = Tensor.broadCast(a, b);
        const data = new Float64Array(ts[0].data.length);
        const shape = ts[0].shape;
        for (let i = 0; i < data.length; i++) {
            data[i] = ts[0].data[i] - ts[1].data[i];
        }
        const ret = new Tensor(data, shape);
        ret.inputs = ts;
        ret.gradFn = () => {
            if (!ret.grad) return;
            ret.inputs![0].addGrad(ret.grad); // ∂(a-b)/∂a = 1
            ret.inputs![1].addGrad(ret.grad.map((item) => -item)); // ∂(a-b)/∂b = -1
        };
        return ret;
    }
    public static mul(a: Tensor, b: Tensor): Tensor {
        const ts = Tensor.broadCast(a, b);
        const data = new Float64Array(ts[0].data.length);
        const shape = ts[0].shape;
        for (let i = 0; i < data.length; i++) {
            data[i] = ts[0].data[i] * ts[1].data[i];
        }
        const ret = new Tensor(data, shape);
        ret.inputs = ts;
        ret.gradFn = () => {
            if (!ret.grad) return;
            ret.inputs![0].addGrad(ret.grad.map((item, idx) => item * ret.inputs![1].data[idx])); // ∂(a·b)/∂a = b（前向值）
            ret.inputs![1].addGrad(ret.grad.map((item, idx) => item * ret.inputs![0].data[idx])); // ∂(a·b)/∂b = a（前向值）
        };
        return ret;
    }
    public static div(a: Tensor, b: Tensor): Tensor {
        const ts = Tensor.broadCast(a, b);
        const data = new Float64Array(ts[0].data.length);
        const shape = ts[0].shape;
        for (let i = 0; i < data.length; i++) {
            data[i] = ts[0].data[i] / ts[1].data[i];
        }
        const ret = new Tensor(data, shape);
        ret.inputs = ts;
        ret.gradFn = () => {
            if (!ret.grad) return;
            ret.inputs![0].addGrad(ret.grad.map((item, idx) => item / ret.inputs![1].data[idx])); // ∂(a/b)/∂a = 1/b
            ret.inputs![1].addGrad(ret.grad.map((item, idx) => item * (-ret.inputs![0].data[idx] / (ret.inputs![1].data[idx] * ret.inputs![1].data[idx])))); // ∂(a/b)/∂b = -a/b²
        };
        return ret;
    }
    /** 矩阵乘法：仅支持二维 [m,n] × [n,p] → [m,p]（向量先 reshape 成二维再用） */
    public static matmul(a: Tensor, b: Tensor): Tensor {
        asserts(a.shape.length === 2 && b.shape.length === 2, `matmul: 仅支持二维矩阵，得到 ${a.shape} 与 ${b.shape}`);
        const [m, n] = a.shape;
        const [n2, p] = b.shape;
        asserts(n === n2, `matmul: 内维不匹配 ${n} vs ${n2}`);

        // 前向：C[m,p] = A[m,n] @ B[n,p]
        const data = new Float64Array(m * p);
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < p; j++) {
                let s = 0;
                for (let k = 0; k < n; k++) s += a.data[i * n + k] * b.data[k * p + j];
                data[i * p + j] = s;
            }
        }
        const ret = new Tensor(data, [m, p]);
        ret.inputs = [a, b];
        ret.gradFn = () => {
            if (!ret.grad) return;
            const g = ret.grad;
            // ∂out/∂A = G @ Bᵀ：ga[i][k] = Σ_j g[i][j] * b[k][j]（第 8 章矩阵乘法推导）
            const ga = new Float64Array(m * n);
            for (let i = 0; i < m; i++) {
                for (let k = 0; k < n; k++) {
                    let s = 0;
                    for (let j = 0; j < p; j++) s += g[i * p + j] * b.data[k * p + j];
                    ga[i * n + k] = s;
                }
            }
            a.addGrad(ga);
            // ∂out/∂B = Aᵀ @ G：gb[k][j] = Σ_i a[i][k] * g[i][j]
            const gb = new Float64Array(n * p);
            for (let k = 0; k < n; k++) {
                for (let j = 0; j < p; j++) {
                    let s = 0;
                    for (let i = 0; i < m; i++) s += a.data[i * n + k] * g[i * p + j];
                    gb[k * p + j] = s;
                }
            }
            b.addGrad(gb);
        };
        return ret;
    }

    /** 逐元素 sigmoid 激活：σ' = σ(1-σ)，用前向缓存的 σ(z) 求导 */
    public static sigmoid(t: Tensor): Tensor {
        const data = new Float64Array(t.data.length);
        for (let i = 0; i < data.length; i++) data[i] = 1 / (1 + Math.exp(-t.data[i]));
        const ret = new Tensor(data, t.shape);
        ret.inputs = [t];
        ret.gradFn = () => {
            if (!ret.grad) return;
            const g = ret.grad;
            const grad = new Float64Array(data.length);
            for (let i = 0; i < data.length; i++) grad[i] = g[i] * data[i] * (1 - data[i]);
            t.addGrad(grad);
        };
        return ret;
    }

    /** 全归约均值：把整张量归约成标量（loss 用），梯度按 1/n 均匀扩散 */
    public static mean(t: Tensor): Tensor {
        let total = 0;
        for (let i = 0; i < t.data.length; i++) total += t.data[i];
        const ret = new Tensor(new Float64Array([total / t.data.length]), []);
        ret.inputs = [t];
        ret.gradFn = () => {
            if (!ret.grad) return;
            const g = ret.grad[0] / t.data.length; // ∂mean/∂x_i = 1/n
            t.addGrad(new Float64Array(t.data.length).fill(g));
        };
        return ret;
    }

    /**
     * 自动求导主管道：DFS 后序 + reverse 得到从本节点到叶子的拓扑序，
     * 置种子梯度后按序调用 gradFn 分发。输出必须是标量（或通过 seedGrad 指定初始梯度）。
     */
    public backward(seedGrad?: Float64Array | number[]): void {
        const topo: Tensor[] = [];
        const visited = new Set<Tensor>();
        const build = (n: Tensor): void => {
            if (visited.has(n)) return;
            visited.add(n);
            if (n.inputs) for (const inp of n.inputs) build(inp);
            topo.push(n);
        };
        build(this);
        topo.reverse();

        if (seedGrad) {
            this.grad = Float64Array.from(seedGrad as ArrayLike<number>);
        } else {
            asserts(this.data.length === 1, `backward: 输出不是标量（len=${this.data.length}），需通过 seedGrad 指定初始梯度`);
            this.grad = new Float64Array([1]); // ∂out/∂out = 1
        }
        for (const n of topo) n.gradFn?.();
    }
}
export { Tensor };
