import test from 'node:test';
import Assert from 'node:assert';
import { Tensor } from './torch.ts';
const shape = [1, 2, 3, 4];
test('stride 测试', () => {
    Assert.deepStrictEqual(Tensor.stride(shape), [24, 12, 4, 1]);
});
test('size 测试', () => {
    Assert.equal(Tensor.size(shape), 24);
});
test('index 测试', () => {
    const stride = Tensor.stride(shape);
    Assert.equal(Tensor.index([0, 0, 0, 0], stride), 0);
    Assert.equal(Tensor.index([0, 0, 0, 1], stride), 1);
    Assert.equal(Tensor.index([0, 0, 1, 0], stride), 4);
    Assert.equal(Tensor.index([0, 0, 1, 1], stride), 5);
    Assert.equal(Tensor.index([0, 1, 1, 1], stride), 17);
    Assert.equal(Tensor.index([1, 1, 1, 1], stride), 41);
});
test('index2idx 测试', () => {
    const stride = Tensor.stride(shape);
    Assert.deepStrictEqual(Tensor.index2idx(0, stride), [0, 0, 0, 0]);
    Assert.deepStrictEqual(Tensor.index2idx(1, stride), [0, 0, 0, 1]);
    Assert.deepStrictEqual(Tensor.index2idx(4, stride), [0, 0, 1, 0]);
    Assert.deepStrictEqual(Tensor.index2idx(5, stride), [0, 0, 1, 1]);
    Assert.deepStrictEqual(Tensor.index2idx(17, stride), [0, 1, 1, 1]);
    Assert.deepStrictEqual(Tensor.index2idx(41, stride), [1, 1, 1, 1]);
});

test('broadCastToShape: 同形直接返回自身', () => {
    const a = new Tensor(Float64Array.from([1, 2, 3, 4, 5, 6]), [2, 3]);
    Assert.strictEqual(Tensor.broadCastToShape(a, [2, 3]), a);
});
test('broadCastToShape: 行复制 [3]->[2,3]', () => {
    const a = new Tensor(Float64Array.from([1, 2, 3]), [3]);
    const r = Tensor.broadCastToShape(a, [2, 3]);
    Assert.deepStrictEqual(Array.from(r.data), [1, 2, 3, 1, 2, 3]);
    Assert.strictEqual(r.sourceTensor, a);
});
test('broadCastToShape: 列复制 [2,1]->[2,3]', () => {
    const a = new Tensor(Float64Array.from([1, 2]), [2, 1]);
    const r = Tensor.broadCastToShape(a, [2, 3]);
    Assert.deepStrictEqual(Array.from(r.data), [1, 1, 1, 2, 2, 2]);
});
test('broadCastToShape: 补维 [2,3]->[4,2,3]', () => {
    const a = new Tensor(Float64Array.from([1, 2, 3, 4, 5, 6]), [2, 3]);
    const r = Tensor.broadCastToShape(a, [4, 2, 3]);
    Assert.deepStrictEqual(Array.from(r.data), [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]);
});
test('broadCastToShape: 标量->[2,3]', () => {
    const a = new Tensor(Float64Array.from([7]), []);
    const r = Tensor.broadCastToShape(a, [2, 3]);
    Assert.deepStrictEqual(Array.from(r.data), [7, 7, 7, 7, 7, 7]);
});
test('broadCastToShape: 中层重复 [2,1,3]->[2,4,3]', () => {
    const a = new Tensor(Float64Array.from([1, 2, 3, 4, 5, 6]), [2, 1, 3]);
    const r = Tensor.broadCastToShape(a, [2, 4, 3]);
    Assert.deepStrictEqual(Array.from(r.data), [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6]);
});

test('broadCast: 矩阵与标量', () => {
    const a = new Tensor(Float64Array.from([1, 2, 3, 4]), [2, 2]);
    const b = new Tensor(Float64Array.from([10]), []);
    const [a2, b2] = Tensor.broadCast(a, b);
    Assert.deepStrictEqual(a2.shape, [2, 2]);
    Assert.strictEqual(a2, a); // 同形的那一方复用自身
    Assert.deepStrictEqual(Array.from(b2.data), [10, 10, 10, 10]);
    Assert.strictEqual(b2.sourceTensor, b);
});
test('broadCast: 向量与矩阵 [3] 与 [2,3]', () => {
    const a = new Tensor(Float64Array.from([1, 2, 3]), [3]);
    const b = new Tensor(Float64Array.from([1, 2, 3, 4, 5, 6]), [2, 3]);
    const [a2, b2] = Tensor.broadCast(a, b);
    Assert.deepStrictEqual(a2.shape, [2, 3]);
    Assert.deepStrictEqual(Array.from(a2.data), [1, 2, 3, 1, 2, 3]);
    Assert.strictEqual(b2, b); // 同形复用
});
test('broadCast: 列向量与行向量 [2,1] 与 [1,3]', () => {
    const a = new Tensor(Float64Array.from([1, 2]), [2, 1]);
    const b = new Tensor(Float64Array.from([10, 20, 30]), [1, 3]);
    const [a2, b2] = Tensor.broadCast(a, b);
    Assert.deepStrictEqual(a2.shape, [2, 3]);
    Assert.deepStrictEqual(Array.from(a2.data), [1, 1, 1, 2, 2, 2]);
    Assert.deepStrictEqual(Array.from(b2.data), [10, 20, 30, 10, 20, 30]);
});
test('broadCast: 不可广播应抛错', () => {
    const a = new Tensor(Float64Array.from([1, 2]), [2]);
    const b = new Tensor(Float64Array.from([1, 2, 3]), [3]);
    Assert.throws(() => Tensor.broadCast(a, b));
});

test('foldBack: 标量折回为总和', () => {
    const a = new Tensor(Float64Array.from([7]), []);
    const b = Tensor.broadCastToShape(a, [2, 3]);
    Assert.deepStrictEqual([...Tensor.foldBack([1, 2, 3, 4, 5, 6], [2, 3], [])], [21]);
});
test('foldBack: 队列归约 [2,1] 与行归约', () => {
    const a = new Tensor(Float64Array.from([1, 2]), [2, 1]);
    const b = Tensor.broadCastToShape(a, [2, 3]);
    Assert.deepStrictEqual([...Tensor.foldBack([1, 2, 3, 4, 5, 6], [2, 3], [2, 1])], [6, 15]);
});
test('addGrad: 标量+矩阵 反向自动折回', () => {
    const s = new Tensor(Float64Array.from([10]), []);
    const m = new Tensor(Float64Array.from([1, 2, 3, 4]), [2, 2]);
    const out = Tensor.add(s, m);
    out.grad = Float64Array.from([1, 2, 3, 4]);
    out.gradFn!();
    // m 同形未广播，梯度原样累加
    Assert.deepStrictEqual([...m.grad!], [1, 2, 3, 4]);
    // s 被广播成 [2,2]，梯度求和折回：1+2+3+4=10
    Assert.deepStrictEqual([...s.grad!], [10]);
});
test('mul 广播反向: 向量×矩阵 双方都折对', () => {
    const v = new Tensor(Float64Array.from([2, 3]), [2]);
    const m = new Tensor(Float64Array.from([1, 2, 3, 4]), [2, 2]);
    const out = Tensor.mul(v, m); // v 行复制成 [[2,3],[2,3]]
    out.grad = Float64Array.from([1, 1, 1, 1]);
    out.gradFn!();
    // ∂out/∂m = v'(广播前向) * g = [2,3,2,3]
    Assert.deepStrictEqual([...m.grad!], [2, 3, 2, 3]);
    // ∂out/∂v：广播形状梯度 = m*g = [1,2,3,4]，行归约折回 [2] → [1+3, 2+4]
    Assert.deepStrictEqual([...v.grad!], [4, 6]);
});
test('matmul: 前向与反向', () => {
    const a = new Tensor(Float64Array.from([1, 2, 3, 4, 5, 6]), [2, 3]);
    const b = new Tensor(Float64Array.from([7, 8, 9, 10, 11, 12]), [3, 2]);
    const c = Tensor.matmul(a, b);
    // C = A@B 手算：[58,64;139,154]
    Assert.deepStrictEqual([...c.data], [58, 64, 139, 154]);
    Assert.deepStrictEqual(c.shape, [2, 2]);
    // g 选 C[0][0] 与 C[1][1] 为 1，其余 0，便于手算
    c.grad = Float64Array.from([1, 0, 0, 1]);
    c.gradFn!();
    // ∂C/∂A = G @ Bᵀ：ga = [g·rowB0? 手算 [7,9,11;8,10,12]]
    Assert.deepStrictEqual([...a.grad!], [7, 9, 11, 8, 10, 12]);
    // ∂C/∂B = Aᵀ @ G：gb = [1,4,2,5,3,6]
    Assert.deepStrictEqual([...b.grad!], [1, 4, 2, 5, 3, 6]);
});
test('matmul: 形状不匹配抛错', () => {
    // a 列数 3 ≠ b 行数 2，内维不符应抛错
    const a = new Tensor(Float64Array.from([1, 2, 3, 4, 5, 6]), [2, 3]);
    const b = new Tensor(Float64Array.from([1, 2, 3, 4]), [2, 2]);
    Assert.throws(() => Tensor.matmul(a, b));
});
test('mean: 归约与梯度扩散', () => {
    const t = new Tensor(Float64Array.from([1, 2, 3, 4]), [2, 2]);
    const m = Tensor.mean(t);
    Assert.equal(m.data[0], 2.5);
    Assert.deepStrictEqual(m.shape, []);
    m.grad = Float64Array.from([1]);
    m.gradFn!();
    Assert.deepStrictEqual([...t.grad!], [0.25, 0.25, 0.25, 0.25]);
});
test('sigmoid: 前向与梯度', () => {
    const t = new Tensor(Float64Array.from([0, 1]), [2]);
    const s = Tensor.sigmoid(t);
    const s0 = 1 / (1 + Math.exp(0));
    const s1 = 1 / (1 + Math.exp(-1));
    Assert.ok(Math.abs(s.data[0] - s0) < 1e-12);
    Assert.ok(Math.abs(s.data[1] - s1) < 1e-12);
    s.grad = Float64Array.from([1, 1]);
    s.gradFn!();
    Assert.ok(Math.abs((t.grad?.[0] ?? 0) - s0 * (1 - s0)) < 1e-12);
    Assert.ok(Math.abs((t.grad?.[1] ?? 0) - s1 * (1 - s1)) < 1e-12);
});
test('backward: mean(mul(A,B)) 端到端', () => {
    const a = new Tensor(Float64Array.from([1, 2, 3, 4]), [2, 2]);
    const b = new Tensor(Float64Array.from([5, 6, 7, 8]), [2, 2]);
    const out = Tensor.mean(Tensor.mul(a, b)); // (5+12+21+32)/4 = 17.5
    out.backward();
    Assert.equal(out.data[0], 17.5);
    Assert.deepStrictEqual([...a.grad!], [1.25, 1.5, 1.75, 2]); // ∂out/∂a_i = b_i/4
    Assert.deepStrictEqual([...b.grad!], [0.25, 0.5, 0.75, 1]); // ∂out/∂b_i = a_i/4
});
test('backward: 从任意中间节点反向（seedGrad）', () => {
    const a = new Tensor(Float64Array.from([1, 2, 3, 4]), [2, 2]);
    const b = new Tensor(Float64Array.from([5, 6, 7, 8]), [2, 2]);
    const c = Tensor.matmul(a, b);
    const _ = Tensor.mean(c); // 图里还有更下游的节点，但我们只从 c 反向
    c.backward(Float64Array.from([1, 0, 0, 1]));
    // ∂c/∂a = G@Bᵀ = [5,7,6,8]；∂c/∂b = Aᵀ@G = [1,3,2,4]
    Assert.deepStrictEqual([...a.grad!], [5, 7, 6, 8]);
    Assert.deepStrictEqual([...b.grad!], [1, 3, 2, 4]);
});
