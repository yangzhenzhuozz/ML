import { Tensor } from './torch.ts';

// 最朴素的 SGD：更新参数并清空梯度
function sgd(params: Tensor[], lr: number): void {
    for (const p of params) {
        if (!p.grad) continue;
        for (let i = 0; i < p.data.length; i++) p.data[i] -= lr * p.grad[i];
        p.grad = undefined;
    }
}
const rand = (n: number): Float64Array => {
    const a = new Float64Array(n);
    for (let i = 0; i < n; i++) a[i] = Math.random() * 2 - 1;
    return a;
};

// XOR 数据集
const X = [0, 0, 0, 1, 1, 0, 1, 1].flat(); // 4×2 行主序
const Y = Float64Array.from([0, 1, 1, 0]); // 4×1

// 参数（叶子张量）：2 → sigmoid(4) → sigmoid(1)
const w1 = new Tensor(rand(2 * 4), [2, 4]);
const b1 = new Tensor(rand(4), [4]);
const w2 = new Tensor(rand(4 * 1), [4, 1]);
const b2 = new Tensor(rand(1), [1]);
const params = [w1, b1, w2, b2];

const lr = 0.6;
const iters = 30000;

let lastLoss = 0;
for (let it = 0; it <= iters; it++) {
    // 动态图：每次按当前参数重新搭建前向
    const x = new Tensor(Float64Array.from(X), [4, 2]);
    const y = new Tensor(Float64Array.from(Y), [4, 1]);

    const z1 = Tensor.matmul(x, w1); // [4,4]
    const h = Tensor.sigmoid(Tensor.add(z1, b1)); // [4,4]（b1 行广播）
    const z2 = Tensor.matmul(h, w2); // [4,1]
    const pred = Tensor.sigmoid(Tensor.add(z2, b2)); // [4,1]（b2 列广播）
    const diff = Tensor.sub(pred, y);
    const loss = Tensor.mean(Tensor.mul(diff, diff)); // 标量 MSE

    loss.backward();
    sgd(params, lr);
    lastLoss = loss.data[0];

    if (it % 5000 === 0) console.log(`iter ${it}: loss = ${lastLoss.toFixed(6)}`);
}

// 用训练好的参数跑一遍前向，打印预测
const x = new Tensor(Float64Array.from(X), [4, 2]);
const h = Tensor.sigmoid(Tensor.add(Tensor.matmul(x, w1), b1));
const pred = Tensor.sigmoid(Tensor.add(Tensor.matmul(h, w2), b2));
console.log('\n训练完成，最终 loss =', lastLoss.toFixed(6));
console.log('预测:');
const rows = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
];
for (let i = 0; i < 4; i++) {
    console.log(`  X=${JSON.stringify(rows[i])} -> ${pred.data[i].toFixed(3)}（目标 ${Y[i]}）`);
}