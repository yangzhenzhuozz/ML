"""
one-hot 编码 + **批量训练** 的 RNN 版本。

和 one-hot.py（逐样本版）的区别：

1. 输入从「一个字符」变成 (T, B) 的索引张量：
   T = 每个窗口的时间步数，B = 一次并行处理的序列条数
2. 隐藏层状态从 (width, 1) 变成 (width, B)，一次矩阵乘就算完 B 条序列，
   矩阵形状变大之后 GPU 才有发挥空间
3. 损失用 F.cross_entropy（内部 log_softmax，比手写 one-hot 相乘更数值稳定），
   并且按字符数取平均，学习率含义和原版一致
4. 采样用随机窗口起点，不再依赖行切分，避免短句/长句长度不齐
"""

from typing import Sequence, cast

import time

import torch
from torch import Tensor
from torch.nn import functional as F

import matplotlib.pyplot as plt

torch.manual_seed(42)  # 固定随机种子以便复现

encoding_space = 27


def char_to_idx(ch: str) -> int:
    return encoding_space - 1 if ch == " " else ord(ch) - ord("a")


def idx_to_char(idx: int) -> str:
    return " " if idx == encoding_space - 1 else chr(ord("a") + idx)


class RNN:
    def __init__(self, depth=2, width=2, device: torch.device | str = "cpu"):
        # 这张卡上逐样本训练 CPU 比 GPU 快 9 倍，所以默认仍用 CPU；批量版可以传 "cuda" 对比
        self.device = torch.device(device)

        self.depth = depth

        self.width = width

        def init_layer(shape: Sequence[int]):
            return torch.randn(shape, device=self.device).requires_grad_()

        self.out_layer = init_layer((encoding_space, width + 1))

        # 每个矩阵的行 = 一个神经元的权重 + 偏置 + 自己的反馈
        # 形状和批量大小无关，权重是按神经元复用的，只是输入从列向量变成 B 列的矩阵
        self.hidden_layers: list[Tensor] = [
            init_layer((width, encoding_space + 1 + width)),
            *[init_layer((width, width + 1 + width)) for _ in range(depth - 1)],
        ]

    # ---------- 编解码 ----------

    def encode(self, text: str) -> Tensor:
        """字符串 -> (T, 1) 索引张量"""
        ids = [char_to_idx(c) for c in text]
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(1)

    def decoding(self, t: Tensor) -> list[tuple[str, float]]:
        """
        将单个样本的概率张量（27x1 或 27）解码为 (字符, 概率) 列表，按概率从高到低排序
        """
        probs = t.detach().flatten().tolist()

        result: list[tuple[str, float]] = []
        for idx, p in enumerate(probs):
            result.append((idx_to_char(idx), float(p)))

        return sorted(result, key=lambda kv: kv[1], reverse=True)

    # ---------- 前向 / 训练 ----------

    def forward(self, xs: Tensor, ys: Tensor | None = None) -> Tensor:
        """
        xs: (T, B) 字符索引，ys: (T, B) 目标字符索引
        训练（ys 不为 None）返回标量损失（每个字符平均）；推理返回 (27, B) 的概率
        """
        if xs.dim() != 2:
            raise ValueError(f"xs 需要是 (T, B) 的二维张量，收到 {tuple(xs.shape)}")

        T, B = xs.shape

        with torch.set_grad_enabled(ys is not None):
            # (width, B)：每一列是一条独立序列的隐藏状态
            last_output: list[Tensor] = [
                torch.zeros(self.width, B, device=self.device)
                for _ in range(self.depth)
            ]

            # 偏置项：只需要造一次，(1, B) 的整行 1
            ones = torch.ones(1, B, device=self.device)

            total_loss: Tensor | None = None
            if ys is not None:
                total_loss = torch.zeros((), device=self.device)

            out: Tensor | None = None

            for t in range(T):
                # F.one_hot: (B,) -> (B, 27)，转置成 (27, B) 后每列就是一个字符的 one-hot
                layer_input = torch.cat(
                    [F.one_hot(xs[t], encoding_space).t().float(), ones], dim=0
                )

                for l_idx in range(self.depth):
                    last_out = last_output[l_idx]
                    # 前一层输出 + 自己上一时刻的输出，纵向拼起来
                    cat = torch.cat([last_out, layer_input], dim=0)
                    layer_output = F.tanh(self.hidden_layers[l_idx] @ cat)
                    last_output[l_idx] = layer_output
                    layer_input = torch.cat([layer_output, ones], dim=0)

                inner_linear = self.out_layer @ layer_input  # (27, B) 线性输出

                if ys is not None:
                    # 每列一个样本：转置成 (B, 27) 做交叉熵，先按样本求和，最后整体除以字符数
                    assert total_loss is not None
                    total_loss = total_loss + F.cross_entropy(
                        inner_linear.t(), ys[t], reduction="sum"
                    )
                else:
                    out = F.softmax(inner_linear, dim=0)

            if ys is not None:
                assert total_loss is not None
                return total_loss / (T * B)

            assert out is not None
            return out

    def train_batch(self, xs: Tensor, ys: Tensor, learning_rate=0.05):
        loss = self.forward(xs, ys)  # 已按字符数平均

        loss.backward()  # 反向传播

        with torch.no_grad():  # 关闭求导系统（别让计算图继续记录梯度的推导了）
            # 开始刷新参数
            for param in [self.out_layer, *self.hidden_layers]:
                assert param.grad is not None
                param -= learning_rate * param.grad
                param.grad = None  # 清空梯度，避免下轮训练被累加

        return loss

    # ---------- 推理 ----------

    @torch.no_grad()
    def generate(self, seed: str, steps: int = 20) -> str:
        """给一段种子文本，续写 steps 个字符"""
        text = seed
        for _ in range(steps):
            # 仍然一次只有一个序列，所以 B = 1
            out = self.forward(self.encode(text))
            ch = self.decoding(out)[0][0]
            text += ch
        return text


# ---------- 数据 ----------

DATA_PATH = "examples/10-one-hot/data.txt"


def load_ids(path: str = DATA_PATH) -> Tensor:
    """把整篇语料摊平成一条 (N,) 的索引序列；换行统一换成空格（空格本身也是 27 类之一）"""
    with open(path, encoding="utf-8") as file:
        lines = [line.strip() for line in file]
    text = " ".join(lines)
    return torch.tensor([char_to_idx(c) for c in text], dtype=torch.long)


# ---------- 训练入口 ----------


def run(
    device: str,
    steps: int = 3000,
    window: int = 32,
    batch_size: int = 64,
    learning_rate: float = 0.05,
    log_every: int = 250,
    width: int = 10,
    depth: int = 5,
) -> tuple[float, list[float]]:
    """
    训练 steps 个 batch，返回（每步平均耗时 ms, 损失曲线）

    注意：批量训练每次更新的梯度是 batch*window 个字符的平均值，
    所以「每轮的更新次数」比逐样本版少几十倍，学习率要相应放大（线性缩放规则）。
    """
    torch.manual_seed(42)
    rnn = RNN(depth=depth, width=width, device=device)

    ids = load_ids().to(rnn.device)

    chars_per_step = window * batch_size
    print(
        f"[{device}] 语料 {ids.numel()} 字符 · {steps} 步 × "
        f"batch {batch_size} × 窗口 {window} ≈ {steps * chars_per_step} 字符"
    )

    curve: list[float] = []
    loss_sum = torch.zeros((), device=rnn.device)
    start = time.perf_counter()

    for step in range(1, steps + 1):
        # 随机挑 batch_size 个窗口起点，一次取出 (B, T+1) 的连续片段
        starts = torch.randint(
            0, ids.numel() - window - 1, (batch_size,), device=rnn.device
        )
        seq = ids[starts[:, None] + torch.arange(window + 1, device=rnn.device)]

        xs = seq[:, :window].t().contiguous()  # (T, B)
        ys = seq[:, 1:].t().contiguous()  # (T, B)，整体右移一位

        loss = rnn.train_batch(xs, ys, learning_rate)
        # 在设备上累加，不要每步 .item()，否则每步都同步、打断 GPU 流水线
        loss_sum = loss_sum + loss.detach()

        if step % log_every == 0:
            curve.append((loss_sum / log_every).item())
            loss_sum = torch.zeros((), device=rnn.device)
            elapsed = time.perf_counter() - start
            print(
                f"[{device}] step {step:5d}/{steps} loss={curve[-1]:.4f} "
                f"耗时 {elapsed:5.1f}s ({step * chars_per_step / elapsed / 1e3:.1f}k 字符/s)"
            )

    elapsed = time.perf_counter() - start
    print(
        f"[{device}] 总耗时 {elapsed:.2f}s · 每步 {elapsed / steps * 1000:.2f}ms · "
        f"{steps * chars_per_step / elapsed / 1e3:.1f}k 字符/s"
    )
    print(f"[{device}] 续写: {rnn.generate('alice', 20)!r}")

    return elapsed / steps * 1000, curve


def main():
    curves: dict[str, list[float]] = {}

    for name in ("cpu", "cuda"):
        if name == "cuda" and not torch.cuda.is_available():
            continue
        _, curves[name] = run(name)

    plt.figure()
    for name, losses in curves.items():
        plt.plot([(i + 1) * 250 for i in range(len(losses))], losses, label=name)
    plt.title("Loss示例")  # 标题
    plt.xlabel("step")  # X轴标签
    plt.ylabel("平均交叉熵损失")  # Y轴标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格线

    plt.show()


if __name__ == "__main__":
    main()
