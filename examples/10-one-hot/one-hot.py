import readline  # 导入此模块会自动接管 input() 并启用历史记录

from typing import Sequence, cast

import time

import torch
from torch import Tensor
from torch.nn import functional as F
import matplotlib.pyplot as plt

torch.manual_seed(42)  # 固定随机种子以便复现

"""
将26个字母和空格使用one-hot编码，输入小说种的文本训练，然后测试模型是否能输出单词或者句子
"""
encoding_space = 27


class RNN:
    def __init__(self, depth=2, width=2):

        # 使用cpu训练，因为这份代码还没有针对批量训练做调整，用GPU会很慢

        self.device = "cpu"

        self.depth = depth

        self.width = width

        def init_layer(shape: Sequence[int]):
            # 参数直接建在目标设备上，避免后面反复 .to(device)
            return (torch.randn(shape, device=self.device) * 0.1).requires_grad_()

        # +1是让下一层的偏置bias生效
        self.out_layer = init_layer(
            (encoding_space, width + 1)
        )  # 输出层是一个encoding_space行向量

        # 每个矩阵的行分别是神经元的权重+偏置
        # 第一行个隐藏层因为只有一个输入向量，所以是width+8（encoding_space个权重(one-hot编码)+一个偏置+自己的反馈）
        # 后面隐藏层每行需要前一层的所有输入权重+一个偏置+自己反馈
        # +1是让下一层的偏置bias生效
        self.hidden_layers: list[Tensor] = [
            init_layer((width, encoding_space + 1 + width)),
            *[init_layer((width, width + 1 + width)) for _ in range(depth - 1)],
        ]

    def encoding(self, s: str) -> Tensor:
        """
        生成一个二阶one-hot编码张量
        """
        idx = encoding_space - 1 if s == " " else ord(s) - ord("a")
        # 输入张量必须和参数在同一设备上，否则矩阵乘会报 device mismatch
        return F.one_hot(
            torch.tensor([idx], dtype=torch.long, device=self.device),
            num_classes=encoding_space,
        ).t()

    def decoding(self, t: Tensor) -> list[tuple[str, float]]:
        """
        将概率张量（27x1）解码为 (字符, 概率) 列表，按概率从高到低排序
        """
        probs = t.detach().flatten().tolist()  # (encoding_space,) 的一维概率列表

        result: list[tuple[str, float]] = []
        for idx, p in enumerate(probs):
            # 0~25 对应 a~z，26 对应空格
            ch = " " if idx == encoding_space - 1 else chr(ord("a") + idx)
            result.append((ch, float(p)))

        # 按概率从高到低排序后返回
        return sorted(result, key=lambda kv: kv[1], reverse=True)

    def forward(self, xs: str, ys: str | None = None) -> Tensor:
        with torch.set_grad_enabled(ys is not None):
            # 初始化各隐藏层上一次的输出
            last_output: list[Tensor] = [
                torch.zeros(self.width, 1, device=self.device)
                for _ in range(self.depth)
            ]

            # 整个窗口的梯度
            total_loss: Tensor | None = None

            # 训练时保存每一个时刻的推导结果，推理时表示最终输出
            out: Tensor | None = None

            if ys is not None:
                total_loss = torch.tensor(0.0, device=self.device)

            for data_idx in range(len(xs)):
                # 补1可以用于偏置，得到这样一个列向量 [[x_n][1]]，当前层的输入
                encoding_vec = self.encoding(xs[data_idx])
                layer_input = torch.cat(
                    [encoding_vec, torch.ones((1, 1), device=self.device)], dim=0
                )
                for l_idx in range(self.depth):

                    # 组装前一层输出和当前层上一次输出
                    lastOut = last_output[l_idx]
                    cat = torch.cat([lastOut, layer_input], dim=0)

                    layer = self.hidden_layers[l_idx]

                    inner_output = layer @ cat  # 矩阵计算

                    layer_output = F.tanh(inner_output)  # 激活函数

                    last_output[l_idx] = layer_output  # 缓存本次输出

                    # 在结果后面补一个标量1作为下一层输入，1用于偏置计算
                    layer_input = torch.cat(
                        [layer_output, torch.ones((1, 1), device=self.device)], dim=0
                    )

                if ys is not None:
                    # 输出层使用线性激活函数（现在系数暂时为1）
                    inner_linear = self.out_layer @ layer_input  # 线性部分
                    out = F.softmax(inner_linear, dim=0)  # 激活函数
                    target = self.encoding(ys[data_idx])  # 对输出编码
                    selected = out * target  # 逐元素相乘，保留正确字符对应的概率
                    prob = selected.sum(dim=0)  # 累加得到正确字符的概率
                    loss = -torch.log(prob)  #  负对数
                    assert total_loss is not None
                    total_loss = total_loss + loss
                elif data_idx == len(xs) - 1:
                    # 推理的时候，暂时不需要输出中间结果
                    # 输出层使用线性激活函数（现在系数暂时为1）
                    inner_linear = self.out_layer @ layer_input  # 线性部分
                    out = F.softmax(inner_linear, dim=0)  # 激活函数

            if ys is not None:
                assert total_loss is not None
                return total_loss
            else:
                assert out is not None
                return out

    def train(self, xs: str, ys: str, learning_rate=0.05):
        total_loss = cast(Tensor, self.forward(xs, ys))

        # 平均梯度，不计算平均梯度也没事，这里相当于是影响学习率而已，可以通过学习率调整
        avg_loss = total_loss / len(xs)

        avg_loss.backward()  # 反向传播

        with torch.no_grad():  # 关闭求导系统（别让计算图继续记录梯度的推导了）
            # 开始刷新参数
            assert self.out_layer.grad is not None
            self.out_layer -= learning_rate * self.out_layer.grad
            self.out_layer.grad = None  # 清空梯度，避免下轮训练被累加
            for layer in self.hidden_layers:
                assert layer.grad is not None
                layer -= learning_rate * layer.grad
                layer.grad = None  # 清空梯度，避免下轮训练被累加

        return avg_loss


def main():
    rnn = RNN(depth=2, width=128)

    print(f"训练设备: {rnn.device}")

    with open("data.txt") as file:
        lines = [line.strip() for line in file]

    loss_list = []
    for epoch in range(3):
        start = time.perf_counter()
        epoch_loss: list[float] = []
        for idx, line in enumerate(lines):
            loss = rnn.train(
                line[:-1],
                line[1:],
                learning_rate=0.01,
            )
            epoch_loss.append(loss.item())
            if (idx + 1) % 100 == 0:
                print(f"训练了{(idx+1)}行文本")
        loss_list.append(epoch_loss)
        print(f"训练完成{epoch+1}轮 耗时 {time.perf_counter() - start:.2f}s")

    plt.ion()
    plt.figure()
    for idx, loss in enumerate(loss_list):
        plt.plot([(i + 1) for i in range(len(loss))], loss, label=f"第{idx}轮Loss")
        plt.title("Loss示例")  # 标题
        plt.xlabel("step")  # X轴标签
        plt.ylabel("平均交叉熵损失")  # Y轴标签
        plt.legend()  # 显示图例
        plt.grid(True)  # 显示网格线

    plt.show(block=False)

    while True:
        text = input()
        for _ in range(20):  # 往后输出20个字符
            ret = rnn.forward(text)
            ch, p = rnn.decoding(ret)[0]
            print(f'"{text}"的下一个字符是:{ch} (概率 {p:.4f})')
            text += ch
        plt.pause(0.1)


if __name__ == "__main__":
    main()
