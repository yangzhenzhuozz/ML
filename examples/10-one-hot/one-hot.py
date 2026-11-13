import readline  # 导入此模块会自动接管 input() 并启用历史记录

import re

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


class RNN:
    def __init__(self, depth=2, width=2):
        self.device = "cuda"

        self.depth = depth

        self.width = width

        self.encoding_space = 27

        def init_layer(shape: Sequence[int]):
            # 参数直接建在目标设备上，避免后面反复 .to(device)
            return (torch.randn(shape, device=self.device) * 0.1).requires_grad_()

        # +1是让下一层的偏置bias生效
        self.out_layer = init_layer(
            (self.encoding_space, width + 1)
        )  # 输出层是一个encoding_space行向量

        # 每个矩阵的行分别是神经元的权重+偏置
        # 第一行个隐藏层因为只有一个输入向量，所以是width+8（encoding_space个权重(one-hot编码)+一个偏置+自己的反馈）
        # 后面隐藏层每行需要前一层的所有输入权重+一个偏置+自己反馈
        # +1是让下一层的偏置bias生效
        self.hidden_layers: list[Tensor] = [
            init_layer((width, self.encoding_space + 1 + width)),
            *[init_layer((width, width + 1 + width)) for _ in range(depth - 1)],
        ]

    def encoding(self, str: list[str]) -> Tensor:
        """
        把字符串变成一个二阶one-hot编码张量（列向量）
        """

        return F.one_hot(
            torch.tensor(
                [
                    self.encoding_space - 1 if s[0] == " " else ord(s[0]) - ord("a")
                    for s in str
                ],
                dtype=torch.long,
                device=self.device,
            ),
            num_classes=self.encoding_space,
        ).t()

    def decoding(self, t: Tensor) -> list[tuple[str, float]]:
        """
        将概率张量（27x1）解码为 (字符, 概率) 列表，按概率从高到低排序
        """
        probs = t.detach().flatten().tolist()  # (encoding_space,) 的一维概率列表

        result: list[tuple[str, float]] = []
        for idx, p in enumerate(probs):
            # 0~25 对应 a~z，26 对应空格
            ch = " " if idx == self.encoding_space - 1 else chr(ord("a") + idx)
            result.append((ch, float(p)))

        # 按概率从高到低排序后返回
        return sorted(result, key=lambda kv: kv[1], reverse=True)

    def forward(self, x_list: list[str], y_list: list[str] | None = None) -> Tensor:
        """
        要求str_list中的每组数据都必须
        """
        with torch.set_grad_enabled(y_list is not None):
            # 初始化各隐藏层上一次的输出
            last_output: list[Tensor] = [
                torch.zeros(self.width, len(x_list), device=self.device)
                for _ in range(self.depth)
            ]

            # 整个窗口的梯度
            total_loss: Tensor | None = None

            # 训练时保存每一个时刻的推导结果，推理时表示最终输出
            out: Tensor | None = None

            if y_list is not None:
                total_loss = torch.tensor(0.0, device=self.device)

            for data_idx in range(len(x_list[0])):
                encoding_vec = self.encoding([str[data_idx] for str in x_list])

                # 补1可以用于偏置，得到这样一个列向量 [[x_n][1]]，当前层的输入

                layer_input = torch.cat(
                    [
                        encoding_vec,
                        torch.ones_like(encoding_vec[:1, :], device=self.device),
                    ],
                    dim=0,
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
                        [
                            layer_output,
                            torch.ones_like(layer_output[:1, :], device=self.device),
                        ],
                        dim=0,
                    )

                if y_list is not None:
                    # 输出层使用线性激活函数（现在系数暂时为1）
                    inner_linear = self.out_layer @ layer_input  # 线性部分
                    out = F.softmax(inner_linear, dim=0)  # 激活函数
                    target = self.encoding(
                        [str[data_idx] for str in x_list]
                    )  # 对输出编码
                    selected = out * target  # 逐元素相乘，保留正确字符对应的概率
                    prob = selected.sum(dim=0)  # 累加得到正确字符的概率
                    loss = -torch.log(prob)  #  负对数
                    assert total_loss is not None
                    total_loss = total_loss + loss
                elif data_idx == len(x_list[0]) - 1:
                    # 推理的时候，暂时不需要输出中间结果
                    # 输出层使用线性激活函数（现在系数暂时为1）
                    inner_linear = self.out_layer @ layer_input  # 线性部分
                    out = F.softmax(inner_linear, dim=0)  # 激活函数

            if y_list is not None:
                assert total_loss is not None
                return total_loss.sum()
            else:
                assert out is not None
                return out

    def train(self, xs: list[str], ys: list[str], learning_rate=0.05):
        total_loss = cast(Tensor, self.forward(xs, ys))

        # 平均loss
        avg_loss = total_loss / (len(xs) * len(xs[0]))

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


def probs_to_markdown(
    probs: list[float], as_percent: bool = True, sort_by_prob: bool = False
) -> str:
    """将长度为 27 的概率列表转换为竖向 Markdown 表格字符串"""
    if len(probs) != 27:
        raise ValueError(f"概率列表长度必须为 27，当前长度为 {len(probs)}")

    headers = [chr(i + ord("a")) if i != 26 else "space" for i in range(27)]
    pairs = list(zip(headers, probs))

    # 如果开启排序，则按概率值（元组的第 2 个元素 x[1]）降序排列
    if sort_by_prob:
        pairs.sort(key=lambda x: x[1], reverse=True)

    lines = ["| 字符 | 概率 |", "| :--- | ---: |"]
    for char, p in pairs:
        val = f"{p * 100:.2f}%" if as_percent else f"{p:.4f}"
        lines.append(f"| {char} | {val} |")

    return "\n".join(lines)


def main():
    rnn = RNN(depth=5, width=5)

    print(f"训练设备: {rnn.device}")

    def process_and_chunk(s: str, n: int) -> list[str]:
        # 1-3步：将连续非字母替换为单个空格，并全部转为小写
        cleaned = re.sub(r"[^a-zA-Z]+", " ", s).lower()

        # 4步：按长度 n 切分，步长为 n，上限设为 len(cleaned) - n + 1 以丢弃末尾不足 n 的部分
        return [cleaned[i : i + n] for i in range(0, len(cleaned) - n + 1, n)]

    x_str = ""
    y_str = ""
    with open("data.txt") as file:
        file_str = file.read()
        x_str = process_and_chunk(file_str, 20)
        y_str = process_and_chunk(file_str[1:], 20)

    loss_list = []
    for epoch in range(500):
        start = time.perf_counter()
        epoch_loss = rnn.train(x_str, y_str)
        loss_list.append(epoch_loss.item())
        print(f"训练完成{epoch+1}轮 耗时 {time.perf_counter() - start:.2f}s")

    plt.ion()
    plt.figure()
    plt.plot([(i + 1) for i in range(len(loss_list))], loss_list, label=f"Loss")
    plt.title("Loss示例")  # 标题
    plt.xlabel("step")  # X轴标签
    plt.ylabel("平均交叉熵损失")  # Y轴标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格线

    plt.show(block=False)

    while True:
        text = input()
        for _ in range(1):  # 往后输出20个字符
            ret = rnn.forward([text])
            print(probs_to_markdown(ret.detach().flatten().tolist(), sort_by_prob=True))
            ch, p = rnn.decoding(ret)[0]
            print(f'"{text}"的下一个字符：{ch}')
            text += ch
        plt.pause(0.1)


if __name__ == "__main__":
    main()
