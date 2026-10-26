from typing import Sequence, cast

import torch
from torch import Tensor
from torch.nn import functional as F
from math import sin

import matplotlib.pyplot as plt

torch.manual_seed(42)  # 固定随机种子以便复现

"""
将26个字母和空格使用one-hot编码，输入小说种的文本训练，然后测试模型是否能输出单词或者句子
"""
encoding_space = 27


class RNN:
    def __init__(self, depth=2, width=2):
        self.depth = depth

        self.width = width

        def init_layer(shape: Sequence[int]):
            return (torch.randn(shape)).requires_grad_()

        self.out_layer = init_layer(
            (encoding_space, width + 1)
        )  # 输出层是一个encoding_space行向量

        # 每个矩阵的行分别是神经元的权重+偏置
        # 第一行个隐藏层因为只有一个输入向量，所以是width+8（encoding_space个权重(one-hot编码)+一个偏置+自己的反馈）
        # 后面隐藏层每行需要前一层的所有输入权重+一个偏置+自己反馈
        self.hidden_layers: list[Tensor] = [
            init_layer((width, encoding_space + 1 + width)),
            *[init_layer((width, width + 1 + width)) for _ in range(depth - 1)],
        ]

    def encoding(self, s: str) -> Tensor:
        """
        生成一个二阶one-hot编码张量
        """
        if s == " ":
            return F.one_hot(torch.tensor([26], dtype=torch.long), num_classes=27).t()
        else:
            return F.one_hot(
                torch.tensor([ord(s) - ord("a")], dtype=torch.long), num_classes=27
            ).t()

    def forward(self, xs: str, ys: str | None = None) -> Tensor:
        with torch.set_grad_enabled(ys is not None):
            # 初始化各隐藏层上一次的输出
            last_output: list[Tensor] = [
                torch.zeros(self.width, 1) for _ in range(self.depth)
            ]

            # 整个窗口的梯度
            total_loss: Tensor | None = None

            # 训练时保存每一个时刻的推导结果，推理时表示最终输出
            out: Tensor | None = None

            if ys is not None:
                total_loss = torch.tensor(0.0)

            for data_idx in range(len(xs)):
                # 补1可以用于偏置，得到这样一个列向量 [[x_n][1]]，当前层的输入
                encoding_vec = self.encoding(xs[data_idx])
                layer_input = torch.cat([encoding_vec, torch.tensor([[1]])], dim=0)
                for l_idx in range(self.depth):

                    # 组装前一层输出和当前层上一次输出
                    lastOut = last_output[l_idx]
                    cat = torch.cat([lastOut, layer_input], dim=0)

                    layer = self.hidden_layers[l_idx]

                    inner_output = layer @ cat  # 矩阵计算

                    layer_output = F.tanh(inner_output)  # 激活函数

                    last_output[l_idx] = layer_output  # 缓存本次输出

                    # 在结果后面补一个标量1作为下一层输入，1用于偏置计算
                    layer_input = torch.cat([layer_output, torch.ones((1, 1))], dim=0)

                if ys is not None:
                    # 输出层使用线性激活函数（现在系数暂时为1）
                    inner_linear = self.out_layer @ layer_input  # 线性部分
                    out = F.softmax(inner_linear, dim=0)  # 激活函数
                    target = self.encoding(ys[data_idx])  # 对输出编码
                    selected = out * target  # 逐元素相乘，保留正确字符对应的概率
                    sum = selected.sum(dim=0)  # 累加得到正确字符的概率
                    loss = -torch.log(sum)  #  负对数
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

        return total_loss


rnn = RNN()

with open("data.txt") as file:
    loss_list = []
    for line in file:
        line = line.strip()
        loss = rnn.train(
            line[:-1],
            line[1:],
            learning_rate=0.01,
        )
        loss_list.append(loss.item())

input = "alice"
ret = rnn.forward(input)

print(f'"{input}"的下一个字符是:{ret.tolist()}')

plt.figure()
plt.plot([x * 0.5 for x in range(len(loss_list))], loss_list, label="损失值")
plt.title("Loss示例")  # 标题
plt.xlabel("X 轴 (时间)")  # X轴标签
plt.ylabel("Y 轴 (损失)")  # Y轴标签
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格线

plt.show()
