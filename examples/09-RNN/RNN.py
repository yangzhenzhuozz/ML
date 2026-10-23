from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F
from math import sin

torch.manual_seed(42)  # 固定随机种子以便复现


class RNN:
    def __init__(_self, depth=2, width=2, learning_rate=0.05):
        """
        设置深度、宽度
        """
        _self.depth = depth
        _self.width = width
        _self.learning_rate = learning_rate
        _self.out_layer = torch.randn((1, width + 1), requires_grad=True)  # 输出层

        # 每个矩阵的行分别是神经元的权重+偏置
        # 第一行个隐藏层因为只有一个输入标量，所以是width*2（一个权重+一个偏置+自己的反馈）
        # 后面隐藏层每行需要前一层的所有输入权重+一个偏置+自己反馈
        _self.hidden_layers: list[Tensor] = [
            (torch.rand((width, 2 + width)) * 0.2 - 0.1).requires_grad_(),
            *[
                (torch.rand((width, width + 1 + width)) * 0.2 - 0.1).requires_grad_()
                for _ in range(depth - 1)
            ],
        ]

    def forward(_self, xs: list[float], y: float | None = None) -> float | Tensor:
        """
        Args:
                y (float | None): 如果y为非None值，表示是在进行训练，否则是在推导
        """
        with torch.set_grad_enabled(y is not None):
            # 初始化各隐藏层上一次的输出
            last_output: list[Tensor] = [
                torch.zeros(_self.width, 1) for _ in range(_self.depth)
            ]

            # 整个窗口的梯度
            total_loss: Tensor | None = None

            # 训练时保存每一个时刻的推导结果，推理时表示最终输出
            out: Tensor | None = None

            if y is not None:
                total_loss = torch.tensor(0.0)

            for x_idx in range(len(xs)):
                # 补1可以用于偏置，得到这样一个列向量 [[x_n][1]]，当前层的输入
                layer_input = torch.tensor([[xs[x_idx]], [1]])
                for l_idx in range(_self.depth):

                    # 组装前一层输出和当前层上一次输出
                    lastOut = last_output[l_idx]
                    cat = torch.cat([lastOut, layer_input], dim=0)

                    layer = _self.hidden_layers[l_idx]

                    inner_output = layer @ cat  # 矩阵计算

                    layer_output = F.tanh(inner_output)  # 激活函数

                    last_output[l_idx] = layer_output  # 缓存本次输出

                    # 在结果后面补一个标量1作为下一层输入，1用于偏置计算
                    layer_input = torch.cat([layer_output, torch.ones((1, 1))], dim=0)

                if y is not None:
                    # 输出层使用线性激活函数（现在系数暂时为1）
                    out = (_self.out_layer @ layer_input) * 1
                    loss = F.mse_loss(torch.tensor([[y]]), out)
                    assert total_loss is not None
                    total_loss = total_loss + loss
                elif x_idx == len(xs) - 1:
                    out = (_self.out_layer @ layer_input) * 1

            if y is not None:
                assert total_loss is not None
                return total_loss
            else:
                assert out is not None
                return out.item()

    def train(_self, xs: list[float], y: float):
        total_loss = cast(Tensor, _self.forward(xs, y))

        # 平均梯度，不计算平均梯度也没事，这里相当于是影响学习率而已，可以通过学习率调整
        avg_loss = total_loss / len(xs)

        avg_loss.backward()  # 反向传播

        with torch.no_grad():  # 关闭求导系统（别让计算图继续记录梯度的推导了）
            # 开始刷新参数
            assert _self.out_layer.grad is not None
            _self.out_layer -= _self.learning_rate * _self.out_layer.grad
            _self.out_layer.grad = None  # 清空梯度，避免下轮训练被累加
            for layer in _self.hidden_layers:
                assert layer.grad is not None
                layer -= _self.learning_rate * layer.grad
                layer.grad = None  # 清空梯度，避免下轮训练被累加


data_length = 3001  # 数据大小
window = 10  # 训练和推理窗口

# 生成训练数据
train_data: list[float] = [sin(step * 0.5) for step in range(data_length)]

rnn = RNN()

for step in range(data_length - 1 - window):
    rnn.train(train_data[step : step + window], train_data[step + window])

for step in range(data_length - 1 - window):
    real = train_data[step + window]
    prediction = rnn.forward(train_data[step : step + window])
    print(f"预测值：{prediction}，真实值：{real}，误差：{((prediction-real)*100):.6f}")
