import torch
from torch import Tensor
from torch.nn import functional as F
from math import sin

data_length = 10000  # 数据大小
window = 10  # 训练窗口
num_windows = 3000  # 训练次数
batch_size = 5  # 训练批次大小，暂未使用
learning_rate = 0.05  # 学习率

depth = 5  # 网络深度
width = 10  # 每层的宽度（整个网络宽度一致，方便残差连接）
hiddens: list[Tensor] = []  # 网络隐藏层，存储各个网络层的权重
out_layer: Tensor

# 每个矩阵的行分别是神经元的权重+偏置
# 第一行个隐藏层因为只有一个输入标量，所以是width*2（一个权重+一个偏置+自己的反馈）
hiddens.append(torch.rand((width, 2 + width), requires_grad=True))

# 第一次前向传播时，上一次的输出全部置0
for d in range(depth - 1):
    # 后面隐藏层每行需要前一层的所有输入权重+一个偏置+自己反馈
    hiddens.append(torch.rand((width, width + 1 + width), requires_grad=True))

out_layer = torch.randn((1, width + 1), requires_grad=True)  # 添加一个输出层


# 生成训练数据
primitiveData: list[float] = []
for step in range(0, data_length):
    primitiveData.append(sin(step * 0.5))


"""
这里不是对同一组数据反复训练，而是从窗口中取一段新数据
"""
# 开始训练
for window_start in range(num_windows):
    # 从原始数据取出训练窗口的数据
    total_loss = torch.tensor(0.0)
    last_output: list[Tensor] = []  # 各隐藏层上一次的输出
    for idx in range(depth):
        last_output.append(torch.zeros(width, 1))
    for idx in range(window):
        # 补1可以用于偏置，得到这样一个列向量 [[x_n][1]]
        tmp = torch.tensor([[primitiveData[window_start + idx]], [1]])
        for layerIdx in range(depth):
            layer = hiddens[layerIdx]
            lastOut = last_output[layerIdx]
            # 组装本层上一次的输出和输入数据
            tmp = layer @ torch.cat([lastOut, tmp], dim=0)
            tmp = F.tanh(tmp)  # 执行激活函数
            # 刷新上轮缓存
            last_output[layerIdx] = tmp
            # 在结果后面补一个标量1，用于下一层的偏置
            tmp = torch.cat([tmp, torch.ones((1, 1))], dim=0)
        out = (out_layer @ tmp) * 1  # 输出层使用线性激活函数（现在系数暂时为1）
        loss = F.mse_loss(torch.tensor([[primitiveData[window_start + idx + 1]]]), out)
        total_loss = total_loss + loss

    total_loss = total_loss / window  # 计算平均值
    total_loss.backward()  # 开始反向传播
    with torch.no_grad():  # 关闭求导系统（别让计算图继续记录梯度的推导了）
        # 开始刷新参数
        assert out_layer.grad is not None
        out_layer -= learning_rate * out_layer.grad
        out_layer.grad = None  # 清空梯度，避免下轮训练被累加
        for layer in hiddens:
            assert layer.grad is not None
            layer -= learning_rate * layer.grad
            layer.grad = None  # 清空梯度，避免下轮训练被累加
print("训练结束")

# 清理训练阶段的输出缓存
last_output: list[Tensor] = []  # 各隐藏层上一次的输出
for d in range(depth):
    last_output.append(torch.zeros(width, 1))

with torch.no_grad():
    data = primitiveData[0:window]
    for idx, x in enumerate(data):
        # 补1可以用于偏置，得到这样一个列向量 [[x_n][1]]
        tmp = torch.tensor([[x], [1]])
        for layerIdx in range(depth):
            layer = hiddens[layerIdx]
            lastOut = last_output[layerIdx]
            tmp = layer @ torch.cat(
                [lastOut, tmp], dim=0
            )  # 组装本层上一次的输出和输入数据
            tmp = F.tanh(tmp)  # 执行激活函数
            last_output[layerIdx] = tmp  # 刷新上轮缓存
            tmp = torch.cat(
                [tmp, torch.ones((1, 1))], dim=0
            )  # 在结果后面补一个标量1，用于下一层的偏置

        if idx == len(data) - 1:  # 给出最后一个输入之后再使用输出层
            out = (out_layer @ tmp) * 1  # 输出层使用线性激活函数（现在系数暂时为1）
            print(f"预测值：{out.item()}，真实值：{primitiveData[window]}")
            print(f"误差：{out.item()-primitiveData[window]}")
