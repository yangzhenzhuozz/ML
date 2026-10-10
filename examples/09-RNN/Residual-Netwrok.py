import torch
import torch.nn.functional as F

"""
可以看到用了PyTorch之后，我们的残差网络代码大幅度的变少了，不需要再小心的处理反向传播
"""

# XOR问题模型
# 行向量，批量数据
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# torch.manual_seed(42)  # 固定随机种子以便复现

layout_buffer: list[torch.Tensor] = []
layout_out_buffer: list[torch.Tensor] = []
out_neural_buffer: torch.Tensor | None = None

width = 3  # 宽度
deep = 2  # 深度
skip = 2  # 残差连接数
learning_rate = 0.5  # 学习率
epoch = 3000


# 补一个全1的列用于计算偏置
def add_bias(x: torch.Tensor) -> torch.Tensor:
    # 创建一个与 x 行数一致、只有 1 列且全为 1.0 的张量
    ones = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
    # 沿列方向（dim=1）拼接在原矩阵右侧
    return torch.cat([x, ones], dim=1)


for epoch_idx in range(0, epoch):
    last_layout = X
    for idx in range(0, deep):
        # hidden是隐藏层，每一列分别是ω1、ω2、b1
        if idx >= len(layout_buffer):
            hidden = torch.randn((last_layout.shape[1] + 1, width), requires_grad=True)
            layout_buffer.append(hidden)
        else:
            hidden = layout_buffer[idx]
        # 结果向量的每一行对应一组输入数据的结果，每一列为对应神经元的输出
        last_layout = (add_bias(last_layout) @ hidden).sigmoid()

        # 残差连接
        if idx % skip == 0 and idx != 0:
            last_layout = last_layout + layout_out_buffer[idx - skip]
        elif idx == deep - 1 and idx != 0:  # 最后一跳不够深度了
            last_layout = last_layout + layout_out_buffer[idx - idx % skip]

        if idx >= len(layout_out_buffer):
            layout_out_buffer.append(last_layout)
        else:
            layout_out_buffer[idx] = last_layout

    if out_neural_buffer is None:
        out_neural = torch.randn((last_layout.shape[1] + 1, 1), requires_grad=True)
        out_neural_buffer = out_neural
    else:
        out_neural = out_neural_buffer
    loss = F.binary_cross_entropy((add_bias(last_layout) @ out_neural).sigmoid(), Y)

    # 反向传播更新梯度
    loss.backward()
    all_weights: list[torch.Tensor] = layout_buffer + [out_neural_buffer]

    # 接下来的计算不需要记录自动求导了
    with torch.no_grad():
        for w in all_weights:
            assert w.grad is not None
            w -= learning_rate * w.grad
            assert w.grad is not None
            w.grad.zero_()  # 清空梯度

assert out_neural_buffer is not None

print("\n预测结果（概率）：")
with torch.no_grad():
    last_layout = X
    for idx, w in enumerate(layout_buffer):
        last_layout = (add_bias(last_layout) @ w).sigmoid()
        if idx % skip == 0 and idx != 0:
            last_layout = last_layout + layout_out_buffer[idx - skip]
        elif idx == deep - 1 and idx != 0:  # 最后一跳不够深度了
            last_layout = last_layout + layout_out_buffer[idx - idx % skip]

        if idx >= len(layout_out_buffer):
            layout_out_buffer.append(last_layout)
        else:
            layout_out_buffer[idx] = last_layout

    preds = (add_bias(last_layout) @ out_neural_buffer).sigmoid()
    for row_x, row_y, row_p in zip(X.tolist(), Y.tolist(), preds.tolist()):
        print(f"  {row_x} -> {row_p[0]:.4f}   (目标 {row_y[0]})")
