import math
import random
import torch
import matplotlib.pyplot as plt

# ============================================================
# 1. 配置
# ============================================================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

data_length = 10000
window = 10
num_windows = 3000
learning_rate = 0.05
depth = 2
width = 2

print(f"Device: {device}")

# ============================================================
# 2. 生成正弦序列
# primitiveData[i] = sin(i * 0.5)
# ============================================================
steps = torch.arange(data_length, dtype=dtype, device=device)
primitiveData = torch.sin(steps * 0.5)

# ============================================================
# 3. 初始化参数
#
# 隐藏层输入为：
# [当前输入 x, 该层上一时刻隐藏状态 h, bias=1]
#
# h_new = tanh(W @ layer_input)
#
# 第一层输入维度为 1，后续层输入维度为 width。
# 每层权重形状：[width, input_dim + width + 1]
# ============================================================
hidden_weights = []

for layer in range(depth):
    input_dim = 1 if layer == 0 else width

    W = torch.nn.Parameter(
        torch.rand(width, input_dim + width + 1, device=device, dtype=dtype) * 0.2 - 0.1
    )

    hidden_weights.append(W)

# 输出层：最后一层隐藏状态 + bias -> 一个预测值
# 保留输出层单独初始化的方式
output_weights = torch.nn.Parameter(torch.rand(width + 1, device=device, dtype=dtype))

parameters = hidden_weights + [output_weights]


# ============================================================
# 4. 手写 RNN 前向传播
# 每个窗口开始时隐藏状态初始化为零
# ============================================================
def forward_window(x_window):
    hidden = [torch.zeros(width, device=device, dtype=dtype) for _ in range(depth)]

    prediction = None

    for t in range(len(x_window)):
        x = x_window[t].reshape(1)

        for layer in range(depth):
            layer_input = torch.cat(
                [x, hidden[layer], torch.ones(1, device=device, dtype=dtype)]
            )

            hidden[layer] = torch.tanh(hidden_weights[layer] @ layer_input)

            x = hidden[layer]

        output_input = torch.cat(
            [hidden[-1], torch.ones(1, device=device, dtype=dtype)]
        )

        prediction = output_weights @ output_input
    assert prediction is not None
    return prediction.squeeze()


# ============================================================
# 5. 训练
#
# 每次更新取一个长度为 window 的滑动窗口。
# 在窗口内的每一个时刻预测下一个值。
# 对整个窗口的 MSE 进行一次反向传播和 SGD 更新。
# ============================================================
num_windows = min(num_windows, data_length - window)

for update in range(num_windows):

    start = update

    x_window = primitiveData[start : start + window]
    y_window = primitiveData[start + 1 : start + window + 1]

    # 清零梯度
    for p in parameters:
        if p.grad is not None:
            p.grad.zero_()

    # 每个窗口的隐藏状态从零开始
    hidden = [torch.zeros(width, device=device, dtype=dtype) for _ in range(depth)]

    predictions_in_window = []

    # 在窗口内逐时刻递归计算
    for t in range(window):
        x = x_window[t].reshape(1)

        for layer in range(depth):
            layer_input = torch.cat(
                [x, hidden[layer], torch.ones(1, device=device, dtype=dtype)]
            )

            hidden[layer] = torch.tanh(hidden_weights[layer] @ layer_input)

            x = hidden[layer]

        output_input = torch.cat(
            [hidden[-1], torch.ones(1, device=device, dtype=dtype)]
        )

        pred = output_weights @ output_input
        predictions_in_window.append(pred.squeeze())

    preds = torch.stack(predictions_in_window)

    # 窗口内所有时刻的一步预测均方误差
    loss = torch.mean((preds - y_window) ** 2)

    # BPTT：沿窗口内的时间步反向传播
    loss.backward()

    # 手写 SGD 参数更新
    with torch.no_grad():
        for p in parameters:
            p -= learning_rate * p.grad

    if update % 100 == 0 or update == num_windows - 1:
        print(f"update={update:4d} | " f"window MSE={loss.item():.8f}")

# ============================================================
# 6. 全序列单步预测评估
#
# 评估规则与训练一致：
# 1. 每次输入前 window 个真实值
# 2. 隐藏状态从零开始
# 3. 预测窗口之后的下一个值
# 4. 不更新模型参数
# ============================================================
predictions = []
targets = []

with torch.no_grad():
    for start in range(data_length - window):

        x_window = primitiveData[start : start + window]

        pred = forward_window(x_window)
        target = primitiveData[start + window]

        predictions.append(pred)
        targets.append(target)

predictions = torch.stack(predictions)
targets = torch.stack(targets)

errors = predictions - targets
abs_errors = torch.abs(errors)

mse = torch.mean(errors**2).item()
rmse = math.sqrt(mse)
mae = torch.mean(abs_errors).item()
max_abs_error = torch.max(abs_errors).item()

print("\n" + "=" * 56)
print("全序列单步预测评估结果")
print(f"评估点数     : {len(targets)}")
print(f"MSE          : {mse:.8f}")
print(f"RMSE         : {rmse:.8f}")
print(f"MAE          : {mae:.8f}")
print(f"最大绝对误差 : {max_abs_error:.8f}")
print("=" * 56)

# 打印前 10 个预测点
print("\n前 10 个预测点：")
print("index\ttrue\t\tpred\t\tabs_error")

for i in range(min(10, len(targets))):
    print(
        f"{i + window}\t"
        f"{targets[i].item():+.6f}\t"
        f"{predictions[i].item():+.6f}\t"
        f"{abs_errors[i].item():.6f}"
    )

# ============================================================
# 7. 准备绘图数据
# ============================================================
target_indices = torch.arange(window, data_length).cpu().numpy()

true_np = targets.detach().cpu().numpy()
pred_np = predictions.detach().cpu().numpy()
abs_err_np = abs_errors.detach().cpu().numpy()

# ============================================================
# 8. 图一：全序列真实值与预测值
# ============================================================
plt.figure(figsize=(14, 5))

plt.plot(target_indices, true_np, label="True", linewidth=1.5)

plt.plot(target_indices, pred_np, label="Prediction", linewidth=1.0, alpha=0.85)

plt.title("Full-sequence One-step Prediction")
plt.xlabel("Sequence index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ============================================================
# 9. 图二：全序列逐点绝对误差
# ============================================================
plt.figure(figsize=(14, 4))

plt.plot(target_indices, abs_err_np, label="Absolute Error", linewidth=0.9)

plt.title("Absolute Prediction Error Across the Sequence")
plt.xlabel("Sequence index")
plt.ylabel("|Prediction - Target|")
plt.legend()
plt.grid(True)
plt.tight_layout()

# ============================================================
# 10. 图三：放大查看前 200 个预测点
# ============================================================
n_zoom = min(200, len(targets))

plt.figure(figsize=(14, 5))

plt.plot(target_indices[:n_zoom], true_np[:n_zoom], label="True", linewidth=1.8)

plt.plot(target_indices[:n_zoom], pred_np[:n_zoom], label="Prediction", linewidth=1.2)

plt.title(f"Prediction Detail (First {n_zoom} Points)")
plt.xlabel("Sequence index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
