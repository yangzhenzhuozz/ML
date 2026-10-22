import torch
import math

# 基础数据
x = []
for step in range(0, 70):
    x.append(math.sin(step * 0.5))

# 滑动窗口
window = 10

print(str(x))
