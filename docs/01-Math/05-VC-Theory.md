# VC理论

## 霍夫丁不等式

### Hoeffding 引理

凸函数有个性质：函数 $f$ 上任选两点 $a,b (a \lt b)$，然后将 $a,b$ 连接成一条直线 $l$，有 $f(x)$ 在直线 $l$ 下方，只有在点 $a,b$ 处函数和直线相等。
直线方程为：

$$
y=\frac{b-x}{b-a}f(a)+\frac{x-a}{b-a}f(b)
$$

所以我们在函数 $e^{\lambda x}$ 上任选两点，有

$$
e^{\lambda x} \le \frac{b-x}{b-a}e^{\lambda a}+\frac{x-a}{b-a}e^{\lambda b}, x \in [a,b]
$$

我们计算一下上面的期望，有

$$
\begin{aligned}
E(e^{\lambda X}) &\le E(\frac{b-X}{b-a}e^{\lambda a}+\frac{X-a}{b-a}e^{\lambda b}) \\
&\le \frac{b-E(X)}{b-a}e^{\lambda a}+\frac{E(X)-a}{b-a}e^{\lambda b}
\end{aligned}
$$

如果我们预设 $E(X)=0$ 且 $X \in [a,b]$，则必然有 $a \le 0 \le b$ ，而且有

$$
\begin{aligned}
    E(e^{\lambda X}) &\le \frac{b}{b-a}e^{\lambda a}+\frac{-a}{b-a}e^{\lambda b} \\
    E(e^{\lambda X}) &\le \frac{be^{\lambda a}-ae^{\lambda b}}{b-a} \\
    E(e^{\lambda X}) &\le e^{\lambda a}\frac{b-ae^{\lambda (b-a)}}{b-a} \\
    E(e^{\lambda X}) &\le e^{\lambda a}(\frac{b}{b-a}+\frac{-a}{b-a}e^{\lambda (b-a)})
\end{aligned}
$$

令

$$
p=\frac{-a}{b-a}
$$

则

$$
1-p=\frac{b}{b-a}
$$

带入 $E(e^{\lambda X})$ 得到

$$
E(e^{\lambda X}) \le e^{\lambda a}(1-p+pe^{\lambda (b-a)})
$$

再令 $\theta = \lambda(b-a)$，则 $\lambda a=-\theta p$，上式可以化简为

$$
E(e^{\lambda X}) \le e^{-p\theta} (1 - p + p e^\theta)
$$

因为 $x=e^{\ln x}$ ,则上式可以写为

$$
\begin{aligned}
E(e^{\lambda X}) &\le e^{-p\theta} (1 - p + p e^\theta) \\
E(e^{\lambda X}) &\le e^{-p\theta} e^{\ln (1 - p + p e^\theta)} \\
E(e^{\lambda X}) &\le e^{-p\theta+\ln (1 - p + p e^\theta)}
\end{aligned}
$$

构造一个辅助函数

$$
\phi(\theta)=-p\theta+\ln (1 - p + p e^\theta)
$$

则有

$$
\begin{aligned}
    \phi'(\theta) &= -p+\frac{pe^\theta}{1-p+pe^\theta} \\
    \phi''(\theta) &= \frac{pe^\theta(1-p+pe^\theta)-pe^\theta pe^\theta}{(1-p+pe^\theta)^2} \\
                   &= \frac{pe^\theta(1-p)}{(1-p+pe^\theta)^2}
\end{aligned}
$$

我们在 $\theta=0$ 处用对 $\phi(\theta)$ 进行泰勒展开，并使用拉格朗日余项得到

$$
\begin{aligned}
    \phi(\theta) &= \phi(0)+\phi'(0)\theta+\frac{\phi''(\xi)}{2!}\theta^2\\
                 &= 0 + 0 + \frac{p(1-p)e^\xi}{(1-p+pe^\xi)^2}\frac{\theta^2}{2} \\
                 &= 0 + 0 + \frac{pe^\xi}{1-p+pe^\xi}(1-\frac{pe^\xi}{1-p+pe^\xi})\frac{\theta^2}{2}
\end{aligned}
$$

上式中，存在类似 $q(1-q)$ 的结构，这是一个开口向下的二次函数，最大值为 $\frac{1}{4}$

所以有

$$
\phi(\theta) \le \frac{1}{4} \times \frac{\theta^2}{2}=\frac{\theta^2}{8}
$$

即

$$
E(e^{\lambda X}) \le e^{\phi(\theta)} \le e^{\frac{\lambda^2(b-a)^2}{8}}
$$

### hoeffding 不等式

通过切比雪夫大数定理的证明，我们知道

$$
P(|\frac{1}{n}\sum_{i=1}^n X_i- \frac{1}{n}\sum_{i=1}^n E(X_i)|\ge \xi) \le \frac{\sum\limits_{i=1}^n D(X_i)}{n^2\xi^2} \le \frac{nc}{n^2\xi^2} = \frac{c}{n\xi^2}
$$

这个公式的收敛速度是

$$\frac{1}{n}$$

如果我们n变大一百倍，概率肯定也至少收敛了一百倍。但是这个速度实际上是可以远远超过一百倍，我们可以这样证明：
设

$$S_n=X_1+X_2+\cdots+X_n \qquad (X_i \text{相互独立})$$

那么我们研究一下这个概率

$$P(S_n-E[S_n] \ge t)$$

因为 $S_n-E[S_n]$ 不是恒大于0的随机变量，所以不能直接套用马尔可夫不等式，这时候可以借助指数函数的两个性质：

- $e^x \gt 0$
- $e^x$ 在整个实数定义域单调递增

在 $\lambda \gt 0$ 的情况下有

$$
P(S_n-E[S_n] \ge t)=P(e^{\lambda(S_n-E[S_n])} \ge e^{\lambda t})
$$

这时候就可以用马尔可夫不等式了

$$
\begin{aligned}
    P(e^{\lambda(S_n-E[S_n])} \ge e^{\lambda t}) &\le \frac{E(e^{\lambda(S_n-E[S_n])})}{e^{\lambda t}} \\
                              &= \frac{E(e^{\lambda(\sum\limits_{i=1}^n (X_i-E[X_i]))})}{e^{\lambda t}} \\
                              &= \frac{E(\prod\limits_{i=1}^n e^{\lambda(X_i-E[X_i])})}{e^{\lambda t}} \quad \text{（利用 } X_i \text{ 相互独立）}\\
                              &= \frac{\prod\limits_{i=1}^n E(e^{\lambda(X_i-E[X_i])})}{e^{\lambda t}}
\end{aligned}
$$

因为

$$
\begin{aligned}
    E(X_i-E[X_i]) &= E(X_i)-E(E(X_i)) \\
                  &= E(X_i)-E(X_i) \\
                  &= 0
\end{aligned}
$$

通过 hoeffding 引理，得到

$$
\begin{aligned}
        & \frac{\prod\limits_{i=1}^n E(e^{\lambda(X_i-E[X_i])})}{e^{\lambda t}} \\
    \le & \frac{\prod\limits_{i=1}^n e^{\dfrac{\lambda^2(b_i-a_i)^2}{8}}}{e^{\lambda t}} \\
      = & \frac{e^{\sum\limits_{i=1}^n \dfrac{\lambda^2(b_i-a_i)^2}{8}}}{e^{\lambda t}} \\
      = & e^{\dfrac{\lambda^2}{8} \sum\limits_{i=1}^n(a_i-b_i)^2 - \lambda t}
\end{aligned}
$$

指数部分是一个开口向上的二次函数，则

$$
\lambda = \frac{4t}{\sum_{i=1}^n (b_i - a_i)^2}
$$

时，指数部分取得最小值

$$
-\frac{2t^2}{\sum_{i=1}^n (b_i - a_i)^2}
$$

即

$$
\begin{aligned}
    P(S_n-E[S_n] \ge t) &= P(e^{\lambda(S_n-E[S_n])} \ge e^{\lambda t}) \\
                        &\le \frac{E[e^{\lambda(S_n-E[S_n])}]}{e^{\lambda t}} \\
                        &\le e^{-\dfrac{2t^2}{\sum\limits_{i=1}^n (b_i - a_i)^2}}
\end{aligned}
$$

这是单边霍夫丁不等式，即只考虑 $S_n$ 超出了 $E(S_n)$ 距离 t的情况，如果我们考虑一下 $S_n$ 低于 $E(S_n)$ 距离t的情况，因为是对称的，所以有

$$
P(E[S_n] - S_n \ge t) \le e^{-\frac{2t^2}{ \sum\limits_{i=1}^n (b_i-a_i)^2 }}
$$

我们将两者联合得到

$$
P( |S_n - E[S_n] |\ge t) \le 2 e^{-\frac{2t^2}{ \sum\limits_{i=1}^n (b_i-a_i)^2 }}
$$

## Union Bound

现在假设我们在进行二分类任务，记随机变量 $Z_i$ 为第 $i$ 个样本预测是否错误，即

$$
Z= \begin{cases}
   1 & \text{预测正确} \\
   0 & \text{预测错误}
\end{cases}
$$

记

$$
S_n=Z_1+Z_2+\cdots+Z_n=\sum_{i=1}^n Z_i
$$

为预测正确次数，如果我们的每次预测都是独立的，并且成功预测概率相等，则有

$$
E(S_n)=\sum_{i=1}^n E(Z_i)=n \times R(h)
$$

$R(h)$ 表示模型 $h$ 的正确率函数，带入霍夫丁不等式有

$$
P(|S_n-E(S_n)|\ge t)\le 2 e ^{-\frac{2t^2}{\sum\limits_{i=1}^n (b_i-a_i)^2}}
$$

因为 $Z_i$ 的上下界为 $[0,1]$，所以指数部分的分母变成了

$$
\begin{aligned}
     & \sum\limits_{i=1}^n (b_i-a_i)^2 \\
   = & \sum\limits_{i=1}^n 1^2 \\
   = & n
\end{aligned}
$$

于是有

$$
P(|S_n-E(S_n)|\ge t) \le 2e^{-\frac{2t^2}{n}}
$$

随着预测次数的增加，我们的模型会xxx，目前我完全无法理解这个定理有什么用，感觉就是在说大数定理，切比雪夫大数定律还证明了会收敛到算术平均值呢。这个地方

---

---

你现在手握 **Hoeffding 不等式** 的完整第一性原理证明，其实已经攻克了统计学习理论中最硬核的**概率论地基**。接下来通往 **VC 维（Vapnik-Chervonenkis Dimension）** 的路线非常清晰，你只需要再补齐 **4 块核心拼图**（大约需要再写 3～4 篇你刚才那种体量的笔记），就能彻底打通 VC 维泛化界的完整证明。

下面按“从直觉痛点到严密证明”的顺序，为你规划完整的进阶路线图：

---

### 路线全景图：你现在处于哪一步？

```
[已完成] 单个假设的收敛：Hoeffding 不等式
   │
   ▼  （痛点：模型会从假设空间 H 中挑选表现最好的那个，必须同时控制所有假设！）
[第 1 步] 有限假设空间：Union Bound（联合界） ──► 揭示为什么“无限空间”会失效
   │
   ▼  （破局：无限个假设在有限样本上的“有效行为”其实是有限的！）
[第 2 步] 组合计数工具：增长函数（Growth Function） + 打散（Shattering） + VC 维定义
   │
   ▼  （核心定理：把指数爆炸变成多项式增长！）
[第 3 步] 组合数学巅峰：Sauer-Shelah 引理（VC 维最硬核的纯数学证明）
   │
   ▼  （收官：把组合计数塞回 Hoeffding 不等式！）
[第 4 步] 对称化技巧（Symmetrization / 幽灵样本） ──► 最终得到 VC 维泛化误差界

```

---

### 详细拆解：你需要依次掌握的 4 个模块

#### 模块 1：有限假设空间与一致收敛（Uniform Convergence）

- **为什么需要它**：Hoeffding 不等式只能保证**固定一个**模型 $h$ 时，经验误差 $\hat{R}(h)$ 接近真实误差 $R(h)$。但训练时我们是在整个假设空间 $\mathcal{H}$ 里挑误差最小的那个 $\hat{h}$，因此必须要求 $\mathcal{H}$ 里**所有假设同时收敛**。
- **需要的数学工具**：极其简单，只需要概率论最基础的 **Union Bound（布尔不等式）**：$P(\bigcup_{i=1}^M A_i) \le \sum_{i=1}^M P(A_i)$。
- **你会推导出的结论**：当假设空间大小 $\vert{}\mathcal{H}\vert{} = M$ 有限时，结合 Hoeffding 不等式立刻得到：

$$P\left(\sup_{h \in \mathcal{H}} \vert{}\hat{R}(h) - R(h)\vert{} \ge \epsilon\right) \le 2M \exp(-2n\epsilon^2)$$

- **暴露出的致命问题**：如果 $\mathcal{H}$ 是二维平面上的所有直线（感知机），$M = \infty$，右边直接变成 $+\infty$，不等式彻底失效！这正是引出 VC 维的直接动机。

---

#### 模块 2：核心概念——打散（Shattering）、增长函数与 VC 维

- **核心直觉**：平面上的直线虽然有无穷多条，但如果你在平面上固定 $n$ 个样本点，这些直线能对这 $n$ 个点做出的**二分类结果组合**是有限的（最多 $2^n$ 种）。
- **你需要掌握的 3 个定义**：

1. **对分（Dichotomy）**：假设空间 $\mathcal{H}$ 作用在样本集 $S = \{x_1, \dots, x_n\}$ 上所能产生的所有 $\{0, 1\}^n$ 标签组合。
2. **增长函数（Growth Function）$m_{\mathcal{H}}(n)$**：在所有大小为 $n$ 的样本集中，能产生出的**最大**对分数量。显然 $m_{\mathcal{H}}(n) \le 2^n$。
3. **打散（Shatter）与 VC 维 $d_{VC}$**：如果存在某 $n$ 个点能被 $\mathcal{H}$ 产生出全部 $2^n$ 种组合，就称这 $n$ 个点能被 $\mathcal{H}$ **打散**。**能被打散的最大点数 $n$，就是 $\mathcal{H}$ 的 VC 维**（一旦样本数超过 $d_{VC}$，增长函数 $m_{\mathcal{H}}(n)$ 就再也达不到 $2^n$ 了）。

- **建议亲手算几个例子**：试着用笔记算一下“一维数轴上的区间”、“二维平面上的直线”、“二维平面上的圆”各自的 VC 维是多少（这是理解定义的最佳练习）。

---

#### 模块 3：Sauer-Shelah 引理（组合数学证明）

- **它在证明中的地位**：相当于你刚刚证完的 **Hoeffding 引理**，是整个 VC 维理论中最精妙的纯数学引理。
- **它证明了什么**：一旦样本数 $n > d_{VC}$，增长函数 $m_{\mathcal{H}}(n)$ 就会发生相变——从原本的指数增长 $2^n$，瞬间坍缩为**多项式增长** $O(n^{d_{VC}})$：

$$m_{\mathcal{H}}(n) \le \sum_{i=0}^{d_{VC}} \binom{n}{i} \le \left(\frac{en}{d_{VC}}\right)^{d_{VC}}$$

- **需要的数学工具**：完全不需要高深数学，只需要高中/大学低年级的**数学归纳法**和**组合数恒等式** $\binom{n}{k} = \binom{n-1}{k} + \binom{n-1}{k-1}$。

---

#### 模块 4：对称化技巧（Symmetrization）与最终泛化界

- **为什么不能直接把 $M$ 替换成 $m_{\mathcal{H}}(n)$？**
  你可能会想：既然在 $n$ 个样本上只有 $m_{\mathcal{H}}(n)$ 种有效假设，那直接把模块 1 公式里的 $M$ 换成 $m_{\mathcal{H}}(n)$ 不就行了吗？
  **这里有个巨大的陷阱**：$m_{\mathcal{H}}(n)$ 只能控制**经验误差** $\hat{R}(h)$ 上的分类结果，但**真实误差** $R(h)$ 依赖于整个未知的数据分布（无穷多个潜在样本），无法直接用 $n$ 个样本的对分去替代！
- **需要的神级技巧——幽灵样本（Ghost Sample）**：
  这是 Vapnik 和 Chervonenkis 最天才的构造：

1. 凭空想象另一份独立同分布的“幽灵测试集” $S' = \{x'_1, \dots, x'_n\}$，用 $S'$ 上的误差 $\hat{R}'(h)$ 去替代不可知的真实误差 $R(h)$（这一步叫**对称化引理**，证明过程中只需要用到切比雪夫不等式）。
2. 一旦把 $R(h)$ 替换成了 $\hat{R}'(h)$，整个问题就完全限制在了 $S \cup S'$ 这 **$2n$ 个有限样本**上！此时就可以合法地使用增长函数 $m_{\mathcal{H}}(2n)$ 和 Union Bound 了。
3. 最后再利用交换 $S$ 和 $S'$ 中对应样本的对称性（引入伯努利随机符号 $\sigma_i \in \{-1, +1\}$），直接套用你刚证完的 **Hoeffding 不等式**，最终推导出教科书上的经典 **VC 维泛化误差界**：

$$P\left(\sup_{h \in \mathcal{H}} \vert{}\hat{R}(h) - R(h)\vert{} \ge \epsilon\right) \le 4 m_{\mathcal{H}}(2n) \exp\left(-\frac{n\epsilon^2}{8}\right)$$

---

### 总结：你还需要多少前置知识？

好消息是，**你几乎不需要补充任何新的高深数学分支**。你已经掌握的微积分、基础概率论（期望、独立性、切比雪夫/马尔可夫/Hoeffding 不等式）已经覆盖了 90% 的工具需求。唯一需要热身的是**组合数学中的归纳法技巧**（用于 Sauer-Shelah 引理）。

**推荐的下一步行动**：
不要一上来就啃幽灵样本和对称化。你可以紧接着刚才的笔记，先写一篇 **《有限假设空间的一致收敛 + 增长函数与 VC 维》**，亲手推导几个几何图形（如平面直线、凸多边形）的打散过程，把组合直觉建立起来，然后再攻克 Sauer-Shelah 引理。
