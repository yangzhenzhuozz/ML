# 信息论

## 信息熵

信息熵是用来度量随机变量“不确定性”或“惊奇程度”的数学指标。

设现在有一个函数 $I(P(X=x_i))$ 表示随机变量 $X=x_i$ 时携带的信息量，那么他应该有如下性质：

- 当一个事件发生的概率为1时，因为我们知道其必然发生，所以这个事件没有携带任何有用信息。相反如果一个事件概率很低，我们就认为这个事件发生时，携带了更多信息。即 $I(p=1)=0,I(p=0)=\infty$，而且该函数在 $[0,1]$ 区间单调递减。
- 当两个独立事件同时发生时，两个事件携带的信息量应该相加。即 $I(P(AB))=I(P(A))+I(P(B))$。根据概率论知识我们知道 $P(AB)=p_a \times p_b$，所以函数 $I$ 应该能把参数的乘法拆分为加法

根据以上性质，我们很容易就想到对数函数，因为

$$\log_a(bc)=\log_ab+\log_ac$$

因为对数函数是单调递增的，我们令 $I(x)=-\log_ax$，就变成了单调递减。并且完美符合

$$
\begin{aligned}
\lim_{p \to 0^+} I(p) &= \lim_{p \to 0^+} (-\log_ap) =& \infty \\
I(1) &= -\log_a1 =& 0
\end{aligned}
$$

而且是单调递减函数。

对数函数的底数我们暂定为任意大于1的数。我们先研究一下信息熵，信息熵的定义就是信息量函数的期望，即

$$
H(X)=E(I(P(X)))
$$

带入具体的信息量函数之后得到

$$
H(X)=-\sum_{i=1}^n p_i \log_a p_i
$$

如果我们规定 $n$ 个等概率基本事件中，每个事件的信息熵为1，则有

$$
\begin{aligned}
 & H(X) = -\sum_{i=1}^n \frac{1}{n} \log_a \frac{1}{n} =1 \\
 \implies & -\log_a \frac{1}{n} =1\\
 \implies & \log_a n =1 \\
 \implies & a=n
\end{aligned}
$$

所以底数怎么选择，取决于我们对样本空间的定义。若规定“ $n$ 个等概率基本事件的平均信息量为 1”，则底数取 $n$。

总结：信息量函数是表示以n为基底时，某个随机变量所携带的信息量。信息熵则是信息量函数的期望，表示这个随机变量的平均信息量。

## 交叉熵

设离散随机变量X有一个真实概率分布P和一个假设概率分布Q，则称

$$
H(P,Q)=-\sum_{i=1}^n P(x_i) \log_a Q(x_i)
$$

为交叉熵。交叉熵表示我们使用一个基于分布Q的最优编码方案去编码服从真实分布P的平均信息量时得到的结果。交叉熵有一个有趣的性质

$$
H(P,Q) \ge H(P)
$$

当且仅当 $P=Q$ 时才能取等号。现在我们先证明这个不等式，再举例一个简单例子说明

**证**

> 在约束 $\sum_{i=1}^n q_i=1$ 下最小化交叉熵
>
> $$
> H(P,Q)=-\sum_{i=1}^n p_i\log_a q_i,\qquad a>1
> $$
>
> 构造拉格朗日函数
>
> $$
> \mathcal L = -\sum_{i=1}^n p_i\log_a q_i+\lambda\left(\sum_{i=1}^n q_i-1\right)
> $$
>
> 对 $q_i$ 求偏导：
>
> $$
> \frac{\partial \mathcal L}{\partial q_i} = -\frac{p_i}{q_i\ln a}+\lambda
> $$
>
> 令其为零，得
>
> $$
> q_i=\frac{p_i}{\lambda\ln a}
> $$
>
> 由 $\sum_i q_i=1$ 和 $\sum_i p_i=1$，得
>
> $$
> 1=\sum_i q_i = \frac{1}{\lambda\ln a}
> $$
>
> 所以
>
> $$
> \lambda\ln a=1
> $$
>
> 从而
>
> $$
> q_i=p_i
> $$
>
> $Q=P$ 是唯一全局最小点（这里的证明需要用到多元函数的泰勒公式、多元函数极大极小值、函数凸性，比较麻烦，所以直接给了结论）。于是
>
> $$
> \min_Q H(P,Q)=H(P,P)=H(P)
> $$
>
> 因此对任意 $Q$，
>
> $$
> H(P,Q)\ge H(P)
> $$
>
> 等号成立当且仅当
>
> $$
> Q=P
> $$

到这里有没有发现这个公式和我们在概率论通过似然估计出来的交叉熵完全对上了，我们通过极大似然估计得到的交叉熵公式为

$$-\ln L(p)=- \sum_{i=1}^n  \sum_{k=1}^c  \left[ y_{ik} \ln \hat{y_{ik} } \right]$$

公式中 $\hat{y_{ik}}$ 是我们假设的概率，$y_{ik}$ 是真实数据的类别的0-1标签，并且因为c个类别中，只允许一个类别为1，其他类别为0，所以真实数据 $y_{ik}$ 本质就是一个概率分布表：1 表示该类别的真实概率为 100%，0 表示概率为 0。内层求和跟信息论的交叉熵完全一致

$$
H(y_{ik},\hat{y_{ik}})=-\sum_{k=1}^c  \left[ y_{ik} \ln \hat{y_{ik} } \right]
$$

就表示了第 $i$ 组样本的交叉熵，而

$$-\ln L(p)=- \sum_{i=1}^n  \sum_{k=1}^c  \left[ y_{ik} \ln \hat{y_{ik} } \right]$$

表示 $n$ 组样本的交叉熵总和

## KL散度

因为真实分布的信息熵 $H(P)$ 是一个常数，所以我们可以用 $H(P,Q)-H(P)$ 来评估我们的预测分布 $Q$ 和真实分布 $P$ 的差异，这个评估指标就是KL散度

$$
D_{KL}(P \parallel Q)=H(P,Q)-H(P)
$$

由交叉熵的定义可以知道，KL散度是一个大于0的值，越低就表示我们的预设越接近真实分布。需要注意的是KL散度是不对称的，即

$$
D_{KL}(P \parallel Q) \ne D_{KL}(Q \parallel P)
$$

所以KL散度不能作为评估的绝对标准，只能做相对标准。不对称的证明如下：

**证：**

> $$
> \begin{aligned}
>   \Delta =& D_{KL}(P \parallel Q) - D_{KL}(Q \parallel P) \\
>          =& H(P,Q)-H(P)-H(Q,P)+H(Q) \\
>          =& \left\{-\left[\sum_{i=1}^n (p_i \log_a q_i)-\sum_{i=1}^n (p_i \log_a p_i)\right]\right\}-\left\{-\left[\sum_{i=1}^n (q_i \log_a p_i)-\sum_{i=1}^n (q_i \log_a q_i)\right]\right\} \\
>          =& \sum_{i=1}^n \left[ (p_i+q_i)(\log_a p_i-\log_a q_i) \right]
> \end{aligned}
> $$
>
> 利用对数函数的单调递增性有
>
> 1. $p_i=q_i$ 时，$\Delta=0$
> 2. $p_i \lt q_i$ 时，有 $p_i+q_i \gt 0,\log_a p_i - \log_a q_i \lt 0$，$\Delta \lt 0$
> 3. $p_i \gt q_i$ 时，有 $p_i+q_i \gt 0,\log_a p_i - \log_a q_i \gt 0$，$\Delta \gt 0$
>
> 所以两者是不对称的
