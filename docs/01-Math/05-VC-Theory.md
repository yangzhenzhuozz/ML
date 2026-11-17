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
                        &\le e^{-\dfrac{2t^2}{\sum_{i=1}^n (b_i - a_i)^2}}
\end{aligned}
$$

如果所有的 $X_i$ 都落在区间 $[a,b]$ （我们可以取一个区别包含所有的 $a_i,b_i$），则霍夫丁不等式可以得到指数级别的收敛

如果你决定投入 3 到 5 天的时间系统性地攻克它，建议不要去啃晦涩的原版论文，而是选择经典的现代教材，分阶段推进：

- **核心推荐教材**：Shai Shalev-Shwartz 的《Understanding Machine Learning: From Theory to Algorithms》（这本书对有数理统计基础的人极为友好）。
- **高效通关四步法**：

1. **第一天（概率基础与 PAC 框架）**：复习 Hoeffding 不等式，理解有限假设空间下的联合界（Union Bound），掌握 PAC（Probably Approximately Correct）的定义。
2. **第二天（成长函数与萨乌尔引理）**：攻克二分法（Dichotomy）和成长函数，理解萨乌尔引理的推导逻辑（不需要死抠每一个代数细节，但要理解其核心思想：成长函数要么指数爆炸，要么是多项式）。
3. **第三天（VC 维与经典计算）**：掌握 VC 维的严格定义，亲手推导一维阈值、二维感知机、以及 $d$ 维超平面的 VC 维。
4. **第四天（泛化界与结构风险最小化）**：推导或理解 VC 泛化误差界，理解经验风险最小化（ERM）和结构风险最小化（SRM）的本质。
