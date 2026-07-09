# 导数

## 极限

定义：$\lim\limits_{x \to x_0} f(x) = A \iff \forall \varepsilon > 0$，$\exists \delta > 0$，使当 $0 < |x - x_0| < \delta$ 时，有 $|f(x) - A| < \varepsilon$。

只要你给出一个函数差值 $\varepsilon$，不管有多小，我总能在 $x_0$ 附近找到一个区间，使这个区间内的所有函数和 $A$ 的差距都小于这个 $\varepsilon$。_需要注意的是，这并不是说 $x \to x_0$ 的过程中，$f(x)$ 一定比前一刻的 $f(x)$ 更靠近 $A$。比如 $\lim\limits_{x \to 0} x \sin\left(\frac{1}{x}\right) = 0$ 就会在 0 附近来回剧烈震荡。但是不管怎么震荡，在去心邻域 $\mathring{U}(x_0, \delta)$ 的这个局部定义域内，函数值与 $A$ 的绝对误差都被 $\varepsilon$ 牢牢限制住了。这也说明函数值是整体趋近 A 的，因为随着 $\varepsilon$ 越来越小，函数值允许的变化范围也越来越小。\_

## 无穷小

定义：如果 $\lim\limits_{x \to x_0} f(x) = 0$，则称 $f(x)$ 为 $x \to x_0$ 时的无穷小。_无穷小描述的是一个动态变化的过程，必须是 $x$ 变化到 $x_0$ 过程中，$f(x)$ 才是无穷小。_

根据极限定义，针对无穷小的简化版写法：$\forall \varepsilon > 0$，$\exists \delta > 0$，使当 $0 < |x-x_0| < \delta$ 时，有 $|f(x)| < \varepsilon$，那么我们就可以说在 $x \to x_0$ 的过程中，$f(x)$ 是无穷小。

### 无穷小与极限

在 $x \to x_0$ 的过程中， $\lim\limits_{x \to x_0} f(x)=A$ 的充要条件是 $f(x)=A+\alpha$，其中 $\alpha$ 是 $x \to x_0$ 时的无穷小。

证明如下：

1. **必要性**（已知 $\lim\limits_{x \to x_0} f(x)=A$，求证 $\alpha = f(x)-A$ 是 $x \to x_0$ 时的无穷小）：
   **证** ：

    > $\because \lim\limits_{x \to x_0}f(x)=A$
    > $\therefore \forall \varepsilon > 0$，$\exists \delta > 0$，当 $0 < |x-x_0| < \delta$ 时，有 $|f(x)-A| < \varepsilon$
    > 令 $\alpha = f(x)-A$，则有 $\forall \varepsilon > 0$，$\exists \delta > 0$，当 $0 < |x-x_0| < \delta$ 时，有 $|\alpha| < \varepsilon$
    > 根据无穷小的定义，$\alpha$ 是 $x \to x_0$ 时的无穷小
    > 证毕

2. **充分性**（已知 $f(x)=A+\alpha$，且 $\alpha$ 是 $x \to x_0$ 时的无穷小，求证 $\lim\limits_{x \to x_0} f(x)=A$）：
   **证**：
    > $\because \alpha$ 是 $x \to x_0$ 时的无穷小
    > $\therefore \forall \varepsilon > 0$，$\exists \delta > 0$，使当 $0 < |x-x_0| < \delta$ 时，有 $|\alpha| < \varepsilon$
    > 将 $\alpha=f(x)-A$ 代入，则有 $|f(x)-A| < \varepsilon$
    > 根据极限的定义，可知 $\lim\limits_{x \to x_0} f(x)=A$
    > 证毕

以上内容其实说明了一个概念：**$\lim\limits_{x \to x_0} f(x)=A$ 表达的是一个渐进的过程。随着 $x$ 靠近 $x_0$，函数值 $f(x)$ 越来越接近 $A$。我们研究的是函数在 $x_0$ 去心邻域的变化趋势，所以就算 $f(x)$ 在 $x_0$ 处没有函数值（比如在该点无定义），极限依然可以存在。如果 $f(x)$ 在 $x_0$ 处不仅有定义，而且其函数值 $f(x_0)$ 刚好等于极限值 $A$，那么函数在该点就是连续的（图像表现为可以一笔画出来）。**

*我也经常会遇到两个问题：“这需要证？这还能证？”，但是上面这个定理在后面推到复合函数导数的时候很有用*
*极限是一个定值，而在 $x \to x_0$ 的时候，$f(x)$ 和极限值 $A$ 之间永远还差一点，$\lim$ 算子（$ε-δ$ 语言）则是让结果锁定了终点的目标值*

## 极限的四则运算法则

1. $\lim [ f(x) \pm g(x) ]=\lim f(x) \pm \lim g(x)$
2. $\lim [ f(x) . g(x) ]=\lim f(x) . \lim g(x)$
3. $\lim \left[ \frac{f(x)}{g(x)} \right]=\frac{\lim f(x)}{\lim g(x)}$ （$\lim g(x) \ne 0$）

## 导数的定义

定义：
$$f'(x_0)=\lim\limits_{x \to x_0}\frac{f(x)-f(x_0)}{x-x_0}$$
当然也有好几种其他形式的写法，如
$$f'(x_0)=\lim\limits_{Δx \to 0} \frac{f(x_0+Δx)-f(x_0)}{Δx}$$

> **例**：求 $f(x)=x^2$ 的导数
> **解**：根据定义有
> $f'(x)=\lim\limits_{Δx \to 0}\frac{f(x+Δx)-f(x)}{Δx}$
> =$\lim\limits_{Δx \to 0}\frac{(x+Δx)^2-x^2}{Δx}$
> =$\lim\limits_{Δx \to 0}\frac{x^2+2xΔx+(Δx)^2-x^2}{Δx}$
> =$2x+\lim\limits_{Δx \to 0}Δx$
> =$2x$

## 函数和差积商的导数

1. $[f(x) \pm g(x)]'=f'(x)+g'(x)$
   **证**：
    > 原式=$\lim\frac{f(x+Δx)+g(x+Δx)-[f(x)+g(x)]}{Δx}$
    > =$\lim\frac{f(x+Δx)-f(x)}{Δx} \pm \lim\frac{g(x+Δx)-g(x)}{Δx}$(由极限的和差得)
    > =$f'(x) \pm g'(x)$
2. $[f(x)g(x)]'=f'(x)g(x)+f(x)g'(x)$
   **证**：
    > 原式=$\lim \frac{f(x+Δx)g(x+Δx)-f(x)g(x)}{Δx}$
    > =$\lim \frac{[f(x+Δx)-f(x)]g(x+Δx)+f(x)g(x+Δx)-f(x)g(x)}{Δx}$
    > =$\lim \left[ \frac{f(x+Δx)-f(x)}{Δx}g(x+Δx) \right]+\lim \left[ f(x)\frac{g(x+Δx)-g(x)}{Δx}\right]$
    > =$f'(x) \lim g(x+Δx)+ \lim f(x) g'(x)$
    > =$f'(x)g(x)+f(x)g'(x)$
3. $\left[ \frac{f(x)}{g(x)} \right]'=\frac{f'(x)g(x)-f(x)g'(x)}{g(x)^2}$
   **证**：
    > 原式=$\lim \dfrac{\dfrac{f(x+Δx)}{g(x+Δx)}-\dfrac{f(x)}{g(x)}}{Δx}$
    > =$\lim \dfrac{\dfrac{f(x+Δx)g(x)-f(x)g(x+Δx)}{g(x+Δx)g(x)}}{Δx}$
    > =$\lim \dfrac{\dfrac{f(x+Δx)g(x)}{Δx}-\dfrac{f(x)g(x+Δx)}{Δx}}{g(x+Δx)g(x)}$
    > =$\dfrac{f'(x)g(x)-f(x)g'(x)}{g(x)^2}$

## 复合函数的导数

$[f(g(x))]'=f'(g(x)).g'(x)$
**证**：

> 令 $u=g(x)$ （这里是把 $u$ 当初变量看待）
> 则有 $f'(u)=\lim\limits_{Δu \to 0} \frac{Δy}{Δu}$
> 在 $Δx \to 0$ 的过程中，$Δu$ 也是趋近0的，即 $Δu \to 0$
> $\because$ $f'(u)=\lim\limits_{Δu \to 0} \frac{Δy}{Δu}$
> $\therefore$ $\frac{Δy}{Δu}=f'(u)+α$
> $\therefore$ $Δy=f'(u)Δu+αΔu$ （这里 $Δu \ne 0$，如果$Δu=0$，那么 $α$ 函数就不存在了）
> 为了让 $Δu=0$ 时也成立，我们打一个补丁
> $α=\begin{cases} \frac{Δy}{Δu}-f'(u), & Δu \ne 0 \\ 0, & Δu = 0 \end{cases}$
> 这时候 $Δy=f'(u)Δu+αΔu$ 在 $Δu = 0$ 和 $ Δu \ne 0 $ 的情况下都是成立的
>  在 $Δx \to 0$ 时 $\frac{Δy}{Δx}=f'(u)\frac{Δu}{Δx}+α\frac{Δu}{Δx}$
> 根据极限和无穷小的关系，有 $\lim\limits_{Δx \to 0}\frac{Δy}{Δx}=f'(u)u'(x)+0.u'(x)$

## 偏导数

## 方向导数

## 梯度

## 下一步

现在知道复合函数的导数以及梯度怎么计算之后，就可以进入后面神经网络的反向传播了
