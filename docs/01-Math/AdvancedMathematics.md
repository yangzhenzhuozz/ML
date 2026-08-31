# 高等数学

## 极限

定义：$\lim\limits_{x \to x_0} f(x) = A \iff \forall \varepsilon > 0$，$\exists \delta > 0$，使当 $0 < |x - x_0| < \delta$ 时，有 $|f(x) - A| < \varepsilon$。

只要你给出一个函数差值 $\varepsilon$，不管有多小，我总能在 $x_0$ 附近找到一个区间，使这个区间内的所有函数和 $A$ 的差距都小于这个 $\varepsilon$。_需要注意的是，这并不是说 $x \to x_0$ 的过程中，$f(x)$ 一定比前一刻的 $f(x)$ 更靠近 $A$。比如 $\lim\limits_{x \to 0} x \sin\left(\frac{1}{x}\right) = 0$ 就会在 0 附近来回剧烈震荡。但是不管怎么震荡，在去心邻域 $\mathring{U}(x_0, \delta)$ 的这个局部定义域内，函数值与 $A$ 的绝对误差都被 $\varepsilon$ 牢牢限制住了。这也说明函数值是整体趋近 A 的，因为随着 $\varepsilon$ 越来越小，函数值允许的变化范围也越来越小。

## 无穷小

定义：如果 $\lim\limits_{x \to x_0} f(x) = 0$，则称 $f(x)$ 为 $x \to x_0$ 时的无穷小。_无穷小描述的是一个动态变化的过程，必须是 $x$ 变化到 $x_0$ 过程中，$f(x)$ 才是无穷小。_

根据极限定义，针对无穷小的简化版写法：$\forall \varepsilon > 0$，$\exists \delta > 0$，使当 $0 < |x-x_0| < \delta$ 时，有 $|f(x)| < \varepsilon$，那么我们就可以说在 $x \to x_0$ 的过程中，$f(x)$ 是无穷小。

### 高阶无穷小

如果有 $\lim\frac{β}{α}=0$，则称 $β$ 是比 $α$ 高阶的无穷小，记作 $β=o(α)$

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

_我也经常会遇到两个问题：“这需要证？这还能证？”，但是上面这个定理在后面推到复合函数导数的时候很有用_
_极限是一个定值，而在 $x \to x_0$ 的时候，$f(x)$ 和极限值 $A$ 之间永远还差一点，$\lim$ 算子（$ε-δ$ 语言）则是让结果锁定了终点的目标值_

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

## 微分

如果函数的增量 $Δy$ 可以写成 $Δy=AΔx+o(Δx)$，则把 $AΔx$ 叫做 $f(x)$ 在点 $x_0$ 处对于增量 $Δx$ 的微分，记作 $\mathrm{d}y$，即
$$\mathrm{d}y=AΔx$$
因为根据导数的定义有：$y'=\lim\limits_{Δx \to 0}\frac{Δy}{Δx}$，由极限和无穷小的关系得：当 $Δx \to 0$ 时，$\frac{Δy}{Δx}=y'+α$
两边同时乘以 $Δx$ 得：$Δy=y'Δx+αΔx$（$Δx \to 0$ 时，$αΔx$是 $Δx$ 的高阶无穷小），因为 $\lim\limits_{Δx \to 0}\frac{αΔx}{Δx}=\lim α=0$

## 复合函数的导数

$[f(g(x))]'=f'(g(x)).g'(x)$
**证**：

> 令 $u=g(x)$ （这里是把 $u$ 当成变量看待）
> 则有 $f'(u)=\lim\limits_{Δu \to 0} \frac{Δy}{Δu}$
> 在 $Δx \to 0$ 的过程中，$Δu$ 也是趋近0的，即 $Δu \to 0$
> $\because$ $f'(u)=\lim\limits_{Δu \to 0} \frac{Δy}{Δu}$
> $\therefore$ $\frac{Δy}{Δu}=f'(u)+α$
> $\therefore$ $Δy=f'(u)Δu+αΔu$ （这里 $Δu \ne 0$，如果$Δu=0$，那么 $α$ 函数就不存在了）
> 为了让 $Δu=0$ 时也成立，我们打一个补丁
> $α=\begin{cases} \frac{Δy}{Δu}-f'(u), & Δu \ne 0 \\ 0, & Δu = 0 \end{cases}$
> 这个补丁是为了让 $Δy=f'(u)Δu+αΔu$ 在 $Δu = 0$ 和 $ Δu \ne 0 $ 的情况下都成立
> 在 $Δx \to 0$ 时 $\frac{Δy}{Δx}=f'(u)\frac{Δu}{Δx}+α\frac{Δu}{Δx}$
> 根据极限和无穷小的关系，有 $\lim\limits_{Δx \to 0}\frac{Δy}{Δx}=f'(u)u'(x)+0.u'(x)=f'(u)u'(x)$

### 补丁技巧

我们经常会遇到函数在某个点存在极限，但是却没有定义的情况，这时候可以对函数打一个补丁，让其成为一个连续函数
$\tilde{f}(x)=\begin{cases} f(x), & \text{原来的函数} \\L, & \text{可去间断点的极限值} L=\lim\limits_{x \to x_0}f(x) \end{cases}$
**例**

> $\lim\limits_{x \to 0} f(x)=\frac{\sin x}{x}$ ,我们知道极限值是1（两个重要极限之一，不能用洛必达否则就循环论证了）
> 所以我们可以补充成
> $f(x)=\begin{cases}\frac{\sin x}{x},& x \ne 0 \\ 1 , & x=0\end{cases}$

## 偏导数

设二元函数 $z=f(x,y)$ 在点 $(x_0,y_0)$ 的邻域有定义，当 $y$ 固定在 $y_0$，如果
$$\lim\limits_{Δx \to 0}\frac{f(x_0+Δx,y_0)-f(x_0,y_0)}{Δx}$$
存在，则这个极限称为 $f(x,y)$ 在点 $(x_0,y_0)$ 处对 $x$ 的偏导数。记作 $\frac{\partial z}{\partial x}|_{\substack{x=x_0 \\ y=y_0}}$ 或者 $f_x(x_0,y_0)$

如果我们把 $y$ 固定在 $y_0$ 处，让 $x$ 自由变化，那么可以想象出是一个平面$y=y_0$和函数曲面相交，相交处是一条曲线，这样就变成了一元函数求导了。

### 示例：马鞍面 ($z=x^2-y^2$) 和 $y=-0.5$ 相交得到一条曲线

<BiFunctionEcharts :exprs="[{x:'u',y:'v',z:'u^2-v^2'},{x:'u',y:'-0.5',z:'v'}]" :title="'双曲抛物面 z = x² - y²和y=-0.5'"/>

## 全微分

如果 $Δz=AΔx+BΔy+o(Δρ)$，那么我们就把 $AΔx+BΔy$ 叫做函数 $f(x,y)$ 在点 $(x,y)$ 处的全微分，并且记成 $\mathrm{d}z=AΔx+BΔy$
_这里是硬性定义的全微分公式，和一元函数类似，函数增量等于导数×变量增量+变量增量的高阶无穷小_

### 全微分-定理1

如果函数 $z=f(x,y)$ 在点$(x,y)$ 处可微，则偏导数存在，且有$A=f_x(x,y)$，$B=f_y(x,y)$

**证**：

> 因为 $Δz=AΔx+BΔy+o(Δρ)$ 要求在任何情况下都成立，所以我们令 $Δy=0$，就能得到 $Δz=AΔx+o(|Δx|)$
> 则有 $\frac{Δz}{Δx}=A+\frac{o(|Δx|)}{Δx}$
> 取一个 $Δx \to 0$ 时候的极限有 $\lim\limits_{Δx \to 0}\frac{Δz}{Δx}=A+\lim\limits_{Δx \to 0}\frac{o(|Δx|)}{Δx}=A$

### 全微分-定理2

如果函数 $z=f(x,y)$ 的偏导数在点 $(x,y)$ 处连续（偏导在 $(x,y)$ 的领域存在且连续），则在该点可微

重要定理，在复合函数多元函数中会用到证明中的式子

**证**：

> 因为函数的偏导数连续，则函数的全增量 $Δz=f(x+Δx,y+Δy)-f(x,y)$ 可以记为
> $Δz=[f(x+Δx,y+Δy)-f(x,y+Δy)]+[f(x,y+Δy)-f(x,y)]$
> 这是两个部分的增量
> 先考虑 $f(x+Δx,y+Δy)-f(x,y+Δy)$ 这部分
> 由拉格朗日中值定理得到 $f(x+Δx,y+Δy)-f(x,y+Δy)=f_x(ξ_1,y+Δy)Δx=f_x(x+θΔx,y+Δy)Δx,θ\in [0,1]$（因为这个多元函数只有x在变化，可以抽象成一元函数）
> 如果函数 $h(x)$ 连续，则有 $h(x+Δx)=h(x)+m$，其中 $m$ 是差值函数，且当 $Δx \to 0$ 时有 $m \to 0$
> 对应到多元函数时，如果函数 $h(x,y)$ 连续，则有 $h(x+Δx,y+Δy)=h(x,y)+m$，其中 $m$ 是差值函数，且当 $Δx \to 0$ 和 $Δy \to 0$ 时有 $m \to 0$
> 这时候因为 $f_x(x,y)$ 在点 $(x,y)$ 附近是连续的，由函数的连续性可以推导出：
> $f_x(x+θΔx,y+Δy)Δx=(f_x(x,y)+m)Δx$，且当 $Δx \to 0$ 和 $Δy \to 0$ 时有 $m \to 0$
>
> 考虑到第二部分 $f(x, y+\Delta y) - f(x, y)$（需要注意的是，这里的 $Δx=0$ 只是一个特例，实际上下面的推导在任意$Δx \to 0$，$Δy \to 0$ 时都成立）
> 因为偏导数 $f_{y}$ 连续，根据连续性，当 $Δx \to 0$，$Δy \to 0$ 时，它产生的差值函数 $n \to 0$。
> 同样我们可以将这一步的增量写为：$f(x,y+\Delta y)-f(x,y)=f_{y}(x,y)\Delta y+n\Delta y$
>
> 通过刚刚的推导，我们得到：
> $Δz=f_x(x,y)Δx+mΔx+f_y(x,y)Δy+n(Δy)$ ，且当 $Δx \to 0$ 和 $Δy \to 0$ 时有 $m \to 0$ $n \to 0$
> 接下来如果 $mΔx+nΔy$ 是 $ρ$ 的高阶无穷小，我们实际上就完成了证明
> $\lim\limits_{Δx \to 0,Δy \to 0} \dfrac{mΔx+nΔy}{\sqrt{Δx^2+Δy^2}}=\lim \dfrac{mΔx}{\sqrt{Δx^2+Δy^2}}+\lim \dfrac{nΔy}{\sqrt{Δx^2+Δy^2}}$
> = $\lim \big[ m \cdot \dfrac{Δx}{\sqrt{Δx^2+Δy^2}}\big]+\lim \big[n \cdot \dfrac{Δy}{\sqrt{Δx^2+Δy^2}}\big]$
> 因为 $|Δx|\le\sqrt{Δx^2+Δy^2}$ （直角边小于斜边，当Δy=0时，就相等了）
> 所以 $-1 \le \dfrac{Δx}{\sqrt{Δx^2+Δy^2}}\le 1$ （有界）
> 所以 $\lim \big( m \cdot \dfrac{Δx}{\sqrt{Δx^2+Δy^2}}\big)=0$ （无穷小×有界=无穷小）
> $nΔy$ 也是同样的处理方式
> 最终证明了 $mΔx+nΔy$ 是 $ρ$ 的高阶无穷小
> 证毕

## 方向导数

偏导数实际反应的是函数值沿着坐标轴方向的变化率，那么如果我们需要关注任意方向的变化率，就得请出方向导数了。
如果函数 $f(x,y)$ 在点 $(x_0,y_0)$ 可微分，那么函数在该点任意方向 $l$ 的方向导数均存在且有
$$\frac{\partial f}{\partial l} = \lim\limits_{Δl \to 0^+} \frac{f(x_0+Δl_x,y_0+Δl_y)-f(x_0,y_0)}{Δl}$$
上面式子中，$Δl_x$ 和 $Δl_y$ 指沿着射线 $l$ 变化了 $Δl$ 之后，$x$ 和 $y$ 的变化量
因为 $Δx=\cos α Δl$，$Δy=\cos β Δl$，其中 $α$ 和 $β$ 是 $l$ 和 $x$，$y$ 轴的夹角
根据全微分可以得到 $Δz=A\cos α Δl+B\cos β Δl+o(ρ)$，因为我们只考虑 $Δl$ 正方向的情况，这时候有 $Δl=ρ$，把$ρ$ 换成 $Δl$，然后两边同时除以 $Δl$ 得到
$\frac{Δz}{Δl}=A\cos α+B \cos β+\frac{o(Δl)}{Δl}$
$\frac{\partial f}{\partial l}=\lim\limits_{Δl \to 0^+}\left(A\cos α+B \cos β+\frac{o(Δl)}{Δl}\right)=A\cos α+B \cos β$

_这里需要注意,因为全微分定义中，要求后面的误差是 $o(Δl)$，这也是一些考题设计成有某两个特定的方向导数，但是不一定有任意方向导数。（x,y偏导数可以理解称两个特殊的方向导数，实际上如果函数可微，则知道了任意两个不平行方向的方向导数后，可以计算出任意第三个方向的方向导数）。同时方向导数也完全只考虑了一个方向（射线 $l$ 的方向），在反方向上的导数不一定存在_

## 梯度

考虑到有一个不在曲面最低点的一滴水，这滴水会顺着等高线最密的方向（即坡度最陡的方向）往下流动，从函数的角度来看就是函数值减小得最快的方向，这个方向就是数学里面**梯度**的反方向。
在方向导数的计算中，还有一个信息：$α+β=\frac{\pi}{2}$，那么就有$\frac{\partial f}{\partial l}=A \cos α+B \cos (\frac{\pi}{2}-α)=A \cos α + B \sin (α)$
令 $f(α)=A \cos α+B \sin α$，则有
$f'(α)=-A\sin α+B\cos α$
当 $f'(α)=0$ 时该点为一个驻点
$-A\sin α+B\cos α=0 \implies A\sin α=B\cos α \implies \overrightarrow{(A,B)} \parallel \overrightarrow{(\cos α,\sin α)}$
这里的 $\cos α$ 和 $\sin α$ 分别当成向量 $l$ 的 $x$ 和 $y$ 分量，即当射线 $l$ 的方向为 $(A,B)$ 或者 $(-A,-B)$ 时，该点是一个变化很快的方向
$f''(α)=-A\cos α-B\sin α$，我们把方向 $(A,B)$ 带入得到 $f''(α)=-A^2-B^2 \le 0$（该向量还没有归一化，但是我们讨论的是符号，所以不影响结果）
说明该方向是一个最快变大（二阶导小于0）的方向，这个方向就是**梯度**，另外一个方向就是减小最快的方向（水流下的方向）

梯度可以记作 $\mathrm{grad} f(x_0,y_0)=f_x(x_0,y_0)i+f_y(x_0,y_0)j$ ，同时这个向量的模长就是函数在该方向的变化率。

## 多元复合函数的导数

### 一元函数与多元函数复合的情况

如果有这样的函数 $z=f(u,v)$，$u=u(t)$，$v=v(t)$，$u$ 和 $v$ 在 $t$ 处可导，$f$ 在 $(u,v)$ 处有连续偏导数，那么当 $t$ 获得增量 $Δt$ 的时候， $u$、$v$ 也相应的获得增量 $Δu$、$Δv$，因此 $z$ 也获得全增量 $Δz$。根据[全微分-定理2](#全微分-定理2)中用拉格朗日中值定理和连续函数推导出来的等式有：
$$Δz=\frac{\partial z}{\partial u}Δu+\frac{\partial z}{\partial v}Δv+mΔu+nΔv$$
两边同时除以 $Δt$ 得到
$$\frac{Δz}{Δt}=\frac{\partial z}{\partial u}\frac{Δu}{Δt}+\frac{\partial z}{\partial v}\frac{Δv}{Δt}+m\frac{Δu}{Δt}+n\frac{Δv}{Δt}$$
因为 $Δt \to 0$ 时， $Δu$ 和 $Δv$ 都趋近0
取极限之后得到 $\lim\limits_{Δt \to 0}\dfrac{Δz}{Δt}=\dfrac{\partial z}{\partial t}=\dfrac{\partial z}{\partial u}\dfrac{Δu}{Δt}+\dfrac{\partial z}{\partial v}\dfrac{Δv}{Δt}$

### 多元函数与多元函数复合的情况

如果函数 $u$、$v$ 都在点 $(x,y)$ 处有对 $x$ 和 $y$ 的偏导数，且 $z=f(u,v)$ 在点 $(u,v)$ 处有连续偏导数，当我们在计算 $\dfrac{\partial z}{\partial x}$ 时，y是不变的常量，这时候就等价于一元函数的情况，即退化成这样的一元函数：$u=u(x)$，$v=v(x)$，则有：
$\dfrac{\partial z}{\partial x}=\dfrac{\partial z}{\partial u}\dfrac{\partial u}{\partial x}+\dfrac{\partial z}{\partial v}\dfrac{\partial v}{\partial x}$
同样的有：
$\dfrac{\partial z}{\partial y}=\dfrac{\partial z}{\partial u}\dfrac{\partial u}{\partial y}+\dfrac{\partial z}{\partial v}\dfrac{\partial v}{\partial y}$

**多元函数与多元函数复合的情况正是神经网络中反向传播的理论支撑**

> **例**：已知 $z=f(u(g(x,y),h(x,y)),v(φ(x,y),η(x,y)))$，且f、u、v、g、h、φ、η以及偏导数都是连续可导函数，求 $\dfrac{\partial z}{\partial x}$ 和 $\dfrac{\partial z}{\partial y}$
> **解**：
> 根据[全微分-定理2](#全微分-定理2)，我们可以得到如下式子
> $Δz=\dfrac{\partial z}{\partial u}Δu+\dfrac{\partial z}{\partial v}Δv+m_zΔu+n_zΔv$
> $Δu=\dfrac{\partial u}{\partial g}Δg+\dfrac{\partial u}{\partial h}Δh+m_uΔg+n_uΔh$
> $Δv=\dfrac{\partial v}{\partial φ}Δφ+\dfrac{\partial v}{\partial η}Δη+m_vΔφ+n_vΔη$
> 且内部的 $Δg$、$Δh$、$Δφ$、$Δη$ 都能继续写下去
> 最终得到：
> $Δz=\dfrac{\partial z}{\partial u}(\dfrac{\partial u}{\partial g}(\dfrac{\partial g}{\partial x}Δx+\dfrac{\partial g}{\partial y}Δy+m_gΔx+n_gΔy)+\dfrac{\partial u}{\partial h}(\dfrac{\partial h}{\partial x}Δx+\dfrac{\partial h}{\partial y}Δy+m_hΔx+n_hΔy)+m_u(\dfrac{\partial g}{\partial x}Δx+\dfrac{\partial g}{\partial y}Δy+m_gΔx+n_gΔy)+n_u(\dfrac{\partial h}{\partial x}Δx+\dfrac{\partial h}{\partial y}Δy+m_hΔx+n_hΔy))+\dfrac{\partial z}{\partial v}(\dfrac{\partial v}{\partial φ}(\dfrac{\partial φ}{\partial x}Δx+\dfrac{\partial φ}{\partial y}Δy+m_φΔx+n_φΔy)+\dfrac{\partial v}{\partial η}(\dfrac{\partial η}{\partial x}Δx+\dfrac{\partial η}{\partial y}Δy+m_ηΔx+n_ηΔy)+m_v(\dfrac{\partial φ}{\partial x}Δx+\dfrac{\partial φ}{\partial y}Δy+m_φΔx+n_φΔy)+n_v(\dfrac{\partial η}{\partial x}Δx+\dfrac{\partial η}{\partial y}Δy+m_ηΔx+n_ηΔy))+m_z(\dfrac{\partial u}{\partial g}(\dfrac{\partial g}{\partial x}Δx+\dfrac{\partial g}{\partial y}Δy+m_gΔx+n_gΔy)+\dfrac{\partial u}{\partial h}(\dfrac{\partial h}{\partial x}Δx+\dfrac{\partial h}{\partial y}Δy+m_hΔx+n_hΔy)+m_u(\dfrac{\partial g}{\partial x}Δx+\dfrac{\partial g}{\partial y}Δy+m_gΔx+n_gΔy)+n_u(\dfrac{\partial h}{\partial x}Δx+\dfrac{\partial h}{\partial y}Δy+m_hΔx+n_hΔy))+n_z(\dfrac{\partial v}{\partial φ}(\dfrac{\partial φ}{\partial x}Δx+\dfrac{\partial φ}{\partial y}Δy+m_φΔx+n_φΔy)+\dfrac{\partial v}{\partial η}(\dfrac{\partial η}{\partial x}Δx+\dfrac{\partial η}{\partial y}Δy+m_ηΔx+n_ηΔy)+m_v(\dfrac{\partial φ}{\partial x}Δx+\dfrac{\partial φ}{\partial y}Δy+m_φΔx+n_φΔy)+n_v(\dfrac{\partial η}{\partial x}Δx+\dfrac{\partial η}{\partial y}Δy+m_ηΔx+n_ηΔy))$
>
> 这样一个巨无霸式子，看着是不是眼花缭乱。
>
> 在我们计算 $\dfrac{\partial z}{\partial x}$ 时，y是一个常量，因为$Δy=0$，而且当 $Δx \to 0$ 时，有$Δg \to 0$、$Δh \to 0$、$Δφ \to 0$、$Δη \to 0$、$Δu \to 0$、$Δv \to 0$
> 所以我们在计算偏导数的时候，加上极限符号，可与把一大堆无穷小量消掉，变成：
> $\lim\limits_{Δx \to 0}\dfrac{Δz}{Δx}=\dfrac{\partial z}{\partial x}=\dfrac{\partial z}{\partial u}\left(\dfrac{\partial u}{\partial g}\dfrac{\partial g}{\partial x}+\dfrac{\partial u}{\partial h}\dfrac{\partial h}{\partial x}\right)+\dfrac{\partial z}{\partial v}\left(\dfrac{\partial v}{\partial \varphi }\dfrac{\partial \varphi }{\partial x}+\dfrac{\partial v}{\partial \eta }\dfrac{\partial \eta }{\partial x}\right)$
>
> 是不是看起来顺眼多了

如果一个函数、内部函数以及一直到最底层的变量都是连续可导的，那么其实我们可以用“套路”来计算偏导，像剥洋葱一样剥下去

1. 先写出 $Δz$ 的表达式，而且可以去掉无穷小量，是几元函数就把这几个元列出来再相加，在当前层级如果某个元和目标变量完全无关（即这个函数没有直接或者间接的使用到目标变量），则在该层完全无效（因为这个元的增量必然是0）
    1. $Δz=\dfrac{\partial z}{\partial u}Δu+\dfrac{\partial z}{\partial v}Δv$
2. 对前面的式子中每一个“微元” $Δ$ 继续拆分，直至达到目标偏导数
3. 最后两边同时除以目标 $Δ$

题外话：同济大学的高数课本偷懒了，上面证明需要偏导数连续，如果函数在某点可微，但是偏导数不连续（比如高频振荡），则无法用上面的步骤证明。

设函数 $z=f(u,v)$，$u=u(x,y)$，$v=v(x,y)$ ，如果 $z$、$u$、$v$ 在 $(x,y)$ 处可微，则同样有 $\dfrac{\partial z}{\partial x}=\dfrac{\partial z}{\partial u}\dfrac{\partial u}{\partial x}+\dfrac{\partial z}{\partial v}\dfrac{\partial v}{\partial x}$
**证**：

> 因为 $z$、$u$、$v$, 可微，则有
> $Δz=\dfrac{\partial z}{\partial u}Δu+\dfrac{\partial z}{\partial v}Δv+o(\sqrt{Δu^2+Δv^2})$
> 两边同时除以 $Δx$ 得
> $\dfrac{Δz}{Δx}=\dfrac{\partial z}{\partial u}\dfrac{Δu}{Δx}+\dfrac{\partial z}{\partial v}\dfrac{Δv}{Δx}+\dfrac{o(\sqrt{Δu^2+Δv^2})}{Δx}$
> 因为计算偏导数的时候，$Δy=0$，且 $u$、$v$ 可微，则有 $Δx \to 0$ 时 $Δu=u_xΔx+αΔx$ 和 $Δv=v_xΔx+βΔx$，其中 $α$ 和 $β$ 是无穷小。
> 上面根号里面的内容则可以写为 $ρ=|Δx| \sqrt{(u_x+α)^2+(v_x+β)^2}$
> 因为 $\lim\limits_{Δx \to 0,Δy=0}\sqrt{(u_x+α)^2+(v_x+β)^2}=C$（C是常数）
> 得到在 $Δx \to 0,Δy=0$ 时 $ρ$ 是 $Δx$ 的同阶无穷小
> 所以 $o(ρ)$ 是 $Δx$ 的高阶无穷小
> 所以得到$\lim\limits_{Δx \to 0,Δy=0}\dfrac{Δz}{Δx}=\dfrac{\partial z}{\partial u}\dfrac{\partial u}{\partial x}+\dfrac{\partial z}{\partial v}\dfrac{\partial v}{\partial x}$

## 下一步

在神经网络中，损失函数对权重的偏导数构成梯度，反向传播正是利用链式法则（复合函数求导）逐层计算梯度，再用梯度下降（沿负梯度方向更新参数）来最小化损失。
