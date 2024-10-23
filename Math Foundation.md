# Math Equation

[TOC]

## Permutation

$P(n,r)=^{b}P_{r}=_{n}P_{r}=\dfrac{n!}{(n-r)!}$

## Combination

$C(n,r)=^{b}C_{r}=_{n}C_{r}=\left(\begin{array}{l} n \\ r \end{array}\right)=\dfrac{n!}{r!(n-r)!}$

## Summation

$$
\begin{alignat}{3}
&\sum_{i=1}^n\left(a_i+b_i\right) &&=\sum_{i=1}^n a_i+\sum_{i=1}^n b_i\\
&\sum_{i=1}^n\left(\lambda a_i\right)&& =\lambda \sum_{i=1}^n a_i\\
&\sum_{i=1}^n\left(a_i-b_i\right)&& =\sum_{i=1}^n a_i-\sum_{i=1}^n b_i\\
&\sum_{i=1}^n a_i && =na  \\
&\sum_{i=1}^n\left(a_i-\bar{a}\right) && =0\\
&\sum_{i=1}^n\left(a_i-\bar{a}\right)^2 && =\sum_{i=1}^n a_i^2-n \bar{a}^2\\
&\sum_{i=1}^n a w^{i-1} && =\frac{a\left(1-w^n\right)}{1-w}\\
&\sum_{i=1}^{\infty} a w^{i-1} && =\frac{a}{1-w}, \quad \text { for }|w|<1\\
&\sum_{i=1}^n \sum_{j=1}^i a_{i j} && =\sum_{j=1}^n \sum_{i=j}^n a_{i j}\\
&\end{alignat}
$$

Multi
$$
\begin{alignat}{3}
&\prod_{i=1}^n a_i b_i && =\left(\prod_{i=1}^n a_i\right)\left(\prod_{b=1}^n b_j\right)\\
&\prod_{i=1}^n\left(\lambda a_i\right) && =\lambda^n \prod_{i=1}^n a_i \\
&\prod_{i=1}^n\left(\frac{a_i}{b_i}\right) && =\frac{\prod_{i=1}^n a_i}{\prod_{i=1}^n b_i}\left(\text { for } b_i \neq 0, i=1,2, \ldots, n\right)\\
&\prod_{i=1}^n a && =a^n\\
&\prod_{i=1}^n \prod_{j=1}^k a_{i j} && =\prod_{j=1}^k \prod_{i=1}^n a_{i j}\\
&\prod_{i=1}^n \prod_{j=1}^i a_{i j} && =\prod_{j=1}^n \prod_{i=j}^n a_{i j}
\end{alignat}
$$

## Expectation and Standard Deviation

$$
\begin{alignat}{3}
&E (X)&&=\mu_{ x }=\Sigma x p ( x )  (\text{discrete case})\\
&E ( g ( X ))&&=\Sigma g ( x ) p ( x )=\mu_{g(X)}(\text { discrete case })\\
&E ( a )&& = a, (\text{constant})\\
&E ( a X)&&= a * E ( X )\\
&E ( a \pm X )&&= a \pm E ( X )\\
&E ( a \pm b X )&&= a \pm b E ( X )\\
&E [( a \pm X ) * b ]&&=( a \pm E ( X )) * b\\
&E ( X + Y )&&= E ( X )+ E ( Y )\\
&E ( X Y )&&= E ( X ) E ( Y ), (\text{X,Y are independent})\\
&E[X^2]&&=Var[X]+E[X]^2\\
&\newline
&\operatorname{COV}(X, Y)&&=E[(X-E(X)) *(Y-E(Y)]=E(X Y)-E(X) E(Y)\\
&\operatorname{cov}(X, Y)&&=0 (\text{X,Y are independent})\\
&\newline
&V( a )&&=0\\
&V(a \pm X)&&=V(X)\\
&V ( a \pm b X )&&= b ^2 * V ( X )=\sigma^2{ }_{ bX }\\
&V(X \pm Y)&&=V(X)+V(Y) \pm 2 \operatorname{COV}(X, Y)=\sigma^2 X \pm Y\\
&V ( X \pm Y )&&= V ( X )+ V ( Y ) (\text{X,Y are independent})\\
&V ( X )&&= E \left[\left( X -\mu_{ X }\right)^2\right]= E \left( X ^2\right)-\mu_{ X ^2}\\
&V ( aX )&&= a ^2 * V ( X )
&\end{alignat}
$$

$$
&E(aX) & =a E(x)\\
&E(X+b) & =E(X)+b\\
&SD(aX) & =|a|SD(X)\\
&SD(X+b) & =SD(X)
$$

$$
\begin{aligned}
& E\left[(X-\bar{x})^2\right] \\
= & E\left(X^2-2 X \bar{x}+\bar{x}^2\right) \\
= & E\left(X^2\right)-2 \bar{x} E(X)+\bar{x}^2 \\
= & E\left(X^2\right)-2 \bar{x} \cdot \bar{x}+\bar{x}^2 \\
= & E\left(X^2\right)-\bar{x}^2 \\
= & E\left(X^2\right)-E(X)^2
\end{aligned}$\mu=\overline{x}=E(X)=\sum x_ip(x_i)$
$$

mean value
$$
\begin{aligned}
Var(x)&=\sigma^2\\
&=\sum(x_i-\mu)^2 p(x_i)\\
&=E(X^2)-E(X)^2\\
&=E[(X-E(X))^2]\\
&=E[(X-\overline{x})^2]

\end{aligned}
$$

## $SS_{Res}\ \&\&\ SS_{T}$  Sum of Square

为了推导 $S S_{R E S}$ 包含 $S S_T$ 的形式, 我们可以从它们的定义出发。
首先定义以下变量：

1. $S S_T$ (总平方和) :
   
   $$
   S S_T=\sum_{i=1}^n\left(y_i-\bar{y}\right)^2
   $$
2. $S S_{R E S}$ (残差平方和) :
   
   $$
   S S_{R E S}=\sum_{i=1}^n\left(y_i-\hat{y}_i\right)^2
   $$
   
   其中, $\hat{y}_i$ 是预测值, 给定为 $\hat{y}_i=\hat{\beta}_0+\hat{\beta}_1 x_i$ 。
3. $S S_{R E G}$ （回归平方和）:
   
   $$
   S S_{R E G}=\sum_{i=1}^n\left(\hat{y}_i-\bar{y}\right)^2
   $$
   
   根据方差分解的属性, 我们有:
   
   $$
   S S_T=S S_{R E S}+S S_{R E G}
   $$
   
   我们的目标是使用 $S S_T$ 来表示 $S S_{R E S}$ 。
   将 $S S_{R E S}$ 展开:
   
   $$
   \begin{aligned}
   & S S_{R E S}=\sum_{i=1}^n\left(y_i-\hat{y}_i\right)^2 \\
   & =\sum_{i=1}^n\left(y_i-\bar{y}+\bar{y}-\hat{y}_i\right)^2 \\
   & =\sum_{i=1}^n\left[\left(y_i-\bar{y}\right)+\left(\bar{y}-\hat{y}_i\right)\right]^2
   \end{aligned}
   $$
   
   现在, 我们可以将上述表达式进一步展开并组合:
   $S S_{R E S}=\sum_{i=1}^n\left(y_i-\bar{y}\right)^2+\sum_{i=1}^n\left(\bar{y}-\hat{y}_i\right)^2+2 \sum_{i=1}^n\left(y_i-\right.$
   $\bar{y})\left(\bar{y}-\hat{y}_i\right)$
   注意到:
   
   $$
   \sum_{i=1}^n\left(y_i-\bar{y}\right)=0
   $$
   
   和
   
   $$
   \sum_{i=1}^n \hat{y}_i=n \bar{y}
   $$
   
   因此，交叉项为 0 :
   
   $$
   2 \sum_{i=1}^n\left(y_i-\bar{y}\right)\left(\bar{y}-\hat{y}_i\right)=0
   $$
   
   于是, 我们得到:
   
   $$
   \begin{aligned}
   & S S_{R E S}=\sum_{i=1}^n\left(y_i-\bar{y}\right)^2+\sum_{i=1}^n\left(\bar{y}-\hat{y}_i\right)^2 \\
   & S S_{R E S}=S S_T-S S_{R E G}
   \end{aligned}
   $$
   
   这就是我们所需要的 $S S_{R E S}$ 关于 $S S_T$ 的表达式。

## Probabiltiy Distribution

| Variable | Function | mean  variance | Usage |
| -------- | -------- | -------------- | ----- |
|          |          |                |       |
|          |          |                |       |
|          |          |                |       |
|          |          |                |       |
|          |          |                |       |
|          |          |                |       |
|          |          |                |       |

## ANOVA (**An**alysis **o**f **V**ari**a**nce)

1. 总平方和 $S S_T$
   $S S_T=\sum_{i=1}^n\left(y_i-\bar{y}\right)^2$
   它表示数据中的总变异性。
2. 回归平方和 $S S_R$
   $S S_R=\sum_{i=1}^n\left(\hat{y}_i-\bar{y}\right)^2$
   其中 $\hat{y}_i$ 是对于给定的 $x_i, y_i$ 的预测值。
3. 残差平方和 $S S_{R e s}$ 或 $S S E$
   $S S_{\text {Res }}=\sum_{i=1}^n\left(y_i-\hat{y}_i\right)^2$
   或者可以使用:
   
   $$
   S S_{\text {Res }}=S S_T-S S_R
   $$
4. $S_{x y}$ （协方差之和）
   
   $$
   S_{x y}=\sum_{i=1}^n\left(x_i-\bar{x}\right)\left(y_i-\bar{y}\right)
   $$
5. $S_{x x}$
   
   $$
   S_{x x}=\sum_{i=1}^n\left(x_i-\bar{x}\right)^2
   $$
   
   它表示 $x$ 的变异性。
6. 残差均方 $M S_{\text {Res }}$
   
   $$
   M S_{\text {Res }}=\frac{S S_{\text {Res }}}{n-2}
   $$
   
   在简单线性回归中, 分母中的 2 是因为我们估计了两个参数：截距 $\beta_0$ 和斜率 $\beta_1$ 。
   以上展开式描述的是简单线性回归的统计量。如果涉及多元线性回归, 其中有多个解释变量, 这些公式将会有所不同。
   $$
   \begin{alignat}{1}
   R^2\\
   r^2

\end{alignat}
$$
## 常用缩写

1. **SS** - Sum of Squares (平方和)
2. **DF** - Degrees of Freedom（自由度）
   - 通常用于卡方检验、ANOVA、回归分析等统计方法。
3. **MS** - Mean Square（均方）
   - 在方差分析（ANOVA）中，MS是SS除以相应的DF。
   - 例如：MSbetween=SSbetweenDFbetweenMSbetween=DFbetweenSSbetween
4. **SE** - Standard Error（标准误）
   - 一个统计量（如均值或回归系数）的标准差。
5. **SD** - Standard Deviation（标准差）
   - 描述数据集或概率分布的离散程度或扩散度。
6. **CV** - Coefficient of Variation（变异系数）
   - 描述数据的相对变异性，通常用于比较不同单位或数量级的数据集的变异性。
7. 
8. **CI** - Confidence Interval（置信区间）
   - 一个数值范围，用于估计一个未知的参数，如均值、比率或差异。
9. **IQR** - Interquartile Range（四分位距）
   - 描述数据的中心50%的变异性，即第三四分位数（Q3）和第一四分位数（Q1）之间的差。
10. **p-value**（p值）
    - 用于假设检验中，描述观测到的数据或更极端的数据在零假设为真的情况下出现的概率。
11. **R²** - Coefficient of Determination（决定系数）
    - 描述了自变量解释因变量变异性的比例，常用于回归分析。
12. **ANOVA** - Analysis of Variance（方差分析）
    - 一种统计方法，用于分析由不同分组或类别引起的变异。
13. **MLE** - Maximum Likelihood Estimation（最大似然估计）
    - 一种估计方法，常用于统计建模和推断。
$$
\sqrt{10}
$$

$s$
