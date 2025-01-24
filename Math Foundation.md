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

## Multiply

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
&V ( aX )&&= a ^2 * V ( X )\\
&E(aX) && =a E(x)\\
&E(X+b) && =E(X)+b\\
&SD(aX) && =|a|SD(X)\\
&SD(X+b) && =SD(X)\\
\end{alignat}
$$



**Expectation $E(X)$**
$$
\begin{aligned}
E(X)= &\mu=\overline{x}=\sum x_ip(x_i)\\
E\left[(X-\bar{x})^2\right]
= & E\left(X^2-2 X \bar{x}+\bar{x}^2\right) \\
= & E\left(X^2\right)-2 \bar{x} E(X)+\bar{x}^2 \\
= & E\left(X^2\right)-2 \bar{x} \cdot \bar{x}+\bar{x}^2 \\
= & E\left(X^2\right)-\bar{x}^2 \\
= & E\left(X^2\right)-E(X)^2
\end{aligned}
$$



**Varance$\mathrm{Var}(X)$**
$$
\begin{aligned}
Var(x)&=\sigma^2\\
&=\sum(x_i-\mu)^2 p(x_i)\\
&=E(X^2)-E(X)^2\\
&=E[(X-E(X))^2]\\
&=E[(X-\overline{x})^2]

\end{aligned}
$$


## Probability

$$
\begin{alignat}{3}
& \textbf{Bayes}\\\\
&P(A|B)&&=\dfrac{P(A\cap B)}{P(B)}=\dfrac{P(A)P(B|A)}{P(B)}=\dfrac{P(B)P(A|B)}{P(B)}\\
&P(A,B)&&=P(A,B)\\
&P(A_{0:T})&&=P(A_0,A_1,A_2,\cdots, A_T)\\\\
&\textbf{Chain Rule of Probability}\\\\
&P(A_0, A_1, A_2, \dots, A_T) &&= P(A_T) P(A_{T-1} \mid A_T) P(A_{T-2} \mid A_{T-1}, A_T) \cdots \\ &&& \qquad P(A_0 \mid A_1, A_2, \dots, A_T)\\
& \textbf{Law of Total Probability}\\
&P(B)&& = \sum_i P(B \mid C_i) P(C_i)\\\\
&\textbf{Conditional Independence}\\ &\qquad \textbf{ given C}\\\\
&P(A|B,C)&&=P(A|C)\\
&P(A,B|C)&&=P(A|C)P(B|C)\\\\
&\textbf{Marginal Probability}\\
&P(A) &&= \int_B P(A, B) dB\\
&P(A) &&= \sum_B P(A, B)\\
&\mathbf{KL}\\
&KL(P \| Q) &&= \int P(x) \log \frac{P(x)}{Q(x)} dx\\
&{待办: 从信息熵到KL散度, 需要去看扫描的草稿}
\end{alignat}
$$

## Stochastic differential equation (SDE)

