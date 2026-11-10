# SUM-SC-NS-FO-TV-COMP-PROX

## 1. Problem Formulation

Consider the strongly convex composite decentralized optimization problem with time-varying communication matrices and access to exact first-order and proximal oracles:

$$
\min_{x\in\mathbb R^d}\phi(x):=f(x)+h(x),
\qquad
f(x):=\frac{1}{n}\sum_{i=1}^n f_i(x),
$$

where the local objective $f_i(x)$ is private to node $i\in[n]$. The shared regularizer has the exact proximal oracle

$$
O_{\operatorname{prox}_h}(v;\eta)
:=\operatorname{prox}_{\eta h}(v)
:=\operatorname*{argmin}_{y\in\mathbb R^d}
\left\{h(y)+\frac{1}{2\eta}\|y-v\|^2\right\}.
$$

Here $\eta>0$, and one proximal query evaluates this mapping once at one node.

The local first-order oracle is

$$
O_i(x):=\nabla f_i(x).
$$

One gradient-oracle call means one exact full local-gradient evaluation.

The nodes communicate through a sequence of time-varying mixing matrices $(W_q)_{q\ge0}$ fixed before the algorithm runs. Here $q$ counts communication-matrix updates: $W_q$ is the matrix in force after update $q$, with $W_0$ the initial matrix. A common, known schedule updates the matrix after every fixed absolute constant number of elementary iterations, independently of all problem parameters. Between updates the matrix remains fixed; idle communication slots do not delay the schedule. Each communication operation uses the current matrix:

$$
Z^+=W_qZ,
\qquad
Z=(z_1^{\top},\ldots,z_n^{\top})^{\top}.
$$

Let

$$
\mathcal N_q(i):=\{i\}\cup\{j\in[n]:w_{ij,q}>0\}.
$$

Node $i$ knows the $i$-th row of $W_q$ and can access only the weighted sum $\sum_{j\in\mathcal N_q(i)}w_{ij,q}z_j$, not the individual vectors $\{z_j\}_{j\in\mathcal N_q(i)}$. One communication round may use the same $W_q$ to mix at most $c$ vectors in $\mathbb R^d$ per node, where $c$ is an absolute constant independent of all problem parameters. This convention covers methods that simultaneously communicate, for example, a primal variable and a gradient-tracking variable.

Fix integers $n\ge2$, $p\ge1$, and parameters $L\ge\mu>0$, $R>0$, $\rho\in[0,1)$. All nodes start from the common initial point $x_0$. Write $\kappa:=L/\mu$. Define the function class

$$
\begin{aligned}
\mathcal F_{L,\mu,R}:=\Bigl\{(f)_\dagger=(f_1,\ldots,f_n,h,x_0):\;&
x_0\in\mathbb R^d,\\
&f_i:\mathbb R^d\to\mathbb R\text{ is convex and }L\text{-smooth for every }i,\\
&h:\mathbb R^d\to\mathbb R\cup\{+\infty\}\text{ is proper, closed, convex, and exactly proximable},\quad x_0\in\operatorname{dom}h,\\
&\phi\text{ is }\mu\text{-strongly convex},\quad
x^\star=\arg\min_x\phi(x),\quad \|x_0-x^\star\|\le R
\Bigr\}.
\end{aligned}
$$

Define the oracle class

$$
\mathcal O_{\mathrm{FO},\mathrm{prox}}((f)_\dagger)
:=\Bigl\{(O)_\dagger=(O_1,\ldots,O_n,O_{\operatorname{prox}_h}):
O_i(x)=\nabla f_i(x),\quad
O_{\operatorname{prox}_h}(v;\eta)=\operatorname{prox}_{\eta h}(v)
\Bigr\}.
$$

Let $J:=(1/n)\mathbf 1\mathbf 1^{\top}$. Define the time-varying communication class by the quantitative product-contraction condition

$$
\begin{aligned}
\mathcal W_{p,\rho}:=\Bigl\{(W)_\dagger=(W_q)_{q\ge0}:\;&
W_q\ge0,\quad
\mathbf 1^{\top}W_q=\mathbf 1^{\top},\quad
W_q\mathbf 1=\mathbf 1,\\
&\left\|W_{q+p-1}\cdots W_q-J\right\|_2\le\rho
\quad\text{for every }q\ge0
\Bigr\}.
\end{aligned}
$$

Here $p$ counts consecutive matrix-update intervals, with each interval's matrix appearing once in the product; $t$ counts elementary iterations, including waiting.

The parameters $n,L,\mu,R,p,\rho,\varepsilon$ are available to the algorithm, as is the current local row of $W_q$. Shared randomness independent of the problem instance is allowed; no objective-dependent auxiliary advice is provided. Define the algorithm class

$$
\mathcal A:=\Bigl\{A:\text{randomized zero-respecting algorithms using, per node and iteration, at most one exact local-gradient query, one exact proximal query, and one communication round}\Bigr\}.
$$

Here, zero-respecting requires that no node can query its oracle at, communicate, or output a vector with a non-zero entry in any coordinate that is zero in its initial point, in all of its previous oracle outputs, and in all of the mixed messages it has received.

For lower-bound constructions, the dimension $d$ may be chosen sufficiently large; $A$ denotes an algorithm family accepting $d$. Let $\widehat x_{t,i}$ be node $i$'s local output after $t$ iterations. Define the averaged output

$$
\widehat x_t:=\frac{1}{n}\sum_{i=1}^n\widehat x_{t,i}
$$

and the convergence metric

$$
T_\varepsilon(A,(f)_\dagger,(O)_\dagger,(W)_\dagger)
:=\inf\left\{t\in\mathbb Z_{\ge0}:
\mathbb E\|\widehat x_t-x^\star\|^2\le\varepsilon
\right\}.
$$

The average is formed by the evaluator and is not given to the algorithm for free. The expectation is over all internal randomness of the algorithm. The iteration count includes every primitive mixing round inside a multi-round communication block.

The final problem is to determine the order of

$$
\inf_{A\in\mathcal A}
\sup_{d\in\mathbb N}
\sup_{(f)_\dagger\in\mathcal F_{L,\mu,R}}
\sup_{(O)_\dagger\in\mathcal O_{\mathrm{FO},\mathrm{prox}}((f)_\dagger)}
\sup_{(W)_\dagger\in\mathcal W_{p,\rho}}
T_\varepsilon(A,(f)_\dagger,(O)_\dagger,(W)_\dagger).
$$

## 2. Task Description

The goal is to give matching upper and lower bounds for this inf-sup problem.

- For the upper bound, construct an algorithm and prove its $\mathcal O(\cdot)$ iteration complexity uniformly over the declared function, oracle, and communication classes. An implementation may use an absolute constant number of oracle calls, proximal calls, or communication rounds in one algorithmic iteration: this can be represented by splitting that iteration into a constant number of unit iterations while respecting the scheduled matrix updates, which preserves the order of the iteration complexity. All waiting and alignment iterations are charged. For clarity, such constant-budget implementations are encouraged instead of literally expanding every substep into separate iterations.
- For the lower bound, for every admissible algorithm construct a hard function, oracle, and communication sequence giving an $\Omega(\cdot)$ iteration lower bound.
- State the dependence on $n,L,\mu,R,p,\rho$, and $\varepsilon$. Work in small $\varepsilon$ scenario and do not omit a term merely because its $\varepsilon$ dependence is dominated when its dependence on another retained parameter differs.
- Combine matching bounds to obtain the optimal $\Theta(\cdot)$ order. Bounds matching up to logarithmic factors may be written with $\widetilde{\mathcal O}$, $\widetilde{\Omega}$, and $\widetilde{\Theta}$.

Every reported bound must retain the dependence on all declared parameters, including terms that are lower order in $\varepsilon$ but have different dependence on other parameters. For example, a rate of the form

$$
\Theta\left(
\frac{L\Delta\sigma^2}{n\varepsilon^2}
+\frac{L\Delta}{\varepsilon}
\right)
$$

must keep both terms: the second term is lower order in $\varepsilon$ than the first, but it does not contain the first term's $1/n$ linear-speedup factor and therefore cannot be absorbed into it.

## 3. Relaxation Rules

Only the standard problem receives full credit. Results under explicit alternatives may receive partial credit. For function, oracle, and communication classes, a class relaxation is marked $(-)$ and a class restriction is marked $(+)$. For the algorithm class, enlarging the admissible class is marked $(+)$ and adding an algorithmic restriction is marked $(-)$. For the convergence metric, adding a success requirement is marked $(-)$ and weakening the success criterion is marked $(+)$.

Usually, a $(+)$ alternative weakens the upper-bound conclusion and strengthens the lower-bound conclusion, whereas a $(-)$ alternative strengthens the upper-bound conclusion and weakens the lower-bound conclusion.

Common alternatives include:

- Function class: require smoothness only of $f$, retaining convex differentiable local functions, $(-)$; require every $f_i$ to be strongly convex or bound gradient heterogeneity at $x^\star$, $(+)$.
- Oracle class: allow controlled inexact proximal evaluations, $(-)$.
- Communication class: remove the $(p,\rho)$ product-contraction bound, $(-)$; require symmetric matrices or a prescribed topology, $(+)$.
- Algorithm class: drop zero-respecting, $(+)$; require deterministic algorithms, $(-)$.
- Convergence metric: additionally require $n^{-1}\sum_i\mathbb E\|\widehat x_{t,i}-x^\star\|^2\le\varepsilon$, $(-)$.

Only alternatives preserving the target geometry, oracle type, and time-varying decentralized communication model are admissible. Forbidden changes are replacing strong convexity by convex, PL, or non-convex geometry; replacing full local-gradient access by component or stochastic-gradient access; or replacing time-varying decentralized communication by a static or centralized model. These forbidden changes are not alternatives in the preceding list.

## 4. Related Work

- *Decentralized Composite Optimization in Stochastic Networks: A Dual Averaging Approach with Linear Convergence* ([arXiv:2106.14075](https://arxiv.org/abs/2106.14075)).
