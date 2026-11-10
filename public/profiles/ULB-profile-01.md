# SUM-SC-SM-FO-TV

## 1. Problem Formulation

Consider decentralized optimization with smooth and strongly convex local objectives over time-varying communication matrices, with access to exact local first-order oracles:

$$
\min_{x\in\mathbb R^d} f(x):=\frac1n\sum_{i=1}^n f_i(x).
$$

The differentiable local objective $f_i:\mathbb R^d\to\mathbb R$ is private to node $i\in[n]$. Each local objective is $L$-smooth,

$$
\|\nabla f_i(x)-\nabla f_i(y)\|
\le L\|x-y\|,
\qquad x,y\in\mathbb R^d,
$$

and each local objective is $\mu$-strongly convex:

$$
f_i(y)\ge f_i(x)+\langle\nabla f_i(x),y-x\rangle
       +\frac\mu2\|y-x\|^2,
\qquad x,y\in\mathbb R^d.
$$

The aggregate has the unique minimizer

$$
x^\star:=\arg\min_x f(x).
$$

The local primal first-order oracle is

$$
O_i(x):=\nabla f_i(x).
$$

One declared local oracle call means one exact full local-gradient evaluation at one query point.
The nodes communicate through a sequence of time-varying mixing matrices $(W_q)_{q\ge0}$, where $q$ counts communication-matrix updates and $t$ counts algorithm iterations. The matrix $W_q$ is in force after update $q$, with $W_0$ the initial matrix. Updates occur every fixed absolute constant number of iterations, independently of whether communication occurs:

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

Fix $L\ge\mu>0$, $R>0$, $n,p\in\mathbb N$, and $\rho\in[0,1)$. All nodes start from the common initial point $x_0$. Define the function class

$$
\begin{aligned}
\mathcal F_{L,\mu,R}:=\Bigl\{(f)_\dagger=(f_1,\ldots,f_n,x_0):\;&
x_0\in\mathbb R^d,\\
&f_i:\mathbb R^d\to\mathbb R\text{ is }\mu\text{-strongly convex and }L\text{-smooth for every }i,\\
&\operatorname*{argmin}_{x\in\mathbb R^d}f(x)\ne\varnothing,\\
&\text{there exists }x^\star\in\operatorname*{argmin}_{x\in\mathbb R^d}f(x)
\text{ with }\|x_0-x^\star\|\le R
\Bigr\}.
\end{aligned}
$$



Define the oracle class

$$
\mathcal O_{\mathrm{FO}}((f)_\dagger)
:=\Bigl\{(O)_\dagger=(O_1,\ldots,O_n):
O_i(x)=\nabla f_i(x)
\Bigr\}.
$$

Let $J:=(1/n)\mathbf 1\mathbf 1^{\top}$. Here $p$ counts consecutive matrix-update intervals. Define the time-varying communication class by the quantitative product-contraction condition

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

The parameters $n,L,\mu,R,p,\rho$ are available to the algorithm, as is the current local row of $W_q$.

 Define the algorithm class

$$
\mathcal A:=\Bigl\{A:\text{deterministic zero-respecting algorithms using, per node and iteration, at most one exact local-gradient query and one communication round}\Bigr\}.
$$

Here, zero-respecting requires that no node can query its oracle at, communicate, or output a vector with a non-zero entry in any coordinate that is zero in its initial point, in all of its previous oracle outputs, and in all of the mixed messages it has received.

For lower-bound constructions, the dimension $d$ may be chosen sufficiently large. Define the averaged output

$$
\widehat x_t:=\frac{1}{n}\sum_{i=1}^n\widehat x_{t,i}
$$

and the convergence metric

$$
T_\varepsilon(A,(f)_\dagger,(O)_\dagger,(W)_\dagger)
:=\inf\left\{t\ge0:
f(\widehat x_t)-f(x^\star)\le\varepsilon
\right\},
$$ where $0<\varepsilon\le LR^2$ .

The final problem is to determine the order of

$$
\inf_{A\in\mathcal A}
\sup_{(f)_\dagger\in\mathcal F_{L,\mu,R}}
\sup_{(O)_\dagger\in\mathcal O_{\mathrm{FO}}((f)_\dagger)}
\sup_{(W)_\dagger\in\mathcal W_{p,\rho}}
T_\varepsilon(A,(f)_\dagger,(O)_\dagger,(W)_\dagger).
$$


## 2. Task Description

The goal is to give matching upper and lower bounds for reaching $f(\widehat x_t)-f(x^\star)\le\varepsilon$.

- For the upper bound, construct an algorithm and prove its $\mathcal O(\cdot)$ iteration complexity uniformly over the declared function, oracle, and communication classes.
- For the lower bound, for every admissible algorithm construct a hard function, oracle, and communication sequence giving an $\Omega(\cdot)$ iteration lower bound.
- State the dependence on $n,L,\mu,R,p,\rho$, and $\varepsilon$. Work in the explicit accuracy range $0<\varepsilon\le LR^2$ and do not omit a term merely because its $\varepsilon$ dependence is dominated when its dependence on another retained parameter differs.
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

Only the standard problem receives full credit. Results under explicit alternatives may receive partial credit. For function and communication classes, a class relaxation is marked $(-)$ and a class restriction is marked $(+)$. For the algorithm class, enlarging the admissible class is marked $(+)$ and adding an algorithmic restriction is marked $(-)$. For the convergence metric, adding a success requirement is marked $(-)$.

Usually, a $(+)$ alternative weakens the upper-bound conclusion and strengthens the lower-bound conclusion, whereas a $(-)$ alternative strengthens the upper-bound conclusion and weakens the lower-bound conclusion.

Common alternatives include:

- Function class: require only the aggregate $f$ to be $L$-smooth and $\mu$-strongly convex, without imposing these properties on every local $f_i$, $(-)$; impose a common minimizer or an additional heterogeneity bound $n^{-1}\sum_i\|\nabla f_i(x)-\nabla f(x)\|^2\le H^2$ for any $x$, $(+)$.
- Communication class: weaken or remove the $(p,\rho)$ product-contraction bound, $(-)$; require $p$-periodicity, symmetry, positive semidefiniteness$(+)$.
- Algorithm class: drop the zero-respecting requirement, $(+)$; allow randomized algorithms with shared randomness independent of the problem instance and evaluate convergence in expectation, $(+)$.
- Convergence metric: additionally require every local output to satisfy the objective-gap tolerance or require a consensus-error bound, $(-)$.

Only alternatives preserving smooth strongly convex optimization, exact primal first-order access, and decentralized communication are admissible. Changing to non-convex, merely convex, or PL geometry; replacing decentralized communication by centralized communication; or treating a full local gradient as a different oracle unit is forbidden.

## 4. Related Work

- *Lower Bounds and Optimal Algorithms for Smooth and Strongly Convex Decentralized Optimization Over Time-Varying Networks* ([arXiv:2106.04469](https://arxiv.org/abs/2106.04469)).
