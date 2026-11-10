# SUM-SC-SM-FO-STAT

## 1. Problem Formulation

Consider decentralized optimization with smooth and strongly convex local objectives over a fixed communication matrix, with access to exact local first-order oracles:

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

The nodes communicate through a fixed matrix $W$. For a stacked communication state $Z_t\in\mathbb R^{n\times d}$, one mixing operation is
$$
Z_{t+1}=WZ_t,
\qquad
Z_t=(z_{1,t}^{\top},\ldots,z_{n,t}^{\top})^{\top}.
$$

Let

$$
\mathcal N(i):=\{i\}\cup\{j\in[n]:w_{ij}>0\}.
$$

Node $i$ knows the $i$-th row of $W$ and can access only the weighted sum $\sum_{j\in\mathcal N(i)}w_{ij}z_{j,t}$, not the individual vectors $\{z_{j,t}\}_{j\in\mathcal N(i)}$. One communication round may use the same $W$ to mix at most $c$ vectors in $\mathbb R^d$ per node, where $c$ is an absolute constant independent of all problem parameters. This convention covers methods that simultaneously communicate, for example, a primal variable and a gradient-tracking variable.

Fix $n\ge2$, $L\ge\mu>0$, $R>0$, and $\rho\in[0,1)$. All nodes start from the common initial point $x_0$. Initial auxiliary vector information contains no objective-dependent advice. Oracle calls and communication rounds used to initialize auxiliary states are charged. Define the function class

$$
\begin{aligned}
\mathcal F_{L,\mu,R}:=\Bigl\{(f)_\dagger=(f_1,\ldots,f_n,x_0):\;&
x_0\in\mathbb R^d,\\
&f_i:\mathbb R^d\to\mathbb R\text{ is }\mu\text{-strongly convex and }L\text{-smooth for every }i,\\
&\operatorname*{argmin}_{x\in\mathbb R^d}f(x)\ne\varnothing,\\
&\text{there exists }x^\star\in\operatorname*{argmin}_{x\in\mathbb R^d}f(x)\ \text{with }\|x_0-x^\star\|\le R
\Bigr\}.
\end{aligned}
$$

Write $f^\star=f(x^\star)$ and $\kappa:=L/\mu$. Define the oracle class

$$
\mathcal O_{\mathrm{FO}}((f)_\dagger)
:=\Bigl\{(O)_\dagger=(O_1,\ldots,O_n):
O_i(x)=\nabla f_i(x)\quad\forall i,x\Bigr\}.
$$

Let $J:=(1/n)\mathbf 1\mathbf 1^{\top}$. Define the static communication class

$$
\begin{aligned}
\mathcal W_{n,\rho}:=\Bigl\{(W)_\dagger = W\in\mathbb R^{n\times n}:\;&
W\ge0,\quad
W = W^{\top},\quad
W\mathbf 1=\mathbf 1,\quad \left\|W-J\right\|_2\le\rho
\Bigr\}.
\end{aligned}
$$

The algorithm has access to $n,L,\mu,R,\rho$, the target $\varepsilon$, its node identity, and its local matrix row. Its decisions use only this information, its local oracle history, received mixed messages, and internal randomness. Shared randomness independent of the problem instance is allowed. Define the algorithm class

$$
\mathcal A:=\Bigl\{A:\text{randomized zero-respecting algorithms using, per node and iteration, at most one exact local-gradient query and one communication round}\Bigr\}.
$$

Here, zero-respecting requires that no node can query its oracle at, communicate, or output a vector with a non-zero entry in any coordinate that is zero in its initial point, in all of its previous oracle outputs, and in all of the mixed messages it has received.

For lower-bound constructions, the dimension $d$ may be chosen sufficiently large. Let $\widehat x_{t,i}$ denote the output of node $i$ after $t$ iterations. Define the averaged output

$$
\widehat x_t:=\frac{1}{n}\sum_{i=1}^n\widehat x_{t,i}
$$

and, for objective accuracy $\varepsilon>0$, the convergence metric

$$
T_\varepsilon(A,(f)_\dagger,(O)_\dagger,(W)_\dagger)
:=\inf\left\{t\in\mathbb N_0:
\mathbb E\left[f(\widehat x_t)-f(x^\star)\right]\le\varepsilon
\right\}.
$$

The expectation is over all internal randomness of the algorithm; set $T_\varepsilon=+\infty$ if the set is empty. Either operation may be idle. This arithmetic average is an evaluation convention, not a free global operation available to the algorithm.

The final problem is to determine the order of

$$
\inf_{A\in\mathcal A}
\sup_{(f)_\dagger\in\mathcal F_{L,\mu,R}}
\sup_{(O)_\dagger\in\mathcal O_{\mathrm{FO}}((f)_\dagger)}
\sup_{(W)_\dagger\in\mathcal W_{n,\rho}}
T_\varepsilon(A,(f)_\dagger,(O)_\dagger,(W)_\dagger).
$$

## 2. Task Description

The goal is to give matching upper and lower bounds for this inf-sup problem.

- For the upper bound, construct an algorithm and prove its $\mathcal O(\cdot)$ iteration complexity uniformly over the declared function, oracle, and communication classes. An implementation may use an absolute constant number of oracle calls or communication rounds in one algorithmic iteration: this can be represented by splitting that iteration into a constant number of unit iterations, which preserves the order of the iteration complexity. For clarity, such constant-budget implementations are encouraged instead of literally expanding every substep into separate iterations.
- For the lower bound, for every admissible algorithm construct a hard function, oracle, and communication matrix giving an $\Omega(\cdot)$ iteration lower bound.
- State the dependence on $n,L,\mu,R,\rho$, and $\varepsilon$. Work in small $\varepsilon$ scenario and do not omit a term merely because its $\varepsilon$ dependence is dominated when its dependence on another retained parameter differs.
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

- Function class: retain local convexity and $L$-smoothness but require $\mu$-strong convexity only of the average $f$, $(-)$; impose a common minimizer or a specified initial gradient-disagreement bound, $(+)$.
- Communication class: remove the quantitative contraction bound while retaining connected static symmetric mixing, $(-)$; prescribe a particular topology or require $W\succeq0$, $(+)$.
- Algorithm class: drop zero-respecting, $(+)$; restrict to deterministic linear-span algorithms, $(-)$.
- Convergence metric: additionally require $\max_i\mathbb E[f(\widehat x_{t,i})-f^\star]\le\varepsilon$ or a specified consensus tolerance, $(-)$.

Only alternatives preserving strongly convex smooth optimization, exact local-gradient access, and static decentralized communication are admissible. Changing to merely convex, non-convex, or PL geometry; replacing the exact oracle by component or stochastic-gradient access; or replacing static decentralized communication by a time-varying or centralized model is forbidden. These forbidden changes are not alternatives in the preceding list.

## 4. Related Work

- *FROST—Fast Row-Stochastic Optimization with Uncoordinated Step-Sizes* ([arXiv:1803.09169v5](https://arxiv.org/abs/1803.09169v5)).
