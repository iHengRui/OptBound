# SUM-NC-NS-SFO-STAT-COMP-PROX

## 1. Problem Formulation

Consider the finite-sum non-convex composite decentralized optimization problem with static communication matrix and access to incremental first-order and proximal oracles:

$$
\min_{x\in\mathbb R^d}\phi(x):=f(x)+h(x),
\qquad
f(x):=\frac{1}{n}\sum_{i=1}^n f_i(x),
$$

where the local finite-sum objective

$$
f_i(x):=\frac{1}{m}\sum_{\xi\in D_i}F_i(x;\xi),
\qquad |D_i|=m,
$$

is private to node $i\in[n]$. Components need not be convex. The shared regularizer has the exact proximal oracle

$$
O_{\operatorname{prox}_h}(v;\eta)
:=\operatorname{prox}_{\eta h}(v)
:=\operatorname*{argmin}_{y\in\mathbb R^d}
\left\{h(y)+\frac{1}{2\eta}\|y-v\|^2\right\}.
$$

Here $\eta>0$, and one proximal query evaluates this mapping once at one node.

The local incremental first-order oracle is

$$
O_i(x,\xi):=\nabla F_i(x;\xi),
\qquad \xi\in D_i.
$$

One gradient-oracle call means one exact component-gradient query, not evaluation of the full local gradient $\nabla f_i(x)$. The algorithm may choose the queried component and may be randomized. Computing $\nabla f_i(x)$ costs $m$ component queries.

The nodes communicate through a fixed symmetric matrix $W$. For a stacked communication state $Z_t\in\mathbb R^{n\times d}$, one mixing operation is
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

Fix integers $n\ge2$, $m\ge1$, and parameters $L>0$, $\Delta>0$, $\rho\in[0,1)$. All nodes start from the common initial point $x_0$. Define the function class

$$
\begin{aligned}
\mathcal F_{L,\Delta,m}:=\Bigl\{(f)_\dagger=(f_1,\ldots,f_n,h,x_0;\{F_i(\cdot;\xi)\}_{i,\xi}):\;&
x_0\in\mathbb R^d,\\
&F_i(\cdot;\xi)\text{ is }L\text{-smooth for every }i,\xi,\\
&h:\mathbb R^d\to\mathbb R\cup\{+\infty\}\text{ is proper, closed, convex, and exactly proximable},\\
&\phi^\star:=\inf_{x\in\mathbb R^d}\phi(x)>-\infty,\\
&\phi(x_0)-\phi^\star\le\Delta
\Bigr\}.
\end{aligned}
$$

Define the oracle class

$$
\mathcal O_{\mathrm{IFO},\mathrm{prox}}((f)_\dagger)
:=\Bigl\{(O)_\dagger=(O_1,\ldots,O_n,O_{\operatorname{prox}_h}):
O_i(x,\xi)=\nabla F_i(x;\xi),\ \xi\in D_i,\quad
O_{\operatorname{prox}_h}(v;\eta)=\operatorname{prox}_{\eta h}(v)
\Bigr\}.
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

The parameters $n,m,L,\Delta,\rho,\varepsilon$ are available to the algorithm, as is each node's local row of $W$. Shared randomness independent of the problem instance is allowed; no objective-dependent auxiliary advice is provided. Define the algorithm class

$$
\mathcal A:=\Bigl\{A:\text{randomized zero-respecting algorithms using, per node and iteration, at most one component-gradient query, one exact proximal query, and one communication round}\Bigr\}.
$$

Here, zero-respecting requires that no node can query its oracle at, communicate, or output a vector with a non-zero entry in any coordinate that is zero in its initial point, in all of its previous oracle outputs, and in all of the mixed messages it has received.

For lower-bound constructions, the dimension $d$ may be chosen sufficiently large. Let $\widehat x_{t,i}$ be node $i$'s local output after $t$ iterations. Define the averaged output

$$
\widehat x_t:=\frac{1}{n}\sum_{i=1}^n\widehat x_{t,i}
$$

Define the proximal-gradient mapping

$$
\mathcal G_{1/L}(x):=L\left(x-\operatorname{prox}_{h/L}\left(x-\frac1L\nabla f(x)\right)\right)
$$

and the convergence metric

$$
T_\varepsilon(A,(f)_\dagger,(O)_\dagger,(W)_\dagger)
:=\inf\left\{T\in\mathbb N_{\ge1}:
\frac1T\sum_{t=1}^T\mathbb E\left[\|\mathcal G_{1/L}(\widehat x_t)\|^2\right]\le\varepsilon
\right\}.
$$

The average is formed by the evaluator and is not given to the algorithm for free. The expectation is over all internal randomness of the algorithm, including any randomized choice of component indices. The iteration count includes full-gradient initialization, batches, proximal queries, and every primitive mixing round.

The final problem is to determine the order of

$$
\inf_{A\in\mathcal A}
\sup_{(f)_\dagger\in\mathcal F_{L,\Delta,m}}
\sup_{(O)_\dagger\in\mathcal O_{\mathrm{IFO},\mathrm{prox}}((f)_\dagger)}
\sup_{(W)_\dagger\in\mathcal W_{n,\rho}}
T_\varepsilon(A,(f)_\dagger,(O)_\dagger,(W)_\dagger).
$$

## 2. Task Description

The goal is to give matching upper and lower bounds for this inf-sup problem.

- For the upper bound, construct an algorithm and prove its $\mathcal O(\cdot)$ iteration complexity uniformly over the declared function, oracle, and communication classes. An implementation may use an absolute constant number of oracle calls, proximal calls, or communication rounds in one algorithmic iteration: this can be represented by splitting that iteration into a constant number of unit iterations, which preserves the order of the iteration complexity. For clarity, such constant-budget implementations are encouraged instead of literally expanding every substep into separate iterations.
- For the lower bound, for every admissible algorithm construct a hard function, oracle, and communication matrix giving an $\Omega(\cdot)$ iteration lower bound.
- State the dependence on $n,m,L,\Delta,\rho$, and $\varepsilon$. Work in small $\varepsilon$ scenario and do not omit a term merely because its $\varepsilon$ dependence is dominated when its dependence on another retained parameter differs.
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

- Function class: require smoothness only of the local averages, $(-)$; impose identical initial local gradients, $(+)$.
- Oracle class: replace selectable indices by independent uniform component sampling, $(+)$; allow controlled inexact proximal evaluations, $(-)$.
- Communication class: remove the quantitative contraction bound, $(-)$; require $W\succeq0$ or a prescribed topology, $(+)$.
- Algorithm class: drop zero-respecting, $(+)$; restrict component-index selection to an oblivious schedule, $(-)$.
- Convergence metric: additionally require a per-node proximal-gradient or consensus condition, $(-)$.

Only alternatives preserving the target geometry, oracle type, and static decentralized communication model are admissible. Forbidden changes are replacing non-convex geometry by convex, strongly convex, or PL geometry; replacing the finite sum by an expectation objective; treating a full local gradient as one component-oracle call; or replacing static decentralized communication by a time-varying or centralized model. These forbidden changes are not alternatives in the preceding list.

## 4. Related Work

- *A Stochastic Proximal Gradient Framework for Decentralized Non-Convex Composite Optimization: Topology-Independent Sample Complexity and Communication Efficiency* ([arXiv:2110.01594](https://arxiv.org/abs/2110.01594)).
