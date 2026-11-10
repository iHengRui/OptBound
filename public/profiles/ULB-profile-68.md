# SUM-NC-NS-FO-TV

## 1. Problem Formulation

Consider the non-convex nonsmooth decentralized optimization problem with time-varying communication matrices and access to exact local first-order oracles:

$$
\min_{x\in\mathbb R^d}f(x):=\frac{1}{n}\sum_{i=1}^n f_i(x),
$$

where the local objective $f_i$ is private to node $i\in[n]:=\{1,\ldots,n\}$.

For a locally Lipschitz function $g$, write $\partial_Cg(x)$ for its Clarke subdifferential: the closed convex hull of limits of gradients at differentiability points approaching $x$. For a declared radius $\delta>0$, define

$$
\partial^G_\delta g(x):=\operatorname{cl\,conv}\!\bigcup_{\|y-x\|_2\le\delta}\partial_Cg(y).
$$

The classical success measure is the Goldstein residual $\operatorname{dist}(0,\partial^G_\delta f(x))$.

The local oracle is

$$
O_i(x):=\nabla f_i(x).
$$

One oracle call returns one exact full local gradient. Local functions are differentiable and $M$-Lipschitz; no Lipschitz bound on their gradients is imposed.

Arbitrary local computation on previously acquired information is permitted.

The nodes communicate through a sequence of time-varying mixing matrices $(W_q)_{q\ge0}$ fixed before the algorithm runs. Here $q$ counts communication-matrix updates: $W_q$ is the matrix in force after update $q$, with $W_0$ the initial matrix. A common, known schedule updates the matrix after every fixed absolute constant number of elementary iterations, independently of all problem parameters. Between updates the matrix remains fixed; idle communication slots do not delay the schedule. Each communication operation uses the current matrix:

$$
Z^+=W_qZ,
\qquad Z=(z_1^{\top},\ldots,z_n^{\top})^{\top}.
$$

Let

$$
\mathcal N_q(i):=\{i\}\cup\{j\in[n]:w_{ij,q}>0\}.
$$

Node $i$ knows the $i$-th row of $W_q$ and can access only the weighted sum $\sum_{j\in\mathcal N_q(i)}w_{ij,q}z_j$, not the individual vectors $\{z_j\}_{j\in\mathcal N_q(i)}$. One communication round may use the same $W_q$ to mix at most $c$ vectors in $\mathbb R^d$ per node, where $c$ is an absolute constant independent of all problem parameters. This convention covers methods that simultaneously communicate, for example, a primal variable and a gradient-tracking variable.

Fix $M,\Delta,\delta>0$, $n\in\mathbb N_{\ge2}$, $p\in\mathbb N_{\ge1}$, and $\rho\in[0,1)$.

All nodes start from the common initial point $x_0$. Initial auxiliary vector information contains no objective-dependent advice. Oracle calls and communication rounds used to initialize auxiliary states are charged.

Define the function class

$$
\begin{aligned}
\mathcal F_{M,\Delta}:=\Bigl\{(f)_\dagger=(d,f_1,\ldots,f_n,x_0):\;&
x_0\in\mathbb R^d,\\
&d\in\mathbb N,\quad f_i:\mathbb R^d\to\mathbb R\text{ is finite},\\
&f_i\text{ is differentiable everywhere},\\
&|f_i(x)-f_i(y)|\le M\|x-y\|_2
\quad(\forall i,\ \forall x,y\in\mathbb R^d),\\
&f^\star:=\inf_{x\in\mathbb R^d}f(x)>-\infty,\quad f(x_0)-f^\star\le\Delta
\Bigr\}.
\end{aligned}
$$

Define the oracle class

$$
\begin{aligned}
\mathcal O_{\mathrm{FO}}((f)_\dagger)
:=\Bigl\{(O)_\dagger=(O_1,\ldots,O_n):\;&
O_i:\mathbb R^d\to\mathbb R^d\text{ is Borel measurable for every }i,\\
&O_i(x)=\nabla f_i(x)\quad(\forall i\in[n],\ \forall x\in\mathbb R^d)
\Bigr\}.
\end{aligned}
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

The algorithm has access to $n,M,\Delta,\delta,p,\rho$, the target $\varepsilon$, its node identity, and its current local matrix row when communication occurs. Its decisions use only this information, its local oracle history, received mixed messages, and internal randomness. Shared randomness independent of the problem instance is allowed. Define the algorithm class

$$
\mathcal A:=\Bigl\{A:\text{randomized zero-respecting algorithms using, per node and iteration, at most one exact local-gradient query and one communication round}\Bigr\}.
$$

Here, zero-respecting requires that no node can query its oracle at, communicate, or output a vector with a non-zero entry in any coordinate that is zero in its initial point, in all of its previous oracle outputs, and in all of the mixed messages it has received.

For lower-bound constructions, the dimension $d$ may be chosen sufficiently large, while $n$ remains a fixed parameter. Let $\widehat x_{t,i}$ denote the output of node $i$ after $t$ iterations. Define the averaged output

$$
\widehat x_t:=\frac{1}{n}\sum_{i=1}^n\widehat x_{t,i}
$$

and, for Goldstein accuracy $\varepsilon>0$, the convergence metric

$$
T_\varepsilon(A,(f)_\dagger,(O)_\dagger,(W)_\dagger)
:=\inf\left\{t\in\mathbb N_0:
\mathbb E\left[\operatorname{dist}(0,\partial^G_\delta f(\widehat x_t))\right]\le\varepsilon
\right\}.
$$

The expectation is over all internal randomness of the algorithm; set $T_\varepsilon=+\infty$ if the set is empty.

This arithmetic average is an evaluation convention, not a free global operation available to the algorithm.

The final problem is to determine the order of

$$
\inf_{A\in\mathcal A}
\sup_{(f)_\dagger\in\mathcal F_{M,\Delta}}
\sup_{(O)_\dagger\in\mathcal O_{\mathrm{FO}}((f)_\dagger)}
\sup_{(W)_\dagger\in\mathcal W_{p,\rho}}
T_\varepsilon(A,(f)_\dagger,(O)_\dagger,(W)_\dagger).
$$

Here $A$ denotes an algorithm family accepting dimension $d$.

## 2. Task Description

The goal is to give matching upper and lower bounds for this inf-sup problem.

- For the upper bound, construct an algorithm and prove its $\mathcal O(\cdot)$ iteration complexity uniformly over the declared function, oracle, and communication classes. An implementation may use an absolute constant number of oracle calls or communication rounds in one algorithmic iteration: this can be represented by splitting that iteration into a constant number of unit iterations while respecting the scheduled matrix updates, which preserves the order of the iteration complexity. All waiting and alignment iterations are charged. For clarity, such constant-budget implementations are encouraged instead of literally expanding every substep into separate iterations.
- For the lower bound, for every admissible algorithm construct a hard function, oracle, and communication sequence giving an $\Omega(\cdot)$ iteration lower bound.
- State the dependence on $n,M,\Delta,\delta,p,\rho$, and $\varepsilon$. Work in small $\varepsilon$ scenario and do not omit a term merely because its $\varepsilon$ dependence is dominated when its dependence on another retained parameter differs.
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

- Function class: retain local $M$-Lipschitz continuity and impose a known weak-convexity modulus on every local objective, $(+)$.
- Communication class: require $\|W_q-J\|_2\le\rho$ at every matrix-update index, or require every $W_q$ to be symmetric, $(+)$.
- Algorithm class: drop the zero-respecting requirement, $(+)$; require all query and output vectors to lie in the span of locally acquired vectors, $(-)$.
- Convergence metric: additionally require a per-node Goldstein-residual or mean squared consensus-error condition, $(-)$; enlarge the declared Goldstein radius, $(+)$.

Only alternatives preserving non-convex nonsmooth optimization, exact local-gradient access, and time-varying decentralized communication are admissible. Changing to convex, strongly convex, or PL geometry; granting a smoothness constant or a proximal oracle; or replacing the declared decentralized communication model is forbidden. These changes are excluded from the preceding list of alternatives.

## 4. Related Work

No setting-specific reference is listed in the source table.
