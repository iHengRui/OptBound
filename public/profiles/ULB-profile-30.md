# SUM-SC-NS-SUBG-STAT

## 1. Problem Formulation

Consider the strongly convex nonsmooth decentralized optimization problem over a fixed undirected communication network with access to exact local subgradient oracles:

$$
\min_{x\in Q_R}f(x):=\frac{1}{n}\sum_{i=1}^n f_i(x),
$$

where the local objective $f_i$ is private to node $i\in[n]:=\{1,\ldots,n\}$.

Let $Q_R:=\{x\in\mathbb R^d:\|x\|_2\le R\}$ be the common known feasible ball. All oracle query points and primal outputs lie in $Q_R$. Euclidean projection onto this known ball is local computation.

The local oracle is

$$
O_i(x)\in\partial f_i(x).
$$

One oracle call returns one exact local subgradient. Here $\partial f_i$ denotes the convex subdifferential of $f_i$.

The nodes communicate through a fixed mixing matrix $W$:

$$
Z_{t+1}=WZ_t,
\qquad
Z_t=(z_{1,t}^\top,\ldots,z_{n,t}^\top)^\top.
$$

Let

$$
\mathcal N(i):=\{i\}\cup\{j\in[n]:w_{ij}>0\}.
$$

Node $i$ knows the $i$-th row of $W$ and receives the weighted sum $\sum_{j\in\mathcal N(i)}w_{ij}z_{j,t}$ as its communication oracle output. One communication round may use the same $W$ to mix at most $c$ vectors in $\mathbb R^d$ per node, where $c$ is an absolute constant independent of all problem parameters.

Fix $R,\mu>0$, $M\ge\mu R$, $n\in\mathbb N_{\ge2}$, and $\rho\in[0,1)$. All nodes start from the common initial point $x_0$. Define the function class

$$
\begin{aligned}
\mathcal F_{M,\mu,R}:=\Bigl\{(f)_\dagger=(f_1,\ldots,f_n,x_0):\;&
x_0\in Q_R,\\
&f_i:\mathbb R^d\to\mathbb R\text{ is finite and }\mu\text{-strongly convex},\\
&\|v\|_2\le M\quad(\forall i,\ \forall x\in Q_R,\ \forall v\in\partial f_i(x)),\\
&\exists x^\star\in Q_R\cap\operatorname*{argmin}_{x\in\mathbb R^d}f(x),
\quad \|x_0-x^\star\|_2\le R
\Bigr\}.
\end{aligned}
$$

Define the oracle class

$$
\begin{aligned}
\mathcal O_{\mathrm{SUBG}}((f)_\dagger)
:=\Bigl\{(O)_\dagger=(O_1,\ldots,O_n):\;&
O_i:Q_R\to\mathbb R^d\text{ is Borel measurable for every }i,\\
&O_i(x)\in\partial f_i(x)\quad(\forall i\in[n],\ \forall x\in Q_R)
\Bigr\}.
\end{aligned}
$$

Let $J:=(1/n)\mathbf1\mathbf1^\top$. Define the static communication class by the quantitative contraction condition

$$
\begin{aligned}
\mathcal W_{n,\rho}:=\Bigl\{(W)_\dagger=W\in\mathbb R^{n\times n}:\;&
W=W^\top,\quad w_{ij}\ge0\ \text{for all }i,j,\\
&\mathbf1^\top W=\mathbf1^\top,\quad W\mathbf1=\mathbf1,\\
&\|W-J\|_2\le\rho
\Bigr\}.
\end{aligned}
$$

Here $\|\cdot\|_2$ denotes the matrix operator norm in the communication condition.

The algorithm has access to $n,M,\mu,R,\rho,\varepsilon$, its node identity, and its local row of $W$. Its decisions use only this information, its local oracle history, received mixed messages, and internal randomness. Shared randomness independent of the problem instance is allowed. Initial auxiliary vector information contains no objective-dependent advice. Oracle calls and communication rounds used to initialize auxiliary states are charged. Define the algorithm class

$$
\mathcal A:=\Bigl\{A:\text{randomized zero-respecting algorithms using, per node and iteration, at most one exact local-subgradient query and one communication round}\Bigr\}.
$$

Here, zero-respecting requires that no node can query its oracle at, communicate, or output a vector with a non-zero entry in any coordinate that is zero in its initial point, in all of its previous oracle outputs, and in all of the mixed messages it has received.

For lower-bound constructions, the dimension $d$ may be chosen sufficiently large, while $n$ remains a fixed parameter. Let $\widehat x_{t,i}$ denote the output of node $i$ after $t$ iterations. Define the averaged output

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

The expectation is over all internal randomness of the algorithm; set $T_\varepsilon=+\infty$ if the set is empty.

The final problem is to determine the order of

$$
\inf_{A\in\mathcal A}
\sup_{d\ge1}
\sup_{(f)_\dagger\in\mathcal F_{M,\mu,R}}
\sup_{(O)_\dagger\in\mathcal O_{\mathrm{SUBG}}((f)_\dagger)}
\sup_{(W)_\dagger\in\mathcal W_{n,\rho}}
T_\varepsilon(A,(f)_\dagger,(O)_\dagger,(W)_\dagger).
$$

Here $A$ denotes an algorithm family accepting dimension $d$.

## 2. Task Description

The goal is to give matching upper and lower bounds for this inf-sup problem.

- For the upper bound, construct an algorithm and prove its $\mathcal O(\cdot)$ iteration complexity uniformly over the declared function, oracle, and communication classes. An implementation may use an absolute constant number of oracle calls or communication rounds in one algorithmic iteration: this can be represented by splitting that iteration into a constant number of unit iterations, which preserves the order of the iteration complexity. For clarity, such constant-budget implementations are encouraged instead of literally expanding every substep into separate iterations.
- For the lower bound, for every admissible algorithm construct a hard function, oracle, and communication matrix giving an $\Omega(\cdot)$ iteration lower bound. State the required dimension and other applicability conditions.
- State the dependence on $n,M,\mu,R,\rho,\varepsilon$. Primarily consider sufficiently small $\varepsilon$, retaining every term with distinct dependence on another declared parameter.
- Combine matching bounds to obtain the optimal $\Theta(\cdot)$ order. Bounds matching up to logarithmic factors may be written with $\widetilde{\mathcal O}$, $\widetilde{\Omega}$, and $\widetilde{\Theta}$.

Every reported bound must retain the dependence on all declared parameters, including terms that are lower order in $\varepsilon$ but have different dependence on other parameters. For example, a rate of the form

$$
\Theta\left(
\frac{L\Delta\sigma^2}{n\varepsilon^2}
+\frac{L\Delta}{\varepsilon}
\right)
$$

must keep both terms: the first carries a $1/n$ linear-speedup factor, while the second has a different dependence on $n$.

## 3. Relaxation Rules

Only the standard problem receives full credit. Results under explicit alternatives may receive partial credit. For function, oracle, and communication classes, a class relaxation is marked $(-)$ and a class restriction is marked $(+)$. For the algorithm class, enlarging the admissible class is marked $(+)$ and adding an algorithmic restriction is marked $(-)$. For the convergence metric, adding a success requirement is marked $(-)$ and weakening the success criterion is marked $(+)$.

Usually, a $(+)$ alternative weakens the upper-bound conclusion and strengthens the lower-bound conclusion, whereas a $(-)$ alternative strengthens the upper-bound conclusion and weakens the lower-bound conclusion.

Common alternatives include:

- Function class: require strong convexity only of the aggregate $f$ on $Q_R$, retaining local convexity and the declared subgradient bound, $(-)$; impose a common minimizer for all local functions, $(+)$.
- Oracle class: restrict all valid selections to the minimum-norm subgradient at every query point, $(+)$; the oracle must still return an exact local subgradient.
- Algorithm class: drop the zero-respecting requirement, $(+)$.
- Convergence metric: additionally require the mean squared consensus error $\frac{1}{n}\sum_{i=1}^n\mathbb E[\|\widehat x_{t,i}-\widehat x_t\|_2^2]\le\delta^2$ for a declared tolerance $\delta>0$, at the same iteration as the objective requirement, $(-)$.

Only alternatives preserving strongly convex nonsmooth optimization, exact local-subgradient access, and decentralized communication over a fixed undirected network are admissible. Changing to merely convex, non-convex, or PL geometry; replacing exact local-subgradient access by stochastic or inexact oracle access; or replacing the fixed undirected decentralized network by centralized, directed, or time-varying communication is forbidden. These changes are excluded from the preceding list of alternatives.

## 4. Related Work

- *Decentralized and Parallel Primal and Dual Accelerated Methods for Stochastic Convex Programming Problems* ([arXiv:1904.09015](https://arxiv.org/abs/1904.09015)).
