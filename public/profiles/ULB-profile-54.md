# STO-SC-NS-SFO-STAT

## 1. Problem Formulation

Consider stochastic nonsmooth strongly convex optimization over a fixed undirected decentralized network of $n$ nodes, with access to local stochastic first-order oracles:

$$
\min_{x\in Q_R} f(x),
\qquad
f(x):=\frac1n\sum_{i=1}^n f_i(x),
$$

where the local objective

$$
f_i(x):=\mathbb E_{\xi_i\sim\mathsf P_i}[F_i(x;\xi_i)]
$$

is private to node $i\in[n]:=\{1,\ldots,n\}$. Each $f_i$ is $\mu$-strongly convex; its subgradients on $Q_R$ have norm at most $M$. Sample-wise convexity and smoothness are not imposed in the standard class.

All vector norms are Euclidean; matrix $\|\cdot\|_2$ denotes the spectral norm.

Let $Q_R:=\{x\in\mathbb R^d:\|x\|_2\le R\}$ be the common known feasible ball. All oracle query points and primal outputs lie in $Q_R$. Euclidean projection onto this known ball is local computation.

The local oracle is

$$
O_i(x):=g_i(x;\xi_i)\in\mathbb R^d.
$$

One declared local oracle call means one stochastic-subgradient sample at one query point. Here $\partial f_i$ denotes the convex subdifferential of $f_i$. A minibatch of $b$ samples costs $b$ calls; the algorithm cannot request reuse of a previous sample.

Arbitrary local computation on previously acquired information is permitted.

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

The retained parameters satisfy

$$
n\ge2,\qquad R,\mu>0,\quad M\ge\mu R,\quad\sigma\ge0,
\qquad 0\le\rho<1.
$$

All nodes start from the common initial point $x_0$. Initial auxiliary vector information contains no objective-dependent advice. Oracle calls and communication rounds used to initialize auxiliary states are charged.

Define the function class

$$
\begin{aligned}
\mathcal F_{M,\mu,R}:=\Bigl\{(f)_\dagger=(f_1,\ldots,f_n,x_0;\{F_i,\mathsf P_i\}_i):\;&
x_0\in Q_R,\\
&f_i:\mathbb R^d\to\mathbb R\text{ is finite and }\mu\text{-strongly convex},\\
&\|v\|_2\le M\quad(\forall i,\ \forall x\in Q_R,\ \forall v\in\partial f_i(x)),\\
&\exists x^\star\in Q_R\cap\operatorname*{argmin}_x f(x),\quad \|x_0-x^\star\|\le R
\Bigr\}.
\end{aligned}
$$

For a fixed function instance, define the compatible oracle class

$$
\begin{aligned}
\mathcal O_{\sigma}((f)_\dagger)
:=\Bigl\{(O)_\dagger=(O_1,\ldots,O_n):\;&
v_{i,t}(x):=\mathbb E[O_i(x)\mid\mathscr H_t]\in\partial f_i(x),\\
&\mathbb E[\|O_i(x)-v_{i,t}(x)\|^2\mid\mathscr H_t]
\le\sigma^2
\Bigr\}.
\end{aligned}
$$

Here, $\mathscr H_t$ denotes the history immediately before the oracle queries at iteration $t$, and the conditions hold for every $i\in[n]$ and every $\mathscr H_t$-measurable query point $x\in Q_R$. Oracles in this class use fresh randomness independent of the past, with samples conditionally independent across nodes given $\mathscr H_t$.


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

The algorithm has access to $n,M,\mu,R,\sigma,\rho$, the target $\varepsilon$, its node identity, and its local row of $W$. Its decisions use only this information, its local oracle history, received mixed messages, and internal randomness. Shared randomness independent of the problem instance is allowed. Define the algorithm class

$$
\mathcal A:=\Bigl\{A:\text{randomized zero-respecting algorithms using,
per node and iteration, at most one stochastic-subgradient query
and one communication round}\Bigr\}.
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

The expectation is over oracle draws and algorithmic randomness; set $T_\varepsilon=+\infty$ if the set is empty. Each oracle and communication slot may be idle. This arithmetic average is an evaluation convention, not a free global operation available to the algorithm.

The final problem is to determine the order of

$$
\inf_{A\in\mathcal A}
\sup_{d\ge1}
\sup_{(f)_\dagger\in\mathcal F_{M,\mu,R}}
\sup_{(O)_\dagger\in\mathcal O_\sigma((f)_\dagger)}
\sup_{(W)_\dagger\in\mathcal W_{n,\rho}}
T_\varepsilon(A,(f)_\dagger,(O)_\dagger,(W)_\dagger).
$$

Here $A$ denotes an algorithm family accepting dimension $d$.

## 2. Task Description

The goal is to give matching upper and lower bounds for this inf-sup problem.

- For the upper bound, construct an algorithm and prove its $\mathcal O(\cdot)$ iteration complexity uniformly over the declared function, oracle, and communication classes. An implementation may use an absolute constant number of oracle calls or communication rounds in one algorithmic iteration: this can be represented by splitting that iteration into a constant number of unit iterations, which preserves the order of the iteration complexity. For clarity, such constant-budget implementations are encouraged instead of literally expanding every substep into separate iterations.
- For the lower bound, for every admissible algorithm construct a hard function, oracle, and communication matrix giving an $\Omega(\cdot)$ iteration lower bound.
- State the dependence on $n,M,\mu,R,\sigma,\rho$, and $\varepsilon$. Work in small $\varepsilon$ scenario and do not omit a term merely because its $\varepsilon$ dependence is dominated when its dependence on another retained parameter differs.
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

- Function class: require strong convexity only of the aggregate $f$ on $Q_R$, retaining local convexity and the declared subgradient bound, $(-)$; impose a common minimizer for all local functions, $(+)$.
- Oracle class: allow correlation across simultaneous samples at different nodes, $(-)$.
- Communication class: remove the quantitative contraction bound while retaining connected static symmetric mixing, $(-)$; prescribe a particular topology or require $W\succeq0$, $(+)$.
- Algorithm class: drop the zero-respecting requirement, $(+)$.
- Convergence metric: additionally require every local output to meet the objective-gap tolerance or impose a consensus-error tolerance, $(-)$.

Only alternatives preserving strongly convex nonsmooth optimization, oracle type, and static decentralized communication model are admissible. Changing to merely convex, non-convex, or PL geometry; replacing stochastic subgradient access by exact or smooth-gradient access; or replacing static decentralized communication by a time-varying or centralized model is forbidden. These forbidden changes are not alternatives in the preceding list.

## 4. Related Work

- *Decentralized and Parallel Primal and Dual Accelerated Methods for Stochastic Convex Programming Problems* ([arXiv:1904.09015](https://arxiv.org/abs/1904.09015)).
