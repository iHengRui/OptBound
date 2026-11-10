# Profile 09 — STO NC SM SFO TV

## 1. Problem Formulation

Consider stochastic smooth non-convex optimization over a time-varying decentralized network of $n$ nodes, with access to local stochastic first-order oracles:

$$
\min_{x\in\mathbb R^d} f(x),
\qquad
f(x):=\frac1n\sum_{i=1}^n f_i(x),
$$

where the local objective

$$
f_i(x):=\mathbb E_{\xi_i\sim\mathsf P_i}[F_i(x;\xi_i)]
$$

is private to node $i\in[n]:=\{1,\ldots,n\}$. 

All vector norms are Euclidean; matrix $\|\cdot\|_2$ denotes the spectral norm.

The local oracle is

$$
O_i(x):=g_i(x;\xi_i)\in\mathbb R^d.
$$

One declared local oracle call means one stochastic-gradient sample at one query point. A minibatch of $b$ samples costs $b$ calls.

Arbitrary local computation on previously acquired information is permitted.

The nodes communicate through a sequence of time-varying mixing matrices $(W_q)_{q\ge0}$ fixed before the algorithm runs, where $q$ counts communication-matrix updates and $t$ counts algorithm iterations. The matrix $W_q$ is in force after update $q$, with $W_0$ the initial matrix. Updates occur every fixed absolute constant number of iterations, independently of whether communication occurs:

$$
Z^+=W_qZ,
\qquad
Z=(z_1^{\top},\ldots,z_n^{\top})^{\top}.
$$

Here, $Z$ and $Z^+$ denote the inputs and outputs of the mixing
operation. Write $w_{ij,q}:=(W_q)_{ij}$ and let

$$
\mathcal N_q(i):=\{i\}\cup\{j\in[n]:w_{ij,q}>0\}.
$$

Node $i$ knows the $i$-th row of $W_q$ and can access only the weighted
sum $\sum_{j\in\mathcal N_q(i)}w_{ij,q}z_j$, not the individual
vectors $\{z_j\}_{j\in\mathcal N_q(i)}$.
One communication round may use the same $W_q$ to mix at most $c$
vectors in $\mathbb R^d$ per node, where $c$ is an absolute constant
independent of all problem parameters.

The retained parameters satisfy

$$
n\ge2,\qquad L>0,\quad\Delta>0,\quad\sigma>0,
\qquad p\in\mathbb N_{\ge1},\quad 0\le\rho<1.
$$

All nodes start from the common initial point $x_0$. Initial auxiliary vector information contains no objective-dependent advice. Oracle calls and communication rounds used to initialize auxiliary states are charged.

Let $f^\star:=\inf_{x\in\mathbb R^d}f(x)$. Define the function class

$$
\begin{aligned}
\mathcal F_{L,\Delta}
:=\Bigl\{(f)_\dagger=(d,f_1,\ldots,f_n,x_0):\;&
x_0\in\mathbb R^d,\\
&d\in\mathbb N,\\
&f_i\text{ is }L\text{-smooth for every }i\in[n],\\
&f^\star>-\infty,\\
&f(x_0)-f^\star\le\Delta
\Bigr\}.
\end{aligned}
$$

For a fixed function instance, define the compatible oracle class

$$
\begin{aligned}
\mathcal O_{\sigma}((f)_\dagger)
:=\Bigl\{(O)_\dagger=(O_1,\ldots,O_n):\;&
\mathbb E[O_i(x)\mid\mathscr H_t]=\nabla f_i(x),\\
&\mathbb E[\|O_i(x)-\nabla f_i(x)\|^2\mid\mathscr H_t]
\le\sigma^2
\Bigr\}.
\end{aligned}
$$

Here, $\mathscr H_t$ denotes the history immediately before the oracle queries at iteration $t$, and the conditions hold for every $i\in[n]$ and every $\mathscr H_t$-measurable query point $x$. Oracles in this class use fresh randomness independent of the past, with samples conditionally independent across nodes given $\mathscr H_t$.


Let $J:=\frac{1}{n}\mathbf 1\mathbf 1^\top$. Here $p$ counts consecutive matrix-update intervals. Define the time-varying communication class by the quantitative product-contraction condition

$$
\begin{aligned}
\mathcal W_{p,\rho}:=\Bigl\{(W)_\dagger=(W_q)_{q\ge0}:\;&
W_q\in\mathbb R^{n\times n},\quad W_q\ge0,\\
&\mathbf1^\top W_q=\mathbf1^\top,\quad W_q\mathbf1=\mathbf1,\\
&\left\|W_{q+p-1}\cdots W_q-J\right\|_2\le\rho
\quad\text{for every }q\ge0
\Bigr\}.
\end{aligned}
$$

The algorithm has access to $n,L,\Delta,\sigma,p,\rho$, the target $\varepsilon$, its node identity, and its current local matrix row when communication occurs. Its decisions use only this information, its local oracle history, received mixed messages, and internal randomness. Shared randomness independent of the problem instance is allowed. Define the algorithm class

$$
\mathcal A:=\Bigl\{A:\text{randomized zero-respecting algorithms using,
per node and iteration, at most one stochastic-gradient query
and one communication round}\Bigr\}.
$$

Here, zero-respecting requires that no node can query its oracle at, communicate, or output a vector with a non-zero entry in any coordinate that is zero in its initial point, in all of its previous oracle outputs, and in all of the mixed messages it has received.

For lower-bound constructions, the dimension $d$ may be chosen sufficiently large. Define the averaged output

$$
\widehat x_t:=\frac{1}{n}\sum_{i=1}^n\widehat x_{t,i},
$$

where $\widehat x_{t,i}$ is node $i$'s locally available output after iteration $t$.

This arithmetic average is an evaluation convention, not a free global operation available to the algorithm. 

Define the convergence metric

$$
T_\varepsilon(A,(f)_\dagger,(O)_\dagger,(W)_\dagger)
:=\inf\left\{T\in\mathbb N_{\ge1}:
\frac1T\sum_{t=1}^{T}
\mathbb E\bigl[\|\nabla f(\widehat x_t)\|^2\bigr]\le\varepsilon
\right\},
$$

with $\inf\varnothing=+\infty$. The expectation is over oracle draws and algorithmic randomness. 

The final problem is to determine the order of

$$
\inf_{A\in\mathcal A}
\sup_{(f)_\dagger\in\mathcal F_{L,\Delta}}
\sup_{(O)_\dagger\in\mathcal O_{\sigma}((f)_\dagger)}
\sup_{(W)_\dagger\in\mathcal W_{p,\rho}}
T_\varepsilon(A,(f)_\dagger,(O)_\dagger,(W)_\dagger).
$$

## 2. Task Description

The goal is to give matching upper and lower bounds for this inf-sup problem.

- For the upper bound, construct an algorithm and prove its $\mathcal O(\cdot)$ iteration complexity uniformly over the declared function, oracle, and communication classes. An implementation may use an absolute constant number of the permitted oracle calls or communication rounds in one algorithmic iteration: splitting it into a constant number of unit iterations while respecting the scheduled matrix updates satisfies the declared per-iteration budgets without changing the order of the iteration complexity. All waiting and alignment iterations are charged. For clarity, descriptions with an absolute constant number of operations per iteration are encouraged instead of explicitly expanding every substep into separate iterations.
- For the lower bound, for every admissible algorithm construct a hard function, oracle, and communication instance giving an $\Omega(\cdot)$ iteration lower bound.
- State the dependence on $n,L,\Delta,\sigma,p,\rho$, and $\varepsilon$. Work with $n\ge2$, $p\in\mathbb N_{\ge1}$, $0\le\rho<1$, $L,\Delta,\sigma>0$, and target accuracy $\varepsilon>0$. State any accuracy restrictions required by the bounds. Do not omit a term merely because its $\varepsilon$ dependence is dominated when its dependence on another retained parameter differs.
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

- Function class: require each $f_i$ to be $L_i$-smooth with $L_i\ge0$ and $n^{-1}\sum_iL_i\le L$, $(-)$; impose a uniform heterogeneity bound $n^{-1}\sum_i\|\nabla f_i(x)-\nabla f(x)\|^2\le\zeta^2$ for all $x$, with $\zeta\ge0$ retained as a parameter, $(+)$.
- Oracle class: allow correlation across nodes' simultaneous samples, $(-)$.
- Communication class: require each mixing matrix to be symmetric, $W_q=W_q^\top$ for every $q$, $(+)$.
- Algorithm class: drop the zero-respecting requirement, $(+)$.
- Convergence metric: additionally require $L^2(nT)^{-1}\sum_{t=1}^T\sum_{i=1}^n\mathbb E\|\widehat x_{t,i}-\widehat x_t\|^2\le\varepsilon$, $(-)$.

Only alternatives preserving the target geometry, oracle type, and decentralized communication model are admissible. Forbidden changes are changing to convex, strongly convex, or PL geometry; restricting communication to static networks; replacing decentralized communication by centralized communication; or granting a deterministic full-gradient oracle for every instance. These forbidden changes are not alternatives in the preceding list.

## 4. Related Work

- *Optimal Complexity in Non-Convex Decentralized Learning over Time-Varying Networks* ([arXiv:2211.00533v1](https://arxiv.org/abs/2211.00533v1)).
