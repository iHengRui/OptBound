# SUM-NC-SM-SFO-TV

## 1. Problem Formulation

Consider smooth non-convex finite-sum decentralized optimization, over time-varying mixing matrices and with access to local component-gradient oracles:

$$
\min_{x\in\mathbb R^d} f(x):=\frac1n\sum_{i=1}^n f_i(x),
$$

where the local objective

$$
f_i(x):=\frac1m\sum_{\xi\in D_i}F_i(x;\xi),
\qquad |D_i|=m,
$$

is private to node $i\in[n]$. Here, a component is one function $F_i(\cdot;\xi)$; $f_i$ is the average of the $m$ components at node $i$. Write $f^\star:=\inf_x f(x)>-\infty$. Every component is $L$-smooth, so each $f_i$ and the aggregate $f$ are also $L$-smooth. Components and local objectives need not be convex.

The local oracle is

$$
O_i(x,\xi):=\nabla F_i(x;\xi),
\qquad \xi\in D_i.
$$

One gradient-oracle call means one exact component-gradient query, not evaluation of the full local gradient $\nabla f_i(x)$. The algorithm may choose the queried component and may be randomized.

The nodes communicate through a sequence of time-varying mixing matrices $(W_q)_{q\ge0}$ fixed before the algorithm runs. Here $q$ counts communication-matrix updates: $W_q$ is the matrix in force after update $q$, with $W_0$ the initial matrix. A common, known schedule updates the matrix after every fixed absolute constant number of elementary iterations, independently of all problem parameters. Between updates the matrix remains fixed; idle communication slots do not delay the schedule. Each communication operation uses the current matrix:

$$
Z^+=W_qZ,
\qquad
Z=(z_1^{\top},\ldots,z_n^{\top})^{\top}.
$$

Here, $Z$ and $Z^+$ denote the inputs and outputs of the mixing operation. Write $w_{ij,q}:=(W_q)_{ij}$ and let

$$
\mathcal N_q(i):=\{i\}\cup\{j\in[n]:w_{ij,q}>0\}.
$$

Node $i$ knows the $i$-th row of $W_q$ and can access only the weighted sum $\sum_{j\in\mathcal N_q(i)}w_{ij,q}z_j$, not the individual vectors $\{z_j\}_{j\in\mathcal N_q(i)}$. One communication round may use the same $W_q$ to mix at most $c$ vectors in $\mathbb R^d$ per node, where $c$ is an absolute constant independent of all problem parameters. This convention covers methods that simultaneously communicate, for example, a primal variable and a gradient-tracking variable.

Fix $n\ge2$, $m\ge1$, $L,\Delta>0$, $p\in\mathbb N_{\ge1}$, $\rho\in[0,1)$, and $\varepsilon>0$. All nodes start from the common initial point $x_0$. Define the function class

$$
\begin{aligned}
\mathcal F_{L,\Delta,m}
:=\Bigl\{(f)_\dagger=(f_1,\ldots,f_n,x_0;\{F_i(\cdot;\xi)\}_{i,\xi}):\;&
x_0\in\mathbb R^d,\\
&F_i(\cdot;\xi)\text{ is }L\text{-smooth for every }i,\xi,\\
&f^\star>-\infty,\quad f(x_0)-f^\star\le\Delta,\\
\Bigr\}.
\end{aligned}
$$

Define the oracle class

$$
\mathcal O_{\mathrm{IFO}}
:=\Bigl\{(O)_\dagger=(O_1,\ldots,O_n):
O_i(x,\xi)=\nabla F_i(x;\xi),\ \xi\in D_i
\Bigr\}.
$$

Let $J:=\frac{1}{n}\mathbf 1\mathbf 1^\top$. Define the time-varying communication class by the quantitative product-contraction condition

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

Here $p$ counts consecutive matrix-update intervals, with each interval's matrix appearing once in the product; $t$ counts elementary iterations, including waiting.

The algorithm has access to $n,m,L,\Delta,p,\rho$, and each node knows its current local row of $W_q$. Define the algorithm class

$$
\mathcal A:=\Bigl\{A:
\text{randomized zero-respecting algorithms using, per node and iteration,
at most one component-gradient query and one communication round}
\Bigr\}.
$$

Here, zero-respecting requires that no node can query its oracle at, communicate, or output a vector with a non-zero entry in any coordinate that is zero in its initial point, in all of its previous oracle outputs, and in all of the mixed messages it has received.

For lower-bound constructions, the dimension $d$ may be chosen sufficiently large. Let $\widehat x_{t,i}$ be node $i$'s local output after $t$ iterations. Define the averaged output

$$
\widehat x_t:=\frac1n\sum_{i=1}^n\widehat x_{t,i}
$$

and the convergence metric

$$
T_\varepsilon(A,(f)_\dagger,(O)_\dagger,(W)_\dagger)
:=\inf\left\{T\in\mathbb N_{\ge1}:
\frac1T\sum_{t=1}^{T}\mathbb E\bigl[\|\nabla f(\widehat x_t)\|^2\bigr]\le\varepsilon
\right\}.
$$

The expectation is over all internal randomness of the algorithm, including any randomized choice of component indices.

The final problem is to determine the order of

$$
\inf_{A\in\mathcal A}
\sup_{(f)_\dagger\in\mathcal F_{L,\Delta,m}}
\sup_{(O)_\dagger\in\mathcal O_{\mathrm{IFO}}}
\sup_{(W)_\dagger\in\mathcal W_{p,\rho}}
T_\varepsilon(A,(f)_\dagger,(O)_\dagger,(W)_\dagger).
$$

## 2. Task Description

The goal is to give matching upper and lower bounds for this inf-sup problem.

- For the upper bound, construct an algorithm and prove its $\mathcal O(\cdot)$ iteration complexity uniformly over the declared function, oracle, and communication classes. An implementation may use an absolute constant number of oracle calls or communication rounds in one algorithmic iteration: this can be represented by splitting that iteration into a constant number of unit iterations while respecting the scheduled matrix updates, which preserves the order of the iteration complexity. All waiting and alignment iterations are charged. For clarity, such constant-budget implementations are encouraged instead of literally expanding every substep into separate iterations.
- For the lower bound, for every admissible algorithm construct a hard function, oracle, and communication sequence giving an $\Omega(\cdot)$ iteration lower bound.
- State the dependence on $n,m,L,\Delta,p,\rho$, and $\varepsilon$. It suffices to consider the regime $\varepsilon\downarrow0$. Do not omit a term merely because its $\varepsilon$ dependence is dominated when its dependence on another retained parameter differs.
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

- Function class: require only the average $f$ to be $L$-smooth, with each component $F_i(\cdot;\xi)$ merely differentiable, $(-)$; retain $L$-smoothness of $f$ but replace componentwise $L$-smoothness by global mean-squared component smoothness, $\frac{1}{nm}\sum_{i=1}^{n}\sum_{\xi\in D_i}\|\nabla F_i(x;\xi)-\nabla F_i(y;\xi)\|^2\le\bar L^2\|x-y\|^2$ for all $x,y$, where $\bar L\ge L$ is known to the algorithm, $(-)$.
- Oracle class: replace algorithm-selected indices by indices sampled independently and uniformly from $D_i$ by the oracle, $(+)$.
- Communication class: remove the $(p,\rho)$ product-contraction bound, $(-)$; require every $W_q$ to be symmetric, $(+)$.
- Algorithm class: drop zero-respecting, $(+)$.
- Convergence metric: additionally require $L^2(nT)^{-1}\sum_{t=1}^T\sum_{i=1}^n\mathbb E\|\widehat x_{t,i}-\widehat x_t\|^2\le\varepsilon$, $(-)$.

Only alternatives preserving the target geometry, oracle type, and decentralized communication model are admissible. Forbidden changes are replacing non-convex geometry by strong convexity, convexity, or PL geometry; replacing the finite sum by an expectation objective; treating a full local gradient as one component-oracle call; or replacing decentralized communication by centralized communication. These forbidden changes are not alternatives in the preceding list.

## 4. Related Work

- *Decentralized Finite-Sum Optimization over Time-Varying Networks* ([arXiv:2402.02490](https://arxiv.org/abs/2402.02490)).
