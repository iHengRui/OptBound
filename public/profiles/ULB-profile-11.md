# SUM-NC-SM-SFO-STAT

## 1. Problem Formulation

Consider smooth non-convex finite-sum decentralized optimization, over a fixed mixing matrix and with access to local component-gradient oracles:

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

The nodes communicate through a fixed symmetric matrix $W$. Write

$$
Z_{t+1}=WZ_t,
\qquad
Z_t=(z_{1,t}^{\top},\ldots,z_{n,t}^{\top})^{\top}.
$$

Here $t$ indexes communication rounds, and $z_{i,t}\in\mathbb R^d$ is the vector node $i$ submits for mixing in round $t$; the displayed equation specifies the mixing operation, not every step of an optimization algorithm. Let $\mathcal N(i):=\{i\}\cup\{j\in[n]:w_{ij}>0\}$. Node $i$ knows the $i$th row of $W$ and obtains the weighted sum $\sum_{j\in\mathcal N(i)}w_{ij}z_{j,t}$, but not the individual vectors sent by its neighbors. One communication round may use the same $W$ to mix at most $c$ vectors in $\mathbb R^d$ per node, where $c$ is an absolute constant independent of all problem parameters.

Fix $n\ge2$, $m\ge1$, $L,\Delta>0$, $\rho\in[0,1)$, and $\varepsilon>0$. All nodes start from the common initial point $x_0$. Define the function class

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

Let $J:=n^{-1}\mathbf1\mathbf1^\top$. Define the fixed quantitative communication class

$$
\mathcal W_\rho
:=\Bigl\{(W)_\dagger=W:
W\ge0,\quad W=W^\top,\quad W\mathbf1=\mathbf1,\quad
\mathbf1^\top W=\mathbf1^\top,\quad
\|W-J\|_2\le\rho
\Bigr\}.
$$

The algorithm has access to $n,m,L,\Delta,\rho$, and each node knows its own row of $W$. Define the algorithm class

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
\sup_{(W)_\dagger\in\mathcal W_\rho}
T_\varepsilon(A,(f)_\dagger,(O)_\dagger,(W)_\dagger).
$$

## 2. Task Description

The goal is to give matching upper and lower bounds for this inf-sup problem.

- For the upper bound, construct an algorithm and prove its $\mathcal O(\cdot)$ iteration complexity uniformly over the declared function, oracle, and communication classes. An implementation may use an absolute constant number of oracle calls or communication rounds in one algorithmic iteration: this can be represented by splitting that iteration into a constant number of unit iterations, which preserves the order of the iteration complexity. For clarity, such constant-budget implementations are encouraged instead of literally expanding every substep into separate iterations.
- For the lower bound, for every admissible algorithm construct a hard function, oracle, and communication instance giving an $\Omega(\cdot)$ iteration lower bound.
- State the dependence on $n,m,L,\Delta,\rho$, and $\varepsilon$. It suffices to consider the regime $\varepsilon\downarrow0$. Do not omit a term merely because its $\varepsilon$ dependence is dominated when its dependence on another retained parameter differs.
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
- Communication class: remove the contraction bound $\|W-J\|_2\le\rho$, $(-)$; require $W$ to be positive semidefinite or prescribe a fixed topology, $(+)$.
- Algorithm class: drop zero-respecting, $(+)$.
- Convergence metric: additionally require $L^2(nT)^{-1}\sum_{t=1}^T\sum_{i=1}^n\mathbb E\|\widehat x_{t,i}-\widehat x_t\|^2\le\varepsilon$, $(-)$.

Only alternatives preserving the target geometry, oracle type, and decentralized communication model are admissible. Forbidden changes are replacing non-convex geometry by strong convexity, convexity, or PL geometry; replacing the finite sum by an expectation objective; treating a full local gradient as one component-oracle call; or replacing decentralized communication by centralized communication. These forbidden changes are not alternatives in the preceding list.

## 4. Related Work

- *On the Complexity of Decentralized Smooth Nonconvex Finite-Sum Optimization* ([arXiv:2210.13931](https://arxiv.org/abs/2210.13931)).
- *DeMuon: A Decentralized Muon for Matrix Optimization over Graphs* ([arXiv:2510.01377](https://arxiv.org/abs/2510.01377)).
