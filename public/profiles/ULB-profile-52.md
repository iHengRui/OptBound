# STO-SC-SM-SFO-STAT

## 1. Problem Formulation

Consider stochastic smooth strongly convex optimization over a fixed decentralized network of $n$ nodes, with access to local stochastic first-order oracles:

$$
\min_{x\in\mathbb R^d} f(x),
\qquad
f(x):=\frac1n\sum_{i=1}^n f_i(x),
$$

where the local objective

$$
f_i(x):=\mathbb E_{\xi_i\sim\mathsf P_i}[F_i(x;\xi_i)]
$$

is private to node $i\in[n]:=\{1,\ldots,n\}$. Each $f_i$ is $\mu$-strongly convex and $L$-smooth; sample-wise convexity and smoothness are not imposed in the standard class.

All vector norms are Euclidean; matrix $\|\cdot\|_2$ denotes the spectral norm.

The local oracle is

$$
O_i(x):=g_i(x;\xi_i)\in\mathbb R^d.
$$

One declared local oracle call means one stochastic-gradient sample at one query point. A minibatch of $b$ samples costs $b$ calls; the algorithm cannot request reuse of a previous sample.

Arbitrary local computation on previously acquired information is permitted.

The nodes communicate through a fixed symmetric matrix $W$. Write

$$
Z_{t+1}=WZ_t,
\qquad
Z_t=(z_{1,t}^{\top},\ldots,z_{n,t}^{\top})^{\top}.
$$

Here $t$ indexes communication rounds, and $z_{i,t}\in\mathbb R^d$ is the vector node $i$ submits for mixing in round $t$; the displayed equation specifies the mixing operation, not every step of an optimization algorithm. Let $\mathcal N(i):=\{i\}\cup\{j\in[n]:w_{ij}>0\}$. Node $i$ knows the $i$th row of $W$ and obtains the weighted sum $\sum_{j\in\mathcal N(i)}w_{ij}z_{j,t}$, but not the individual vectors sent by its neighbors. One communication round may use the same $W$ to mix at most $c$ vectors in $\mathbb R^d$ per node, where $c$ is an absolute constant independent of all problem parameters.

The retained parameters satisfy

$$
n\ge2,\qquad L\ge\mu>0,\quad R>0,\quad\sigma\ge0,
\qquad 0\le\rho<1.
$$

All nodes start from the common initial point $x_0$. Initial auxiliary vector information contains no objective-dependent advice. Oracle calls and communication rounds used to initialize auxiliary states are charged.

Define the function class

$$
\begin{aligned}
\mathcal F_{L,\mu,R}:=\Bigl\{(f)_\dagger=(f_1,\ldots,f_n,x_0;\{F_i,\mathsf P_i\}_i):\;&
x_0\in\mathbb R^d,\\
&f_i\text{ is }\mu\text{-strongly convex and }L\text{-smooth},\\
&\exists x^\star\in\operatorname*{argmin}_x f(x),\quad \|x_0-x^\star\|\le R
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


Let $J:=n^{-1}\mathbf1\mathbf1^\top$. Define the fixed quantitative communication class

$$
\mathcal W_\rho
:=\Bigl\{(W)_\dagger=W:
W\ge0,\quad W=W^\top,\quad W\mathbf1=\mathbf1,\quad
\mathbf1^\top W=\mathbf1^\top,\quad
\|W-J\|_2\le\rho
\Bigr\}.
$$

The algorithm has access to $n,L,\mu,R,\sigma,\rho$, the target $\varepsilon$, its node identity, and its local row of $W$. Its decisions use only this information, its local oracle history, received mixed messages, and internal randomness. Shared randomness independent of the problem instance is allowed. Define the algorithm class

$$
\mathcal A:=\Bigl\{A:\text{randomized zero-respecting algorithms using,
per node and iteration, at most one stochastic-gradient query
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
\sup_{(f)_\dagger\in\mathcal F_{L,\mu,R}}
\sup_{(O)_\dagger\in\mathcal O_\sigma((f)_\dagger)}
\sup_{(W)_\dagger\in\mathcal W_\rho}
T_\varepsilon(A,(f)_\dagger,(O)_\dagger,(W)_\dagger).
$$

Here $A$ denotes an algorithm family accepting dimension $d$.

## 2. Task Description

The goal is to give matching upper and lower bounds for this inf-sup problem.

- For the upper bound, construct an algorithm and prove its $\mathcal O(\cdot)$ iteration complexity uniformly over the declared function, oracle, and communication classes. An implementation may use an absolute constant number of oracle calls or communication rounds in one algorithmic iteration: this can be represented by splitting that iteration into a constant number of unit iterations, which preserves the order of the iteration complexity. For clarity, such constant-budget implementations are encouraged instead of literally expanding every substep into separate iterations.
- For the lower bound, for every admissible algorithm construct a hard function, oracle, and communication instance giving an $\Omega(\cdot)$ iteration lower bound.
- State the dependence on $n,L,\mu,R,\sigma,\rho$, and $\varepsilon$. Work in small $\varepsilon$ scenario and do not omit a term merely because its $\varepsilon$ dependence is dominated when its dependence on another retained parameter differs.
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

- Function class: require only the aggregate $f$ to be $L$-smooth while retaining local strong convexity, $(-)$; require convex $L$-smooth sample functions and a known bound on gradient dissimilarity at an optimizer, $(+)$.
- Oracle class: allow correlation across simultaneous samples at different nodes, $(-)$.
- Communication class: remove the contraction bound $\|W-J\|_2\le\rho$, $(-)$; require $W$ to be positive semidefinite or prescribe a fixed topology, $(+)$.
- Algorithm class: drop the zero-respecting requirement, $(+)$.
- Convergence metric: additionally require every local output to meet the objective-gap tolerance or impose a consensus-error tolerance, $(-)$.

Only alternatives preserving the target geometry, oracle type, and static decentralized communication model are admissible. Changing to merely convex, non-convex, or PL geometry; replacing stochastic first-order access by an exact full-gradient oracle; or replacing static decentralized communication by a time-varying or centralized model is forbidden. These forbidden changes are not alternatives in the preceding list.

## 4. Related Work

- *Asymptotic Properties of S-AB Method with Diminishing Stepsize* ([arXiv:2109.07981](https://arxiv.org/abs/2109.07981)).
