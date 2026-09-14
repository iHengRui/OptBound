# Decentralized LB/UB Table

## Notation

### Objective functions, accuracy, and stochastic model

\(F\) denotes global objective, \(F(x)=N^{-1}\sum_{i=1}^N F_i(x)\); \(F_i\) denotes local objective at node \(i\); \(F^\star\) denotes optimal value, \(F^\star=\inf_x F(x)\); \(\eps\) denotes target accuracy, \(\eps>0\); \(\E\) denotes expectation operator; \(\sigma^2\) denotes stochastic-gradient variance bound, \(\E\|g_i(x;\xi_i)-\nabla F_i(x)\|^2\le\sigma^2\); \(g_i(x;\xi_i)\) denotes the stochastic first-order oracle output at node \(i\), evaluated at \(x\) using local random sample \(\xi_i\); \(\sigma\) denotes standard-deviation bound, \(\sigma=\sqrt{\sigma^2}\); \(\zeta_{\rm stat}^2\) denotes uniform cross-node gradient-heterogeneity bound satisfying \(N^{-1}\sum_{i=1}^N\|\nabla F_i(x)-\nabla F(x)\|^2\le\zeta_{\rm stat}^2\) for all \(x\) [[48]](#ref-arxiv-2412-07252); \(\delta_{\rm ht}\) denotes failure probability in a high-probability guarantee, \(\delta_{\rm ht}\in(0,1)\) [[10]](#ref-arxiv-2006-08085) [[26]](#ref-arxiv-1904-09015) [[52]](#ref-arxiv-1406-0919); \(\Pr\) denotes probability under the stochastic oracle and algorithm randomness [[26]](#ref-arxiv-1904-09015) [[52]](#ref-arxiv-1406-0919).

### Problem geometry and parameters

\(\Delta\) denotes initial optimality gap, \(\Delta=F(x^0)-F^\star\); \(\Delta_0\) denotes initial objective-gap bound, \(\Delta_0\ge F(x^0)-F^\star\) [[27]](#ref-arxiv-2210-03232) [[41]](#ref-arxiv-1912-12110) [[11]](#ref-arxiv-2606-04757); \(N\) denotes number of nodes; \(T\) denotes iteration horizon; \(T_{\rm out}\) denotes outer-iteration horizon when each outer step contains a local batch or multiple gossip rounds; \(K_{\rm loc}\) denotes number of local component functions, where \(f_{ij}\) is component \(j\) at node \(i\) and \(F_i(x)=K_{\rm loc}^{-1}\sum_{j=1}^{K_{\rm loc}}f_{ij}(x)\); \(S\) denotes epoch index [[32]](#ref-arxiv-2112-10389); \(L\) denotes smoothness constant satisfying \(\|\nabla F_i(x)-\nabla F_i(y)\|\le L\|x-y\|\) for every node \(i\); \(\bar L\) denotes mean-squared component smoothness constant satisfying \((N K_{\rm loc})^{-1}\sum_{i=1}^N\sum_{j=1}^{K_{\rm loc}}\|\nabla f_{ij}(x)-\nabla f_{ij}(y)\|^2\le \bar L^2\|x-y\|^2\) [[9]](#ref-arxiv-2210-13931); \(\mu\) denotes strong-convexity constant satisfying \(h(y)\ge h(x)+\langle g,y-x\rangle+\frac{\mu}{2}\|y-x\|^2\) for \(g\in\partial h(x)\); \(\kappa\) denotes condition number, \(\kappa=L/\mu\); \(\kappa_s\) denotes mean-squared finite-sum condition number, \(\kappa_s=\bar L/\mu\) [[9]](#ref-arxiv-2210-13931) [[22]](#ref-arxiv-2402-02490); \(\kappa_{s,\rm VR}\) denotes VR-EXTRA/DIGing stochastic condition number, \(\kappa_{s,\rm VR}=\mu^{-1}\max_i K_{\rm loc}^{-1}\sum_{j=1}^{K_{\rm loc}}L_{ij}\), where \(L_{ij}\) is the smoothness constant of component \(j\) at node \(i\) [[20]](#ref-arxiv-2009-04373); \(q_{\rm DPS}\) denotes DPSVRG exponent \(q_{\rm DPS}=\log(1/r_{\rm DPS})/\log\beta_{\rm DPS}\in(0,1)\) [[32]](#ref-arxiv-2112-10389); \(M\) denotes uniform local subgradient bound satisfying \(\|g_i(x)\|\le M\) for every \(g_i(x)\in\partial F_i(x)\); \(R\) denotes initial-distance bound satisfying \(\|x^0-x^\star\|\le R\); \(c\) denotes initial error constant in the primal-dual PL bound [[41]](#ref-arxiv-1912-12110); \(\alpha_{\rm DSP}\) denotes positive constant [[46]](#ref-arxiv-1911-12665); \(\ell_{\rm DSP}\) denotes D-SPIDER-SFO complexity coefficient [[46]](#ref-arxiv-1911-12665); \(C_{\rm DDA}\) denotes DDA constant coefficient [[38]](#ref-arxiv-2211-06914); \(C_2\) denotes DAGP heterogeneity coefficient [[27]](#ref-arxiv-2210-03232); \(\bar\zeta\) denotes root-mean-square local-gradient heterogeneity at a global optimum, \(\bar\zeta=(N^{-1}\sum_{i=1}^N\|\nabla F_i(x^\star)\|^2)^{1/2}\) [[44]](#ref-arxiv-2003-10422); \(p\) denotes higher-order smoothness degree, an integer \(p\ge2\) such that \(\|\mathcal D^pF_i(X)-\mathcal D^pF_i(Y)\|_*\le L_{p,*}\|X-Y\|\) for some \(L_{p,*}>0\), where \(\mathcal D^p\) is the \(p\)-th derivative [[39]](#ref-arxiv-2510-01377); \(\Psi^0\) denotes initial MG-Skip Lyapunov value [[23]](#ref-arxiv-2312-11861); \(C_0\) denotes DAGP initialization coefficient; \(\kappa_{\rm snc}\) denotes sum-of-nonconvex composite condition number, \(\kappa_{\rm snc}=(L_{\rm snc}+\sqrt{\ell_1\ell_2})/\mu\), where \(L_{\rm snc}\) is the smooth aggregate's Lipschitz-gradient constant (the source's \(L\)), \(\mu\) is the total strong convexity, and component Bregman remainders lie between \(-\ell_2\|x-y\|^2/2\) and \(\ell_1\|x-y\|^2/2\), with \(\ell_2\ge\ell_1>0\) [[50]](#ref-arxiv-2402-02356); \(L_{\rm rms,loc}\) denotes uniform node-local RMS component smoothness: for every node \(i\), \(K_{\rm loc}^{-1}\sum_j\|\nabla f_{ij}(x)-\nabla f_{ij}(y)\|^2\le L_{\rm rms,loc}^2\|x-y\|^2\) [[22]](#ref-arxiv-2402-02490); \(L_F\) denotes smoothness constant of the global objective \(F\), used when a source distinguishes global smoothness from node or component smoothness [[9]](#ref-arxiv-2210-13931); \(C_{\rm DPS}\) denotes full DPSVRG initial-gap, distance, and estimator-error constant stated in the row-37 assumption note [[32]](#ref-arxiv-2112-10389); \(M_\phi\) denotes uniform subgradient bound for the nonsmooth convex component \(\phi_i\) in row 55 [[45]](#ref-arxiv-2508-09029).

### Topology, network, and constraints

\(\kappa_A\) denotes equilibrium skewness of row-stochastic matrix \(A\), \(\kappa_A=\max_i(\pi_A)_i/\min_i(\pi_A)_i\) [[47]](#ref-arxiv-2506-04600); \(\kappa_\pi\) denotes equilibrium skewness of column-stochastic matrix \(W_\pi\), \(\kappa_\pi=\max_i\pi_i/\min_i\pi_i\) [[7]](#ref-arxiv-2312-04928); \(\rho\) denotes spectral gap, \(\rho=1-\|W-N^{-1}\mathbf 1_N\mathbf 1_N^\top\|_2\); \(\rho_{\rm K}\) denotes reference-specific spectral gap, \(\rho_{\rm K}=1-\lambda_2(P)\), for the fixed symmetric doubly stochastic gossip matrix \(P\), where \(\lambda_2(P)\) is the second-largest eigenvalue of \(P\) [[11]](#ref-arxiv-2606-04757); \(\beta_{\rm H}\) denotes one-step mixing factor satisfying \(\|W^t-N^{-1}\mathbf 1_N\mathbf 1_N^\top\|_2\le\beta_{\rm H}<1\) [[8]](#ref-arxiv-2211-00533); \(\beta_A\) denotes row-stochastic mixing factor of \(A\), \(\beta_A=\|A-\mathbf 1_N\pi_A^\top\|_{\pi_A}\in[0,1)\), where \(\Pi_A=\operatorname{diag}(\pi_A)\) and \(\|B\|_{\pi_A}:=\|\Pi_A^{1/2}B\Pi_A^{-1/2}\|_2\) is the induced weighted matrix norm [[47]](#ref-arxiv-2506-04600); \(\beta_\pi\) denotes column-stochastic mixing factor of \(W_\pi\), \(\beta_\pi=\|W_\pi-\pi\mathbf 1_N^\top\|_\pi\in[0,1)\), where \(\Pi=\operatorname{diag}(\pi)\) and \(\|B\|_\pi:=\|\Pi^{-1/2}B\Pi^{1/2}\|_2\) is the induced weighted matrix norm [[7]](#ref-arxiv-2312-04928); \(\delta_{\rm net}\) denotes normalized network contraction rate, \(\delta_{\rm net}:=p_{\rm mix}/\tau_{\rm mix}\), where \(1-p_{\rm mix}\) is the \(\tau_{\rm mix}\)-step expected squared-consensus-error factor [[44]](#ref-arxiv-2003-10422); \(W\) denotes doubly stochastic mixing matrix \(W\in\mathbb R^{N\times N}\) satisfying \(W\mathbf 1_N=\mathbf 1_N\) and \(\mathbf 1_N^\top W=\mathbf 1_N^\top\); \(\omega(\widehat W)\) denotes smallest positive eigenvalue of the positive semidefinite gossip/Laplacian operator \(\widehat W\) [[12]](#ref-arxiv-2004-02635); \(C_{\mathbf S}\) denotes DAGP network-and-algorithm factor satisfying \(C_{\mathbf S}>-\lambda_{\min}(\mathbf S)\), where \(\mathbf S\) is the theorem matrix determined by the gossip matrices and algorithm parameters [[27]](#ref-arxiv-2210-03232); \(W_\pi\) denotes primitive column-stochastic mixing matrix satisfying \(\mathbf 1_N^\top W_\pi=\mathbf 1_N^\top\) [[7]](#ref-arxiv-2312-04928); \(\pi\) denotes normalized positive right Perron vector of \(W_\pi\), satisfying \(W_\pi\pi=\pi\) and \(\mathbf 1_N^\top\pi=1\) [[7]](#ref-arxiv-2312-04928); \(\pi_A\) denotes normalized positive left Perron vector of row-stochastic \(A\), satisfying \(\pi_A^\top A=\pi_A^\top\) and \(\pi_A^\top\mathbf 1_N=1\) [[47]](#ref-arxiv-2506-04600); \(p_{\rm mix}\) denotes blockwise expected consensus-contraction amount in the definition of \(\delta_{\rm net}\) [[44]](#ref-arxiv-2003-10422); \(\tau_{\rm AC}\) denotes AC-GT mixing-window coefficient, \(\tau_{\rm AC}=\eta_{\rm AC}\bar\tau d_{\mathcal G}\), where \(\bar\tau\) is the joint-connectivity window, \(d_{\mathcal G}\) is the graph diameter, and \(\eta_{\rm AC}\in\mathbb N\) satisfies the cited block-contraction condition [[18]](#ref-arxiv-2309-02626).

### Iterates and outputs

\(x_i\) denotes local iterate/output at node \(i\); \(x_i^t\) is its value at iteration \(t\), while the unindexed form is the theorem's returned output; \(\bar x\) denotes network-average iterate/output, \(\bar x^t=N^{-1}\sum_{i=1}^N x_i^t\); the unindexed \(\bar{x}\) denotes the averaged output specified by the cited theorem; \(x_{o,i}\) denotes output selected by node \(i\) from its internal memory [[4]](#ref-arxiv-2405-18031); \(x_o^T\) denotes network-average output, \(x_o^T=N^{-1}\sum_{i=1}^N x_{a,i}^T\), where \(x_{a,i}^T\) is node \(i\)'s component of the averaged primal vector \(x_a^T\) [[4]](#ref-arxiv-2405-18031); \(\bar x_{i,T}\) denotes time-averaged output at node \(i\), \(\bar x_{i,T}=T^{-1}\sum_{t=0}^{T-1}x_i^t\) [[27]](#ref-arxiv-2210-03232); \(\bar y_T\) denotes averaged primal output after \(T\) iterations [[29]](#ref-arxiv-2312-15845); \(\hat x_i(T)\) denotes time-averaged output at node \(i\), \(\hat x_i(T)=T^{-1}\sum_{t=1}^T x_i^t\) [[30]](#ref-arxiv-1005-2012); \(\tilde x_i^T\) denotes time-averaged output at node \(i\), \(\tilde x_i^T=T^{-1}\sum_{t=1}^T x_i^t\) [[31]](#ref-arxiv-2506-12901); \(\hat x\) denotes the algorithm output used to evaluate the accuracy criterion in the corresponding source; its specific construction follows the theorem cited in that row [[9]](#ref-arxiv-2210-13931) [[8]](#ref-arxiv-2211-00533) [[6]](#ref-arxiv-2210-07863) [[10]](#ref-arxiv-2006-08085) [[21]](#ref-arxiv-1910-04057) [[22]](#ref-arxiv-2402-02490) [[26]](#ref-arxiv-1904-09015) [[40]](#ref-arxiv-2110-01594) [[49]](#ref-arxiv-2406-01484); \(\hat y\) denotes tuple of node-wise time-averaged outputs [[33]](#ref-arxiv-1411-4186); \(\tilde x\) denotes virtual centralized output after epoch \(S\) in DPSVRG [[32]](#ref-arxiv-2112-10389); \(\tilde x_i\) denotes weighted time-averaged output at node \(i\), \(\tilde x_i=A_T^{-1}\sum_{t=1}^T a_t x_i^t\), where \(A_T=\sum_{t=1}^T a_t\) and \(a_t>0\) are theorem-specified averaging weights [[25]](#ref-arxiv-2106-14075) [[34]](#ref-arxiv-1806-07081); \(x^\star\) denotes optimizer of \(F\), \(x^\star\in\arg\min_x F(x)\); \(X\) denotes stacked local iterate/output, \(X^t=[(x_1^t)^\top,\ldots,(x_N^t)^\top]^\top\); the unindexed form is the returned stacked output; \(X^\star\) denotes stacked consensus optimizer, \(X^\star=\mathbf 1_N\otimes x^\star\); \(y\) denotes consensus auxiliary primal iterate in DDA, \(y^t=\nabla d^*(-\bar z^t)\), where \(d^*\) is the conjugate of the prox function [[38]](#ref-arxiv-2211-06914); \(\mathbf y\) denotes stacked auxiliary iterate in DDA, \(\mathbf y^t=\mathbf 1_N\otimes y^t\) [[38]](#ref-arxiv-2211-06914); \(z\) denotes node-wise dual accumulator \(z_i^t\) in DDA [[38]](#ref-arxiv-2211-06914); \(\bar z\) denotes network-average auxiliary iterate, \(\bar z^t=N^{-1}\sum_{i=1}^N z_i^t\) [[38]](#ref-arxiv-2211-06914); \(z_{\rm VR}^k\) denotes VR-EXTRA/DIGing auxiliary primal iterate returned at iteration \(k\) [[20]](#ref-arxiv-2009-04373); \(\bar x_u^k\) denotes left-Perron-weighted aggregate \(\bar x_u^k=N^{-1}(u^\top\otimes I)X^k\), where \(u^\top R=u^\top\) and \(u^\top\mathbf 1_N=N\) [[36]](#ref-arxiv-2601-03566); \(\bar x_{\pi_{\rm SAB}}^k\) denotes S-AB Perron-weighted aggregate \(\bar x_{\pi_{\rm SAB}}^k=(\pi_{\rm SAB}^\top\otimes I)X^k\), where \(\pi_{\rm SAB}\) is the normalized positive left Perron vector of its row-stochastic mixing matrix [[43]](#ref-arxiv-2109-07981); \(\bar x_{\pi_A}^t\) denotes row-stochastic Perron-weighted aggregate \(\bar x_{\pi_A}^t=(\pi_A^\top\otimes I)X^t\) [[47]](#ref-arxiv-2506-04600); \(\tilde y_{\rm DA}^T\) denotes dual-averaging auxiliary output: the time average of the source's prox points formed from averaged dual accumulators; generally not the mean of local primal outputs [[25]](#ref-arxiv-2106-14075); \(\bar y_{\rm PM}^T\) denotes PMGT-KatyushaX network-average output after \(T\) epochs, \(N^{-1}\sum_i y_i^T\) [[50]](#ref-arxiv-2402-02356).

### Targets and stationarity measures

\(\delta_{\rm G}\) denotes Goldstein-stationarity radius; \(G\) denotes Bregman distance of the stacked separable objective, \(G(X,X^\star)=\sum_{i=1}^N[F_i(X_i)-F_i(x^\star)-\langle\nabla F_i(x^\star),X_i-x^\star\rangle]\), where \(X_i\) is node \(i\)'s block of \(X\) [[5]](#ref-arxiv-1910-10666); \(\mathbf s\) denotes proximal-gradient mapping of \(F+h\), \(\mathbf s(x)=\alpha^{-1}[x-\operatorname{prox}_{\alpha h}(x-\alpha\nabla F(x))]\), where \(h\) is the convex regularizer and \(\alpha>0\) [[40]](#ref-arxiv-2110-01594); \(G_a\) denotes DDA gradient mapping, \(G_a(x,z)=a^{-1}[\nabla d^*(-z)-\nabla d^*(-z-\nabla f(x))]\) [[38]](#ref-arxiv-2211-06914); \(\bar g\) denotes node-averaged local gradient at the stacked iterate, \(\bar g(X)=N^{-1}\sum_{i=1}^N\nabla F_i(X_i)\) [[39]](#ref-arxiv-2510-01377); \(\nabla F\) denotes gradient of the global objective; \(\|\cdot\|\) denotes Euclidean norm of \(v\in\mathbb R^d\), \(\|v\|=(\sum_{r=1}^d v_r^2)^{1/2}\); \(\|\nabla F(\hat x)\|_{\delta_{\rm G}}\) denotes Goldstein stationarity measure with radius \(\delta_{\rm G}\); \(\|\cdot\|_*\) denotes nuclear norm of a matrix \(X\), \(\|X\|_* =\sum_r\sigma_r(X)\) [[39]](#ref-arxiv-2510-01377); \(\|\cdot\|_{\E}\) denotes root-mean-square norm of a random vector \(v\), \(\|v\|_{\mathbb E}=(\mathbb E\|v\|^2)^{1/2}\) [[38]](#ref-arxiv-2211-06914); \(\|\cdot\|_F\) denotes Frobenius norm of \(X\in\mathbb R^{a\times b}\), \(\|X\|_F=(\sum_{r=1}^a\sum_{s=1}^b X_{rs}^2)^{1/2}\); \(V^t\) denotes source-defined primal-dual Lyapunov value combining squared primal error, squared dual error, and gradient-estimator variance [[12]](#ref-arxiv-2004-02635); \(\mathcal E_{\rm obj}\) denotes average local-objective error, \(\mathcal E_{\rm obj}(y_1,\ldots,y_N)=N^{-1}\sum_{i=1}^N F_i(y_i)-F^\star\) [[33]](#ref-arxiv-1411-4186).

### Asymptotic notation

\(O(\cdot)\) denotes asymptotic upper bound, \(a=O(b)\) when \(a\le Cb\) for a constant \(C>0\); \(\Omega(\cdot)\) denotes asymptotic lower bound, \(a=\Omega(b)\) when \(a\ge cb\) for a constant \(c>0\); \(\widetilde O(\cdot)\) denotes asymptotic upper bound hiding poly-logarithmic factors, \(a=\widetilde O(b)\).

### Indices and algorithmic parameters

\(i\) denotes node index; \(j\) denotes local component index; \(t\) denotes iteration index; \(\eta\) denotes positive stepsize [[18]](#ref-arxiv-2309-02626) [[36]](#ref-arxiv-2601-03566); \(\theta\) denotes linear contraction parameter \(\theta\in(0,1)\) in the factor \((1-\theta)^t\) [[41]](#ref-arxiv-1912-12110); \(\eta_{\rm alg}\) denotes positive DAGP auxiliary parameter [[27]](#ref-arxiv-2210-03232); \(p_{\rm comm}\) denotes communication probability, \(p_{\rm comm}\in(0,1]\) [[23]](#ref-arxiv-2312-11861); \(w_t\) denotes positive theorem weights used to average strongly-convex decentralized-SGD objective errors [[44]](#ref-arxiv-2003-10422); \(W_T\) denotes cumulative weight, \(W_T=\sum_{t=0}^T w_t\) [[44]](#ref-arxiv-2003-10422); \(s\) denotes candidate iteration index in \(\min_{1\le s\le t}\); \(\alpha_{\rm DAGP}\) denotes DAGP gradient stepsize, \(\alpha_{\rm DAGP}>0\).

## Tags

### Fixed scope

Every row in the current Decentralized Optimization release assumes peer-to-peer gossip, synchronous communication rounds, and no communication compression. These fixed attributes are omitted from the row labels; the structured problem tags retain them for taxonomy and audit purposes.

### Objective form

`SUM` denotes a deterministic distributed aggregate \(F(x)=\frac{1}{N}\sum_{i=1}^{N}F_i(x)\), where each local objective \(F_i\) is deterministic. A `SUM`+`SFO` row is used when the deterministic local objective is accessed by a sampled oracle: either explicit finite-sum component sampling or an unbiased noisy-gradient oracle for \(F_i\); oracle-only stochasticity is disclosed by a Remark marker. `STO` denotes an expectation objective \(F(x)=\frac{1}{N}\sum_{i=1}^{N}\mathbb{E}_{\xi_i}[f_i(x;\xi_i)]\), where \(\xi_i\) is a local random sample and the objective is accessed through sampled stochastic first-order information. Thus `SUM`/`STO` describes the objective model, whereas the oracle tag below describes what one local query returns.

### Geometry and regularity

`SC` means strong convexity with parameter \(\mu>0\); `C` means convexity without an assumed positive strong-convexity parameter; `NC` means that convexity is not assumed; and `PL` means the classical differentiable Polyak-Lojasiewicz condition \(\frac12\|\nabla F(x)\|^2\geq\mu(F(x)-F^\star)\). The classification follows the level used by the cited statement (global objective, local objectives, or both), with stronger local assumptions retained as caveats. `SM` means differentiability with an \(L\)-Lipschitz gradient at the theorem's stated level; `NS` means that the total objective is not assumed differentiable with an \(L\)-Lipschitz gradient and the result instead uses nonsmooth first-order, subgradient, or proximal information. `COMP` marks a composite objective consisting of a smooth loss and an explicit nonsmooth/proximable regularizer; it supplements rather than replaces `NS`.

### Local oracle

`FO` is an exact local first-order oracle (the exact gradient in the smooth case, or the exact first-order mapping specified by the paper); `SFO` is a sampled stochastic or finite-sum component oracle returning \(g_i(x;\xi_i)\) under the cited theorem's bias, variance, and independence assumptions; and `SUBG` is a deterministic local subgradient oracle returning \(g_i(x)\in\partial F_i(x)\). All displayed complexities count per-node local oracle calls, not total calls summed over the network. `PROX` means that the algorithm additionally assumes exact evaluation of the regularizer's proximal operator; these side-access calls are not included in the displayed `FO`/`SFO` counts.

### Network evolution

`STAT` means that one communication graph and its mixing operator remain fixed over the run. `TV` means that the graph or mixing operator may vary by communication round, subject to the connectivity and mixing assumptions stated by the cited theorem.

| No. | Problem Type | Measure | Lower Bound | Upper Bound | Status | Reference (LB) | Reference (UB) |
|---:|---|---|---|---|---|---|---|
| 1 | `SUM SC SM FO TV`<sup>[S12](#remark-S12)</sup> | \(F(x_i)-F^\star \le \eps\)<sup>[M3](#remark-M3)</sup> | \(\Omega(\sqrt{\kappa} \log(1/\eps))\) | \(O(\sqrt{\kappa}\log(1/\eps))\)<sup>[N2](#remark-N2), [N1](#remark-N1)</sup> | `EXACT` | [[1]](#ref-arxiv-2106-04469), Theorem 3 | [[1]](#ref-arxiv-2106-04469), Theorem 6 |
| 2 | `SUM C NS SUBG STAT` | \(F(\bar x)-F^\star \le \eps\) | \(\Omega(M^2 R^2 / \eps^2)\)<sup>[A10](#remark-A10), [A11](#remark-A11)</sup> | \(O(M^2 R^2 / \eps^2)\)<sup>[A10](#remark-A10)</sup> | `EXACT` | [[2]](#ref-arxiv-1806-00291), Theorem 3 | [[2]](#ref-arxiv-1806-00291), Theorem 5 |
| 3 | `SUM SC SM FO STAT`<sup>[S16](#remark-S16)</sup> | \(F(x_i)-F^\star \le \eps, \forall i\) | \(\Omega(\sqrt{\kappa} \log(1/\eps))\)<sup>[O3](#remark-O3)</sup> | \(O(\sqrt{\kappa} \log(1/\eps))\)<sup>[O3](#remark-O3)</sup> | `COND` | [[3]](#ref-arxiv-1702-08704), Theorem 2 | [[3]](#ref-arxiv-1702-08704), Theorem 4 |
| 4 | `SUM SC NS SUBG TV`<sup>[S12](#remark-S12), [S13](#remark-S13)</sup> | \(F(x_{o,i})-F^\star\le\eps_{\rm LB};\allowbreak F(x_o^T)-F^\star\le\eps_{\rm UB}\)<sup>[M4](#remark-M4)</sup> | \(\Omega(M^2/(\mu \eps))\) | \(O(M^2/(\mu \eps))\) | `COND` | [[4]](#ref-arxiv-2405-18031), Theorem 1 | [[4]](#ref-arxiv-2405-18031), Theorem 3 |
| 5 | `SUM C SM FO STAT`<sup>[S12](#remark-S12)</sup> | \(G(X,X^\star) \le \eps\)<sup>[M1](#remark-M1)</sup> | \(\Omega(\sqrt{LR^2/\eps}+R \sqrt{N}\,\bar\zeta/\eps)\) | \(O(\sqrt{LR^2/\eps}+R \sqrt{N}\,\bar\zeta/\eps)\)<sup>[A9](#remark-A9)</sup> | `COND` | [[5]](#ref-arxiv-1910-10666), Theorem 2; Corollary 3 | [[5]](#ref-arxiv-1910-10666), Theorem 7 |
| 6 | `SUM C NS SUBG TV`<sup>[S12](#remark-S12), [S13](#remark-S13)</sup> | \(F(x_{o,i})-F^\star\le\eps_{\rm LB};\allowbreak F(x_o^T)-F^\star\le\eps_{\rm UB}\)<sup>[M4](#remark-M4)</sup> | \(\Omega(M^2 R^2 / \eps^2)\) | \(O(M^2 R^2 / \eps^2)\) | `COND` | [[4]](#ref-arxiv-2405-18031), Theorem 2 | [[4]](#ref-arxiv-2405-18031), Theorem 4 |
| 7 | `STO NC SM SFO STAT`<sup>[S3](#remark-S3)</sup> | \(\E\|\nabla F(\hat x)\|^2\le\eps_{\rm LB};\allowbreak \frac1{T_{\rm out}}\sum_{t=1}^{T_{\rm out}}\E\|\nabla F(\bar x^{(t)})\|^2\le\eps_{\rm UB}\)<sup>[M5](#remark-M5)</sup> | \(\Omega(\Delta L \sigma^2/(N \eps^2) + \Delta L/(\sqrt{\rho} \eps))\) | \(\widetilde O(\Delta L \sigma^2/(N \eps^2) + \Delta L/(\sqrt{\rho} \eps))\)<sup>[A3](#remark-A3)</sup> | `COND` | [[6]](#ref-arxiv-2210-07863), Table 6 | [[6]](#ref-arxiv-2210-07863), Theorem 4 |
| 8 | `STO NC SM SFO STAT`<sup>[S11](#remark-S11), [S14](#remark-S14)</sup> | \(\frac1{T_{\rm out}+1}\sum_{t=0}^{T_{\rm out}}\E\|\nabla F(\bar x^{(t)})\|^2\le\eps\) | \(\Omega(\sigma^2L\Delta/(N\eps^2)+(1+\log\kappa_\pi)L\Delta/((1-\beta_\pi)\eps))\)<sup>[G7](#remark-G7)</sup> | \(\widetilde O(\sigma^2L\Delta/(N\eps^2)+(1+\log\kappa_\pi)L\Delta/((1-\beta_\pi)\eps))\)<sup>[N3](#remark-N3)</sup> | `COND` | [[7]](#ref-arxiv-2312-04928), Theorem 1; Equation (11) | [[7]](#ref-arxiv-2312-04928), Equation (14) |
| 9 | `STO NC SM SFO TV`<sup>[S2](#remark-S2), [S4](#remark-S4), [S13](#remark-S13)</sup> | \(\E\|\nabla F(\hat x)\|^2\le\eps_{\rm LB};\allowbreak \frac1{T+1}\sum_{k=0}^T\E\|\nabla F(\bar x^{(k)})\|^2\le\eps_{\rm UB}\) | \(\Omega(\Delta L \sigma^2/(N \eps^2) + \Delta L/((1-\beta_{\rm H})\eps))\)<sup>[O5](#remark-O5), [A8](#remark-A8), [N8](#remark-N8)</sup> | \(\widetilde O(\Delta L \sigma^2/(N \eps^2) + \Delta L/((1-\beta_{\rm H})\eps))\)<sup>[N8](#remark-N8)</sup> | `COND` | [[8]](#ref-arxiv-2211-00533), Theorem 4 | [[8]](#ref-arxiv-2211-00533), Theorem 6 |
| 10 | `SUM PL SM SFO STAT` | \(\E[F(x_i)-F^\star] \le \eps\) | \(\Omega(K_{\rm loc}+\min\{K_{\rm loc}\kappa,\sqrt{K_{\rm loc}/N}\kappa_s\}\log(1/\eps))\)<sup>[A12](#remark-A12), [N9](#remark-N9)</sup> | \(O((K_{\rm loc}+\min\{K_{\rm loc}\kappa,\sqrt{K_{\rm loc}/N}\kappa_s\})\log(\Delta/\eps))\)<sup>[N10](#remark-N10)</sup> | `LOG` | [[9]](#ref-arxiv-2210-13931), Theorem 7; Table 4 | [[9]](#ref-arxiv-2210-13931), Corollary 2; Table 4 |
| 11 | `SUM NC SM SFO STAT` | \(\E\|\nabla F(x_i)\| \le \eps,\ \forall i\) | \(\Omega(K_{\rm loc}+\min\{K_{\rm loc}L_F,\sqrt{K_{\rm loc}/N}\,\bar L\}\Delta/\eps^2)\)<sup>[A28](#remark-A28), [N14](#remark-N14)</sup> | \(O(K_{\rm loc}+\min\{K_{\rm loc}L_F,\sqrt{K_{\rm loc}/N}\,\bar L\}\Delta/\eps^2)\)<sup>[A28](#remark-A28), [N15](#remark-N15)</sup> | `COND` | [[9]](#ref-arxiv-2210-13931), Theorem 3; Table 2; Remark 9 | [[9]](#ref-arxiv-2210-13931), Theorem 1; Corollary 1; Table 2 |
| 12 | `STO NC SM SFO STAT` | \(\E\|\nabla F(\hat x)\| \le \eps\) | \(\Omega(\Delta L\sigma^2/(N\eps^4))\)<sup>[A29](#remark-A29), [N16](#remark-N16)</sup> | \(\widetilde O(\Delta L\sigma^2/(N\eps^4)+\Delta L/(\sqrt{\rho}\eps^2))\)<sup>[N4](#remark-N4), [A2](#remark-A2)</sup> | `GAP` | [[10]](#ref-arxiv-2006-08085), Corollary 1; Appendix Setting 1 | [[10]](#ref-arxiv-2006-08085), Theorem 3 |
| 13 | `STO PL SM SFO STAT`<sup>[S3](#remark-S3)</sup> | \(\E[F(\hat x)-F^\star]\le\eps_{\rm LB};\allowbreak \frac1{T_{\rm out}}\sum_{t=1}^{T_{\rm out}}\E[F(\bar x^{(t)})-F^\star]\le\eps_{\rm UB}\)<sup>[M5](#remark-M5)</sup> | \(\Omega(\sigma^2/(\mu N \eps) + \sqrt{\kappa/\rho} \log(1/\eps))\) | \(\widetilde O(L \sigma^2/(\mu^2 N \eps) + \kappa/\sqrt{\rho} \log(1/\eps))\)<sup>[A3](#remark-A3)</sup> | `GAP` | [[6]](#ref-arxiv-2210-07863), Table 7 | [[6]](#ref-arxiv-2210-07863), Theorem 5 |
| 14 | `SUM C SM FO STAT`<sup>[S23](#remark-S23)</sup> | \(F(\bar x) - F^\star \le \eps\)<sup>[M8](#remark-M8)</sup> | \(\Omega(\sqrt{L R^2/(\rho \eps)})\)<sup>[N7](#remark-N7)</sup> | \(\widetilde O(\sqrt{LR^2/(\rho\eps)})\)<sup>[G17](#remark-G17), [N8](#remark-N8), [G18](#remark-G18)</sup> | `COND` | [[11]](#ref-arxiv-2606-04757), Theorem 2 and paragraph following the theorem | [[11]](#ref-arxiv-2606-04757), Theorem 1 |
| 15 | `SUM SC SM FO STAT` | \(\E V^t\le\eps\) | Unknown | \(O(\max(L/\mu, \|\widehat W\|/\omega(\widehat W)) \log(1/\eps))\)<sup>[A13](#remark-A13)</sup> | `LB?` | / | [[12]](#ref-arxiv-2004-02635), Appendix C, paragraph following Theorem C.1 |
| 16 | `SUM SC SM FO STAT`<sup>[S1](#remark-S1)</sup> | \(\|x_i - x^\star\|^2 \le \eps\) | Unknown | \(O(\log(1/\eps))\) | `LB?` | / | [[13]](#ref-arxiv-2308-08164), Theorem 1 |
| 17 | `SUM SC SM FO STAT` | \(\|x_i-x^\star\|^2 \le \eps\) | Unknown | \(O(\log(1/\eps))\)<sup>[A14](#remark-A14)</sup> | `LB?` | / | [[14]](#ref-arxiv-1803-09169), Theorem 2; Theorem 1 |
| 18 | `SUM SC SM FO TV`<sup>[S1](#remark-S1), [S15](#remark-S15)</sup> | \(\|X - X^\star\|^2 \le \eps\) | Unknown | \(O(\log(1/\eps))\)<sup>[G9](#remark-G9)</sup> | `LB?` | / | [[15]](#ref-arxiv-1810-07393), Theorem 1 |
| 19 | `SUM SC SM FO TV`<sup>[S24](#remark-S24)</sup> | \(\|X-X^\star\|^2 \le \eps\) | Unknown | \(O(\log(1/\eps))\)<sup>[G15](#remark-G15)</sup> | `LB?` | / | [[16]](#ref-arxiv-1611-00990), Theorem 4.1; Corollary 4.2 |
| 20 | `SUM SC SM FO TV` | \(\|X^T-X^\star\|_F^2 \le \eps\) | Unknown | \(O(\kappa \log(R^2 / \eps))\)<sup>[O6](#remark-O6)</sup> | `LB?` | / | [[17]](#ref-arxiv-1911-08527), Theorem 4.3 |
| 21 | `SUM SC SM FO TV` | \(\|\bar x_k-x^\star\|^2 \leq \eps\) | Unknown | \(O((L\tau_{\rm AC}^2/\mu)\log(1/\eps))\) | `LB?` | / | [[18]](#ref-arxiv-2309-02626), Theorem 3.1; Corollary 3.1 |
| 22 | `SUM SC SM SFO STAT`<sup>[S22](#remark-S22)</sup> | \(\E[F(x_i) - F^\star] \le \eps, \forall i\) | Unknown | \(\widetilde O(\sigma^2/(N\mu\eps)+\sqrt{L/(\mu\rho_{\rm K})}\log(\Delta_0/\eps))\)<sup>[A24](#remark-A24), [A25](#remark-A25), [G13](#remark-G13)</sup> | `LB?` | / | [[11]](#ref-arxiv-2606-04757), Theorem 5; Algorithm 4 |
| 23 | `SUM SC SM SFO STAT`<sup>[S1](#remark-S1), [S20](#remark-S20)</sup> | \(\E\|\bar x_k-x^\star\|^2\le\eps\) | Unknown | \(O(\eps^{-1})\)<sup>[G4](#remark-G4), [A19](#remark-A19)</sup> | `LB?` | / | [[19]](#ref-arxiv-2005-07785), Theorem 2; Section IV-B |
| 24 | `SUM SC SM SFO STAT` | \(\E\|z_{\rm VR}^k-x^\star\|^2 \le \eps\) | Unknown | \(O((\sqrt{K_{\rm loc}\kappa_{s,\rm VR}}+K_{\rm loc})\log(1/\eps))\) | `LB?` | / | [[20]](#ref-arxiv-2009-04373), Corollary 5 |
| 25 | `SUM SC SM SFO STAT` | \(\E[\|\hat x-x^\star\|^2] \le \eps\) | Unknown | \(O((K_{\rm loc}+\kappa^2/\rho^2)\log(1/\eps))\)<sup>[G10](#remark-G10)</sup> | `LB?` | / | [[21]](#ref-arxiv-1910-04057), Theorems 1 and 2 |
| 26 | `SUM SC SM SFO TV` | \(\|X^T-X^\star\|^2 \le \eps\) | Unknown | \(O((K_{\rm loc}+\sqrt{K_{\rm loc}\kappa_s})\log(1/\eps))\) | `LB?` | / | [[22]](#ref-arxiv-2402-02490), Corollary 4.2; Table 2 |
| 27 | `SUM SC NS FO STAT COMP PROX`<sup>[S18](#remark-S18)</sup> | \(\E[\|X^t-X^\star\|_F^2] \le \eps\) | Unknown | \(O(\max\{\kappa,p_{\rm comm}^{-2}\} \log(\Psi^0/\eps))\)<sup>[O1](#remark-O1)</sup> | `LB?` | / | [[23]](#ref-arxiv-2312-11861), Theorem 1; Equation (11); following complexity paragraph; Table I |
| 28 | `SUM SC NS FO STAT COMP PROX` | \(\|X^T-X^\star\|^2 \le \eps\) | Unknown | \(O(\log(1/\eps))\)<sup>[O1](#remark-O1), [G11](#remark-G11)</sup> | `LB?` | / | [[24]](#ref-arxiv-1905-07996), Theorem 1 |
| 29 | `SUM SC NS FO TV COMP PROX`<sup>[S29](#remark-S29)</sup> | \(\E\|\tilde x_i^T-x^\star\|^2\le\eps,\ \forall i\)<sup>[M9](#remark-M9)</sup> | Unknown | \(O(\log(1/\eps))\)<sup>[O1](#remark-O1), [A30](#remark-A30), [N17](#remark-N17)</sup> | `LB?` | / | [[25]](#ref-arxiv-2106-14075), Corollary 1; Equation (21) |
| 30 | `SUM SC NS SUBG STAT` | \(F(\hat x)-F^\star \le \eps\) | Unknown | \(O(M^2/(\mu \eps))\) | `LB?` | / | [[26]](#ref-arxiv-1904-09015), Table 3 |
| 31 | `SUM C SM FO STAT` | \(\left|\sum_{i=1}^N F_i(\bar x_{i,T})-\sum_{i=1}^N F_i(x^\star)\right|\le\eps\) | Unknown | \(O(C_{\mathbf S}C_0/(\alpha_{\rm DAGP}\eps)+C_{\mathbf S}C_0 C_2/(\eta_{\rm alg}N\eps^2))\)<sup>[O2](#remark-O2)</sup> | `LB?` | / | [[27]](#ref-arxiv-2210-03232), Theorem 2 |
| 32 | `SUM C SM FO TV`<sup>[S17](#remark-S17)</sup> | \(F(\bar x)-F^\star \le \eps\) | Unknown | \(O(\sqrt{L/\eps})\)<sup>[N2](#remark-N2), [N1](#remark-N1)</sup> | `LB?` | / | [[28]](#ref-arxiv-2104-02596), Theorem 1; Corollary 3 |
| 33 | `SUM C SM SFO STAT`<sup>[S22](#remark-S22)</sup> | \(\E[F(x_i)-F^\star] \le \eps,\ \forall i\) | \(\Omega(\sigma^2R^2/(N\eps^2))\)<sup>[N18](#remark-N18), [N19](#remark-N19)</sup> | \(\widetilde O(\sigma^2 R^2/(N\eps^2)+\sqrt{L R^2/(\rho_{\rm K}\eps)})\)<sup>[A24](#remark-A24), [G1](#remark-G1), [G14](#remark-G14), [A25](#remark-A25)</sup> | `GAP` | [[11]](#ref-arxiv-2606-04757), Paragraph following Theorem 2; see also: [[51]](#ref-arxiv-1009-0571), Theorem 1 | [[11]](#ref-arxiv-2606-04757), Theorem 6 |
| 34 | `SUM C NS FO STAT COMP PROX` | \(F(\bar y_T)-F^\star\le\eps\) | Unknown | \(O(\eps^{-1/2})\)<sup>[O1](#remark-O1), [N2](#remark-N2), [N1](#remark-N1)</sup> | `LB?` | / | [[29]](#ref-arxiv-2312-15845), Theorem 3; Corollary 4 |
| 35 | `SUM C NS SFO STAT` | \(\E[F(\hat x_i(T))-F^\star]\le\eps\) | Unknown | \(\widetilde O(M^2 R^2/(\rho \eps^2))\) | `LB?` | / | [[30]](#ref-arxiv-1005-2012), Theorem 4 |
| 36 | `SUM C NS SFO TV COMP` | \(F(\tilde x_i^T)-F^\star\le\eps\) | Unknown | \(\widetilde O(\eps^{-2})\)<sup>[G2](#remark-G2), [O4](#remark-O4), [A15](#remark-A15), [G16](#remark-G16)</sup> | `LB?` | / | [[31]](#ref-arxiv-2506-12901), Theorem 4 |
| 37 | `SUM C NS SFO TV COMP PROX` | \(\E[F(\tilde x^S)-F^\star]\le\eps\)<sup>[M10](#remark-M10)</sup> | Unknown | \(\widetilde O(K_{\rm loc}+(C_{\rm DPS}/\eps)^{1/q_{\rm DPS}})\)<sup>[O1](#remark-O1), [A31](#remark-A31), [N20](#remark-N20)</sup> | `LB?` | / | [[32]](#ref-arxiv-2112-10389), Theorems 2 and 3; Remark 1; Algorithm 1 |
| 38 | `SUM C NS SUBG STAT`<sup>[S26](#remark-S26)</sup> | \(F(\hat x_i(T))-F^\star\le\eps\) | Unknown | \(\widetilde O(N^2 M^2 R^2/\eps^2)\) | `LB?` | / | [[30]](#ref-arxiv-1005-2012), Corollary 1 |
| 39 | `SUM C NS SUBG STAT`<sup>[S27](#remark-S27)</sup> | \(F(\hat x_i(T))-F^\star\le\eps\) | Unknown | \(\widetilde O(M^2 R^2/\eps^2)\) | `LB?` | / | [[30]](#ref-arxiv-1005-2012), Corollary 1 |
| 40 | `SUM C NS SUBG STAT`<sup>[S25](#remark-S25)</sup> | \(F(\hat x_i(T))-F^\star\le\eps\) | Unknown | \(\widetilde O(N M^2 R^2/\eps^2)\) | `LB?` | / | [[30]](#ref-arxiv-1005-2012), Corollary 1 |
| 41 | `SUM C NS SUBG STAT`<sup>[S5](#remark-S5)</sup> | \(\mathcal E_{\rm obj}(\hat y)\le\eps\) | Unknown | \(O(M^2\sqrt{N\log N}/\eps^2)\)<sup>[A23](#remark-A23)</sup> | `LB?` | / | [[33]](#ref-arxiv-1411-4186), Proposition 3.12 |
| 42 | `SUM C NS SUBG STAT` | \(|F(\tilde x_i(t))-F^\star|\le\eps\) | Unknown | \(\widetilde O(\eps^{-2})\) | `LB?` | / | [[34]](#ref-arxiv-1806-07081), Theorem 5; Corollary 2(b) |
| 43 | `SUM C NS SUBG TV`<sup>[S6](#remark-S6)</sup> | \(F(\hat x_j[T])-F^\star \le \eps\) | Unknown | \(O(1/\eps^2)\) | `LB?` | / | [[35]](#ref-arxiv-1606-08904), Theorem 2 |
| 44 | `SUM NC SM FO STAT`<sup>[S8](#remark-S8)</sup> | \(\min_{0\le k<T}\|\nabla F(\bar x_u^k)\|\le\eps\) | Unknown | \(O(1/(\eta^2\eps^2))\)<sup>[A6](#remark-A6), [A7](#remark-A7)</sup> | `LB?` | / | [[36]](#ref-arxiv-2601-03566), Theorem 1 |
| 45 | `SUM NC SM FO TV` | \(\frac1T\sum_{t=0}^{T-1}\|\nabla F(\bar x^{(t)})\|^2\le\eps\) | Unknown | \(O(\eps^{-1})\) | `LB?` | / | [[37]](#ref-arxiv-2512-24483), Theorem 2; Appendix A.9 |
| 46 | `SUM NC SM FO TV`<sup>[S9](#remark-S9)</sup> | \(\min_{1\le s\le t}\left(N\|G_a(y^{s-1},\bar z^{s-1})\|_{\E}^2+\|X^{s-1}-\mathbf{y}^{s-1}\|_{\E}^2\right)\le\eps\)<sup>[M2](#remark-M2)</sup> | Unknown | \(O(C_{\rm DDA}/\eps)\) | `LB?` | / | [[38]](#ref-arxiv-2211-06914), Theorem 2; Equation (23); Algorithm 1 |
| 47 | `SUM NC SM SFO STAT` | \(\frac1T\sum_{t=0}^{T-1}\E\|\bar g(X^{(t)})\|_* \le \eps\)<sup>[M2](#remark-M2)</sup> | Unknown | \((p-1)T=O((p-1)\eps^{-(3p+1)/p})\)<sup>[A1](#remark-A1), [A20](#remark-A20)</sup> | `LB?` | / | [[39]](#ref-arxiv-2510-01377), Theorem 6; Equation (23) |
| 48 | `SUM NC SM SFO TV` | \(\E\|\nabla F(\hat x)\|^2 \le \eps^2\) | Unknown | \(O(K_{\rm loc}+\sqrt{K_{\rm loc}}L_{\rm rms,loc}\Delta/\eps^2)\)<sup>[A27](#remark-A27), [N13](#remark-N13)</sup> | `LB?` | / | [[22]](#ref-arxiv-2402-02490), Theorem 4.3; Corollary 4.5 |
| 49 | `SUM NC NS SFO STAT COMP PROX` | \(\E\|\mathbf s(\hat x)\|^2 \le \eps^2\) | Unknown | \(O((L\Delta/\eps^2)\max\{\sqrt{K_{\rm loc}/N},1\}+\max\{K_{\rm loc},\sqrt{NK_{\rm loc}}\})\)<sup>[O1](#remark-O1), [A17](#remark-A17)</sup> | `LB?` | / | [[40]](#ref-arxiv-2110-01594), Theorem 3; Table 1 |
| 50 | `SUM PL SM FO STAT` | \(F(\bar x)-F^\star \le \eps\) | Unknown | \(O(\theta^{-1}\log(c/(N\eps)))\) | `LB?` | / | [[41]](#ref-arxiv-1912-12110), Theorem 2; Equation (15) |
| 51 | `SUM PL SM FO TV` | \(F(\bar x)-F^\star \le \eps\) | Unknown | \(O(\kappa\log(\Delta/\eps))\)<sup>[G6](#remark-G6)</sup> | `LB?` | / | [[42]](#ref-arxiv-2210-03810), Theorem 3.1 |
| 52 | `STO SC SM SFO STAT`<sup>[S1](#remark-S1), [S19](#remark-S19)</sup> | \(\E\|\bar x_{\pi_{\rm SAB}}^k-x^\star\|^2\le\eps\) | Unknown | \(O(\eps^{-1})\)<sup>[G4](#remark-G4), [A18](#remark-A18)</sup> | `LB?` | / | [[43]](#ref-arxiv-2109-07981), Theorem 1 and following rate paragraph |
| 53 | `STO SC SM SFO TV` | \(\sum_{t=0}^T\frac{w_t}{W_T}\E[F(\bar x^{(t)})-F^\star]+\mu\E\|\bar x^{(T+1)}-x^\star\|^2\le\eps\) | Unknown | \(\widetilde O(\sigma^2/(\mu N\eps)+\sqrt{L}(\bar\zeta+\sigma\sqrt{\delta_{\rm net}})/(\mu\delta_{\rm net}\sqrt{\eps})+L\log(1/\eps)/(\mu\delta_{\rm net}))\)<sup>[N5](#remark-N5)</sup> | `LB?` | / | [[44]](#ref-arxiv-2003-10422), Theorem 2 |
| 54 | `STO SC NS SFO STAT` | \(F(\hat x)-F^\star \le \eps\) | Unknown | \(O((M^2+\sigma^2)/(\mu \eps))\)<sup>[G2](#remark-G2)</sup> | `LB?` | / | [[26]](#ref-arxiv-1904-09015), Table 4 |
| 55 | `STO SC NS SFO TV`<sup>[S30](#remark-S30)</sup> | \(\E[F(\bar x)-F^\star]\le\eps\) | Unknown | \(O((M_\phi+\sigma)^2/(\mu\eps))\)<sup>[A32](#remark-A32), [G19](#remark-G19)</sup> | `LB?` | / | [[45]](#ref-arxiv-2508-09029), Theorem 2; Algorithm 1; Appendix C |
| 56 | `STO C SM SFO TV` | \((1/(T+1)) \sum_{t=0}^T \E[F(\bar x^{(t)})-F^\star] \le \eps\) | Unknown | \(O(R^2(\sigma^2/(N\eps^2)+\sqrt{L}(\bar\zeta+\sigma\sqrt{\delta_{\rm net}})/(\delta_{\rm net}\eps^{3/2})+L/(\delta_{\rm net}\eps)))\)<sup>[N5](#remark-N5)</sup> | `LB?` | / | [[44]](#ref-arxiv-2003-10422), Theorem 2 |
| 57 | `STO C NS SFO STAT` | \(\Pr\{F(\hat x)-F^\star\le\eps\}\ge1-\delta_{\rm ht}\) | Unknown | \(O((M^2+\sigma^2)R^2/\eps^2)\)<sup>[A33](#remark-A33), [G20](#remark-G20), [N21](#remark-N21)</sup> | `LB?` | / | [[26]](#ref-arxiv-1904-09015), Table 4; high-probability discussion; see also: [[52]](#ref-arxiv-1406-0919), Section 4; Corollaries 3 and 4 |
| 58 | `STO C NS SFO TV` | \(\E[F(\bar x)-F^\star]\le\eps\) | Unknown | \(O((M+\sigma)^2R^2/\eps^2)\) | `LB?` | / | [[45]](#ref-arxiv-2508-09029), Theorem 3 |
| 59 | `STO NC SM SFO STAT` | \(\frac1T\sum_{t=0}^{T-1}\E\|\nabla F(\bar x^{(t)})\|^2\le3\eps^2+\alpha_{\rm DSP}\eps^3\) | Unknown | \(O(\ell_{\rm DSP}\sigma\eps^{-3}+\sigma^2\eps^{-2})\)<sup>[A21](#remark-A21)</sup> | `LB?` | / | [[46]](#ref-arxiv-1911-12665), Theorem 1; Corollary 1 |
| 60 | `STO NC SM SFO STAT` | \(\frac1T\sum_{t=1}^{T}\E\|\nabla F(\bar x_{\pi_A}^t)\|^2 \le \eps\) | \(\Omega(\sigma^2L\Delta/(N\eps^2))\)<sup>[N22](#remark-N22), [N23](#remark-N23)</sup> | \(O\!\left(\frac{\sigma^2L\Delta}{N\eps^2}+\frac{(1+\log\kappa_A+\log N)L\Delta}{(1-\beta_A)\eps}\right)\)<sup>[G12](#remark-G12)</sup> | `GAP` | [[10]](#ref-arxiv-2006-08085), Theorem 1; Appendix Setting 1 | [[47]](#ref-arxiv-2506-04600), Theorem 3 |
| 61 | `STO NC SM SFO TV`<sup>[S21](#remark-S21)</sup> | \((1/T) \sum_{t=0}^{T-1} \E[\|\nabla F(\bar x^{(t)})\|^2] \le \eps\) | Unknown | \(O((1+\sigma^2)^2/(N\eps^2)+N\zeta_{\rm stat}^2/\eps)\)<sup>[G3](#remark-G3), [A22](#remark-A22)</sup> | `LB?` | / | [[48]](#ref-arxiv-2412-07252), Theorem 2; Corollary 1 |
| 62 | `STO NC NS SFO STAT` | \(\E\|\nabla F(\hat x)\|_{\delta_{\rm G}} \le \eps\) | Unknown | \(O(\rho^{-2}\delta_{\rm G}^{-1}\eps^{-3})\) | `LB?` | / | [[49]](#ref-arxiv-2406-01484), Theorem 2; Theorem 1 |
| 63 | `SUM SC NS SFO STAT COMP PROX` | \(\E[F(\bar y_{\rm PM}^T)-F^\star]\le\eps\)<sup>[M7](#remark-M7)</sup> | Unknown | \(O\!\left((K_{\rm loc}+K_{\rm loc}^{3/4}\sqrt{\kappa_{\rm snc}})\left[1+\log(\Delta/\eps)\right]\right)\)<sup>[A26](#remark-A26), [O7](#remark-O7), [N12](#remark-N12), [O1](#remark-O1)</sup> | `LB?` | / | [[50]](#ref-arxiv-2402-02356), Theorem 5.1; Corollary 5.3 |
| 64 | `SUM SC NS SFO TV` | \(-\) | Unknown | Unknown | `Unknown` | / | / |
| 65 | `SUM C SM SFO TV` | \(-\) | Unknown | Unknown | `Unknown` | / | / |
| 66 | `SUM C NS FO TV COMP PROX`<sup>[S28](#remark-S28)</sup> | \(\E[F(\tilde y_{\rm DA}^T)-F^\star]\le\eps\)<sup>[M6](#remark-M6)</sup> | Unknown | \(O(\eps^{-1})\)<sup>[N11](#remark-N11), [O1](#remark-O1)</sup> | `LB?` | / | [[25]](#ref-arxiv-2106-14075), Corollary 2; Equation (22) |
| 67 | `SUM NC NS FO STAT` | \(-\) | Unknown | Unknown | `Unknown` | / | / |
| 68 | `SUM NC NS FO TV` | \(-\) | Unknown | Unknown | `Unknown` | / | / |
| 69 | `SUM NC NS SFO TV` | \(-\) | Unknown | Unknown | `Unknown` | / | / |
| 70 | `SUM NC NS SUBG STAT` | \(-\) | Unknown | Unknown | `Unknown` | / | / |
| 71 | `SUM NC NS SUBG TV` | \(-\) | Unknown | Unknown | `Unknown` | / | / |
| 72 | `SUM PL SM SFO TV` | \(-\) | Unknown | Unknown | `Unknown` | / | / |
| 73 | `STO NC NS SFO TV` | \(-\) | Unknown | Unknown | `Unknown` | / | / |
| 74 | `STO PL SM SFO TV` | \(-\) | Unknown | Unknown | `Unknown` | / | / |

## Remark Notes

Superscript markers next to main-table entries identify conditions that qualify a result's scope, assumptions, guarantee, target, or oracle-query accounting. `S`: Scope and topology; `O`: Oracle and side access; `A`: Additional assumptions; `G`: Guarantee regime; `M`: Metric and target qualification; `N`: Normalization and counting.

### Scope and topology

These notes identify restrictions on the communication graph, mixing model, topology family, or decentralized method class covered by the cited result.

<a id="remark-S1"></a>
**S1.** The cited result is stated for directed communication graphs rather than the unrestricted static or time-varying graph class.

<a id="remark-S2"></a>
**S2.** The cited lower bound uses the paper's sun-shaped time-varying graph construction.

<a id="remark-S3"></a>
**S3.** The cited result assumes the mixing parameter satisfies \(\beta\leq\cos(\pi/N)\).

<a id="remark-S4"></a>
**S4.** The cited result assumes the mixing parameter satisfies \(\beta\leq1-1/N\).

<a id="remark-S5"></a>
**S5.** The cited result is restricted to two-dimensional grid or geometric network topologies.

<a id="remark-S6"></a>
**S6.** The packet-drop model starts from a fixed strongly connected directed graph and requires every original communication edge to succeed at least once in every B consecutive rounds.

<a id="remark-S8"></a>
**S8.** The cited result uses separate directed row-stochastic and column-stochastic mixing matrices.

<a id="remark-S9"></a>
**S9.** The cited result is formulated for a random-network model rather than an arbitrary deterministic time-varying network.

<a id="remark-S11"></a>
**S11.** The cited lower bound is proved for a linear-span method class on directed graphs.

<a id="remark-S12"></a>
**S12.** The cited lower bound is restricted to the paper's zero-initialized linear-span or local-memory decentralized first-order method class.

<a id="remark-S13"></a>
**S13.** Every time-varying mixing operator must preserve consensus and satisfy the paper's uniform contraction bound on the disagreement subspace.

<a id="remark-S14"></a>
**S14.** The cited result uses one fixed strongly connected directed graph with a primitive column-stochastic mixing matrix.

<a id="remark-S15"></a>
**S15.** The time-varying directed graphs are jointly strongly connected over every C-step window; the row- and column-stochastic weights have self-loops and uniform positive lower bounds.

<a id="remark-S16"></a>
**S16.** The cited result assumes a fixed connected undirected network and a symmetric positive-semidefinite gossip operator with the stated consensus kernel.

<a id="remark-S17"></a>
**S17.** The cited result assumes doubly stochastic mixing and a finite window parameter for which the paper's contraction factor \(\sigma_\gamma<1\).

<a id="remark-S18"></a>
**S18.** The cited multi-gossip result uses a symmetric doubly stochastic mixing matrix on a connected static graph.

<a id="remark-S19"></a>
**S19.** The cited S-AB result uses separate row- and column-stochastic directed matrices whose associated graphs have spanning trees with intersecting root sets.

<a id="remark-S20"></a>
**S20.** The cited S-ADDOPT result assumes a fixed strongly connected directed graph and requires each node to know its out-degree.

<a id="remark-S21"></a>
**S21.** The cited Push-SUM result assumes directed communication with strong connectivity over every fixed-length window.

<a id="remark-S22"></a>
**S22.** The cited result assumes a fixed symmetric doubly-stochastic gossip matrix and uses a reference-specific gap based on the second-largest eigenvalue, rather than the global one-step contraction gap.

<a id="remark-S23"></a>
**S23.** The cited lower bound uses an \(N\)-node path with \(N\geq6\), \(\rho=\Theta(N^{-2})\), origin initialization, averaged output, and \(d=\Theta(S\sqrt\rho)\).

<a id="remark-S24"></a>
**S24.** The graph sequence is time-varying and directed; every union over B0 consecutive graphs is strongly connected, and each mixing matrix is column stochastic with the paper's out-degree weights.

<a id="remark-S25"></a>
**S25.** The cited upper bound is specialized to a two-dimensional grid topology.

<a id="remark-S26"></a>
**S26.** The cited upper bound is specialized to a cycle or path topology.

<a id="remark-S27"></a>
**S27.** The cited upper bound is specialized to an expander topology.

<a id="remark-S28"></a>
**S28.** The cited dual-averaging result uses independent-in-time random undirected doubly stochastic mixing with a uniform second-moment contraction; it is not an arbitrary deterministic time-varying-network guarantee.

<a id="remark-S29"></a>
**S29.** The source uses independent-in-time random undirected doubly stochastic matrices with \(\lambda_{\max}(\mathbb E[P_t^\top P_t]-J)\le\beta^2<1\), not arbitrary deterministic time-varying mixing.

<a id="remark-S30"></a>
**S30.** The source requires its per-time network contraction assumption, normalized here by \(\chi_{\rm TV}\); joint connectivity alone is insufficient.

### Oracle and side access

These notes record oracle primitives or exact local operations required in addition to the local first-order calls counted in the displayed complexity.

<a id="remark-O1"></a>
**O1.** The cited result assumes exact access to a proximal operator in addition to the counted first-order oracle calls.

<a id="remark-O2"></a>
**O2.** The cited result requires exact local orthogonal projections onto the node constraint sets in addition to the counted gradients.

<a id="remark-O3"></a>
**O3.** The oracle model allows exact gradients of the local Fenchel conjugates in addition to ordinary local gradients; MSDA uses this conjugate-function gradient primitive.

<a id="remark-O4"></a>
**O4.** The cited result requires a mirror-map operation in addition to the counted stochastic first-order oracle calls.

<a id="remark-O5"></a>
**O5.** The cited lower bound restricts algorithms to the paper's partial-averaging communication model.

<a id="remark-O6"></a>
**O6.** The cited result relies on a finite consensus routine whose accuracy depends on the target accuracy.

<a id="remark-O7"></a>
**O7.** The cited PMGT-KatyushaX analysis is retained with a common initial point and an exactly consensual tracker equal to the global smooth-part gradient. This explicitly interprets the source's ambiguous composite-gradient initialization. Initialization costs \(K_{\rm loc}\) component calls per node plus uncounted aggregation communication; ordinary unequal local-gradient initialization is not certified here.

### Additional assumptions

These notes list hypotheses that are stronger or more specific than the canonical row tags alone, including regularity, heterogeneity, initialization, and algorithmic restrictions.

<a id="remark-A1"></a>
**A1.** The cited result assumes order-\(p\) smoothness, which is stronger than ordinary first-order smoothness when \(p>1\).

<a id="remark-A2"></a>
**A2.** The cited upper bound assumes the paper's weak close-distribution condition, which bounds cross-node gradient dissimilarity only at the common initial point \(x=0\).

<a id="remark-A3"></a>
**A3.** The cited upper bound assumes a uniform bound on gradient dissimilarity across nodes.

<a id="remark-A6"></a>
**A6.** The cited result uses the paper's generalized-smoothness condition.

<a id="remark-A7"></a>
**A7.** The cited result assumes relative dissimilarity among local objectives.

<a id="remark-A8"></a>
**A8.** The cited lower bound is restricted to zero-respecting algorithms.

<a id="remark-A9"></a>
**A9.** The cited result requires \(\nu=O(R/(\sqrt N\,\bar\zeta))\) to attain the displayed rate.

<a id="remark-A10"></a>
**A10.** The cited nonsmooth result uses local regularity: every local objective is convex and has its own finite Lipschitz constant.

<a id="remark-A11"></a>
**A11.** The cited nonsmooth lower bound is stated in the sufficiently high-dimensional regime required by its hard-instance construction.

<a id="remark-A12"></a>
**A12.** The cited lower bound holds only in its stated accuracy and smoothness-parameter regime, including the explicit separation between the mean and aggregate smoothness constants.

<a id="remark-A13"></a>
**A13.** The cited primal-dual linear-rate result assumes a fixed connected undirected graph, smooth strongly convex local functions, exact local gradients, and the prescribed initialization and stepsize conditions.

<a id="remark-A14"></a>
**A14.** The cited FROST result assumes a strongly connected static digraph, smooth strongly convex local objectives, and uncoordinated stepsizes in its admissible range.

<a id="remark-A15"></a>
**A15.** The cited Sub-Weibull result additionally assumes bounded local subgradients, Lipschitz regularizers, Sub-Weibull noise, strong graph connectivity, and convexity of the Bregman divergence in its second argument.

<a id="remark-A17"></a>
**A17.** The cited ProxGT-SR-E result assumes mean-squared smooth component gradients, unbiased sampling with bounded variance, consensual initialization, the stated spectral mixing condition, and exact prox.

<a id="remark-A18"></a>
**A18.** The cited S-AB asymptotic result assumes second-order growth at the optimum and differentiable stochastic gradients with the stated higher-moment and random-Lipschitz bounds.

<a id="remark-A19"></a>
**A19.** The cited S-ADDOPT result assumes conditionally unbiased stochastic gradients with uniformly bounded conditional variance.

<a id="remark-A20"></a>
**A20.** The cited DeMuon result assumes consensual initialization, bounded matrix noise, a primitive doubly stochastic mixer, coupled extrapolation and momentum parameters, horizon-dependent stepsizes, and the theorem's minimum iteration threshold.

<a id="remark-A21"></a>
**A21.** The cited D-SPIDER-SFO result assumes smooth local objectives, unbiased bounded-variance stochastic gradients, the stated spectral mixing condition, consensual initialization, and the prescribed batch and stepsize rules.

<a id="remark-A22"></a>
**A22.** The cited Push-SUM result assumes bounded stochastic-gradient variance and a uniform global bound on cross-node gradient dissimilarity.

<a id="remark-A23"></a>
**A23.** The optimization consequence treats its initial-distance factor as bounded and requires every node to know a common size estimate \(U\) satisfying \(U\geq N\) and \(U=O(N)\).

<a id="remark-A24"></a>
**A24.** The objective is a deterministic node sum; stochasticity enters only through the local oracle.

<a id="remark-A25"></a>
**A25.** Heterogeneity is bounded only at the optimum, not uniformly for all x.

<a id="remark-A26"></a>
**A26.** The cited sum-of-nonconvex result assumes a strongly convex total composite objective, a convex smooth aggregate, and component curvature bounds defining \(\kappa_{\rm snc}\). Individual components need not be convex. The audited subcase requires a strictly positive strong-convexity modulus for the smooth aggregate because the proof divides by that modulus.

<a id="remark-A27"></a>
**A27.** The cited GT-PAGE analysis assumes the component mean-squared smoothness inequality separately at every node, with bound \(L_{\rm rms,loc}\); a network-averaged component bound alone is insufficient. The display follows the network-average objective used in the nonconvex analysis, despite the source introduction's sum notation.

<a id="remark-A28"></a>
**A28.** Here \(L_F\) is the smoothness of the global objective \(F\), while \(\bar L\) is component mean-squared smoothness. The cited DEAREST result also assumes a connected undirected static network with a symmetric doubly stochastic positive-semidefinite mixing matrix and common initialization.

<a id="remark-A29"></a>
**A29.** The retained Lu--De Sa component uses the paper's zero-respecting/linear-span class, fresh independent unbiased variance-bounded stochastic gradients, \(\sigma>0\), and the stated sufficiently small-\(\eps\) regime.

<a id="remark-A30"></a>
**A30.** Row 29 assumes a shared closed convex proximable \(h\), exact local gradients and prox steps, common initialization, and a fixed \(a\in(0,1/\mu)\) satisfying source restrictions (16) and (18). The exact prefactor is \(C_{29}=(2/a)(2C/\mu+D)\).

<a id="remark-A31"></a>
**A31.** The cited DPSVRG theorem assumes componentwise smooth finite sums plus a shared proximable \(h\), exact prox, time-varying \(b\)-connected mixing, bounded estimator/local-gradient/subgradient errors, and source parameters independent of \(\eps\). Its full error constant is \(C_{\rm DPS}=(1+r_{\rm DPS}/(2n_{0,{\rm DPS}}))\Delta+\|x^0-x^\star\|^2/[n_{0,{\rm DPS}}(2\alpha_{\rm DPS}-8\alpha_{\rm DPS}^2L)]+C_{{\rm err},{\rm DPS}}/[n_{0,{\rm DPS}}(r_{\rm DPS}\beta_{\rm DPS}-1)]\).

<a id="remark-A32"></a>
**A32.** For row 55, \(F_i(x)=\phi_i(x)+\mu\|x\|^2/2\). The bound \(M_\phi\) applies only to subgradients of convex \(\phi_i\); its oracle is conditionally unbiased with variance at most \(\sigma^2\), while the quadratic term is known and evaluated exactly.

<a id="remark-A33"></a>
**A33.** Row 57 uses a common compact convex Euclidean domain of diameter at most \(R\), exact projections, convex \(M\)-Lipschitz local objectives, and conditionally unbiased worker-independent noise satisfying \(\mathbb E[\exp(\|g_i-v_i\|^2/\sigma^2)\mid\mathcal F]\le e\).

### Guarantee regime

These notes explain the probabilistic, initialization, stepsize, finite-time, or asymptotic regime in which the displayed guarantee holds.

<a id="remark-G1"></a>
**G1.** The cited result provides the displayed guarantee only from a prescribed warm start.

<a id="remark-G2"></a>
**G2.** The displayed complexity is a high-probability guarantee rather than an expectation guarantee.

<a id="remark-G3"></a>
**G3.** The displayed rate is the sufficiently-large-T simplification in Corollary 1; Theorem 2 contains additional finite-time initial-error, network, variance, and heterogeneity terms.

<a id="remark-G4"></a>
**G4.** The cited result uses a diminishing stepsize schedule.

<a id="remark-G6"></a>
**G6.** The displayed rate assumes an exact oracle (zero oracle bias); the consensus tolerance is chosen with the target accuracy so the additive error floor is at most the target, affecting communication rounds but not the displayed per-node gradient-query order.

<a id="remark-G7"></a>
**G7.** The cited directed lower bound requires \(\beta_\pi\) to be bounded away from zero, \(\beta_\pi\leq1-1/N\), and \(\log\kappa_\pi=\Omega(N(1-\beta_\pi))\); it is not uniform over all parameter tuples.

<a id="remark-G9"></a>
**G9.** The displayed logarithmic iteration complexity suppresses a contraction factor whose constants depend on graph connectivity, mixing weights, objective regularity, and the admissible stepsize.

<a id="remark-G10"></a>
**G10.** The cited GT-SVRG result requires the prescribed stepsize, inner-loop length, and initialization conditions.

<a id="remark-G11"></a>
**G11.** The displayed \(\epsilon\)-only logarithmic order treats the cited result's contraction factor as fixed; otherwise the iteration count is proportional to \(\log(1/\epsilon)/\log(1/\gamma)\).

<a id="remark-G12"></a>
**G12.** The cited MG-PullDiag-GT result requires the specified stepsize, minibatch, inner-loop, and directed-mixing parameter choices.

<a id="remark-G13"></a>
**G13.** The first restart epoch starts in consensus; later epochs use warm-started outputs.

<a id="remark-G14"></a>
**G14.** The cited warm-start result requires its two explicit RMS disagreement inequalities.

<a id="remark-G15"></a>
**G15.** Each node uses a positive time-invariant but node-dependent stepsize. The stepsize maximum and ratio must satisfy Theorem 4.1 or Corollary 4.2; the displayed logarithmic oracle order treats the induced contraction factor \(\lambda\in(0,1)\) as fixed, and otherwise carries a factor \(1/\log(1/\lambda)\).

<a id="remark-G16"></a>
**G16.** The displayed Sub-Weibull complexity suppresses confidence- and tail-dependent polylogarithmic factors, including powers of \(\log(T/\delta_{\rm ht})\) governed by the tail index, where \(\delta_{\rm ht}\) is the failure probability.

<a id="remark-G17"></a>
**G17.** The row-14 upper bound specializes the cited stochastic theorem to \(\sigma=0\), uses its integer budget schedule, and hides the theorem's logarithmic factor.

<a id="remark-G18"></a>
**G18.** The matching order is conditional on the source nonblocking coupled clock and on the path/linear-span lower-bound class; it is not an unrestricted oracle-query LOG claim.

<a id="remark-G19"></a>
**G19.** The full achievable expression contains \(1+\chi_{\rm TV}M_\phi/\sqrt{\mu\eps}+(M_\phi+\sigma)^2/(\mu\eps)\). The main cell records its fixed-network, sufficiently small-\(\eps\) leading oracle order, with integer-rounding constants suppressed.

<a id="remark-G20"></a>
**G20.** The displayed row-57 order fixes the graph and failure probability and records the leading small-\(\eps\) term. A conservative full form is \(\widetilde O(1+(M^2+\sigma^2)R^2/\eps^2+MR\sqrt{\chi_L}/\eps)\), with confidence logarithms suppressed.

### Metric and target qualification

These notes clarify how the cited result's controlled metric or merit target differs from, or is related to, the measure displayed in the table.

<a id="remark-M1"></a>
**M1.** The cited result controls a Bregman merit function rather than the ordinary objective gap or Euclidean distance.

<a id="remark-M2"></a>
**M2.** Only the stationarity component of the cited result is retained in the displayed metric; its other residual terms are omitted.

<a id="remark-M3"></a>
**M3.** The displayed metric follows from the equivalence between the strongly convex smooth metric in the source and the table measure.

<a id="remark-M4"></a>
**M4.** The cited lower bound uses a node-local output selected from local memory, whereas the cited upper bound uses the global average output; only the oracle-order expressions are compared.

<a id="remark-M5"></a>
**M5.** The cited lower bound permits an arbitrary algorithm output, whereas the cited upper bound controls the iteration-average metric over network-average iterates; only the resulting oracle orders are compared.

<a id="remark-M6"></a>
**M6.** The cited convex dual-averaging result controls the time average of auxiliary prox points formed from averaged dual accumulators, not the local returned iterates or their arithmetic average. Forming this virtual output requires aggregation and prox side work beyond the counted local gradients.

<a id="remark-M7"></a>
**M7.** The cited PMGT-KatyushaX guarantee concerns the objective gap at the network-average primal output, not each individual node's output.

<a id="remark-M8"></a>
**M8.** The source upper bound controls every node's objective gap; convexity then gives the displayed bound for \(F(\bar x)\).

<a id="remark-M9"></a>
**M9.** The guarantee is for every node's weighted output \(\tilde x_i^T=A_T^{-1}\sum_{t=1}^T a_t x_i^t\), with \(a_t=a(1-a\mu)^{-t}\), rather than an unspecified final iterate.

<a id="remark-M10"></a>
**M10.** The source controls the virtual centralized epoch output \(\tilde x^S\), not every node's returned local iterate.

### Normalization and counting

These notes explain how iteration, batch, and sample counts in cited results are converted into per-node oracle queries and which communication work is excluded.

<a id="remark-N1"></a>
**N1.** The displayed complexity counts local gradient evaluations and does not count the additional consensus rounds.

<a id="remark-N2"></a>
**N2.** Each outer iteration performs multiple gossip or consensus rounds, which are not counted as oracle queries in the displayed rate.

<a id="remark-N3"></a>
**N3.** The displayed count \(N_{\rm SFO}=T_{\rm out}R\) includes the \(R\) samples per outer iteration; each outer iteration also performs two \(R\)-round gossip phases, counted separately as communication.

<a id="remark-N4"></a>
**N4.** The displayed stochastic-oracle complexity is normalized to local batch size \(B=1\).

<a id="remark-N5"></a>
**N5.** The displayed effective per-iteration network rate normalizes the cited result's block-contraction parameter by its mixing-window length.

<a id="remark-N7"></a>
**N7.** The displayed per-node epsilon form is derived from the communication-round bound in its nonconstant-error branch.

<a id="remark-N8"></a>
**N8.** The displayed network-dependent lower-bound term is an oracle projection of a coupled round model in which each node performs one local query and one gossip step per round; it is not an unconditional communication-free oracle lower bound.

<a id="remark-N9"></a>
**N9.** The displayed lower bound divides an aggregate LIFO lower bound by \(N\); it applies to average and worst-node workload, not every node individually. The full computation-round lower bound additionally contains \(L\Delta/\epsilon^2\) in the nonconvex row and \(\kappa\log(1/\epsilon)\) in the PL row. These serial-depth terms are excluded from the oracle count because partial participation permits different nodes to carry different rounds.

<a id="remark-N10"></a>
**N10.** The displayed upper bound divides expected aggregate LIFO calls by the node count under uniform node-component sampling; it is an expected per-node workload bound rather than an expected-maximum workload bound.

<a id="remark-N11"></a>
**N11.** The cited dual-averaging algorithm uses one initial local gradient and at most one fresh local gradient per iteration. The displayed epsilon-only order holds for a fixed admissible stepsize, network, smoothness and initialization; their full coefficient is retained in the evidence ledger.

<a id="remark-N12"></a>
**N12.** The cited PMGT-KatyushaX rate counts expected component-gradient evaluations per node, including full-gradient snapshots, both evaluations in sampled gradient differences and initialization. The geometric inner-loop length and batch size of order \(\sqrt{K_{\rm loc}}\) give an expected epoch cost of order \(K_{\rm loc}\); multi-consensus and exact prox work are separate.

<a id="remark-N13"></a>
**N13.** The cited GT-PAGE count is expected work at each node, including the initial full local gradient and stochastic refresh/difference calls; it is not an expected maximum over nodes or a deterministic stopping budget.

<a id="remark-N14"></a>
**N14.** The row-11 lower bound is for the cited IFO/LIFO span class in its small-\(\eps\) regime. Dividing aggregate LIFO work by \(N\) proves an average or some worst-node requirement, not the same lower bound at every node; the separate computation-round term is excluded.

<a id="remark-N15"></a>
**N15.** Uniform node-component sampling makes the displayed DEAREST+ order the expected local query count at every node; communication rounds are counted separately.

<a id="remark-N16"></a>
**N16.** With \(B=1\), only the Setting-1 sample/noise component is normalized from total work to a uniform-cap or worst-node per-node query count. Corollary 1's spectral branch counts communication rounds and is not promoted to \(N_{\rm SFO}\).

<a id="remark-N17"></a>
**N17.** The \(O(\log(1/\eps))\) display fixes \(a,\mu,C_{29}\) and the network constants and includes initialization; it does not assert an explicit unrestricted graph dependence.

<a id="remark-N18"></a>
**N18.** Only the classical stochastic-oracle term is promoted. With total sample budget \(N_{\rm tot}=NQ\), the minimax error \(\Omega(\sigma R/\sqrt{N_{\rm tot}})\) implies \(Q=\Omega(\sigma^2R^2/(N\eps^2))\). It remains valid with free gossip and gives a uniform-cap or worst-node requirement, not the same burden at every node; it is nontrivial only for \(\sigma>0\).

<a id="remark-N19"></a>
**N19.** The path/network lower-bound term is not entered because converting its communication rounds to oracle calls uses the source's one-query/one-gossip coupled clock and deterministic path linear-span class.

<a id="remark-N20"></a>
**N20.** The source count is \(K_{\rm loc}S+2\sum_{s=1}^S\lceil n_{0,{\rm DPS}}\beta_{\rm DPS}^s\rceil\); the display hides epoch logarithms and excludes multi-consensus and exact-prox work.

<a id="remark-N21"></a>
**N21.** One stacked stochastic-oracle evaluation in the cited decentralized method means one query at each node; the per-worker count is therefore not divided by \(N\).

<a id="remark-N22"></a>
**N22.** The Lu--De Sa Setting-1 hard instance yields the squared-stationarity noise term after total-query/\(N\) normalization under a uniform cap or worst-node count. It permits free communication but assumes fresh unbiased bounded-variance queries, a zero-respecting/linear-span class, \(\sigma>0\), and sufficiently small \(\eps\).

<a id="remark-N23"></a>
**N23.** The separate \(\beta_A,\kappa_A\) term from the row-60 reference is not promoted: its argument counts a coupled communication/iteration clock rather than unrestricted per-node stochastic-oracle queries.

## Summary

Across 74 rows, this table presents 47 of the 96 vanilla settings and omits the remaining 49 unknown settings. Evidence covers 37 of the 47 presented settings (37 of 96 in the full grid), while 10 presented settings are explicitly `Unknown`. The table retains 16 lower-bound and 64 upper-bound claim sides (80 total), citing 52 distinct arXiv papers; 12 papers are cited on lower-bound sides and 51 on upper-bound sides, including supporting provenance and with overlap between the two paper sets. Row statuses are `EXACT` 2; `COND` 9; `LOG` 1; `GAP` 4; `UB?` 0; `LB?` 48; `Unknown` 10; status totals count rows and paper totals are deduplicated by arXiv ID.

## References

<a id="ref-arxiv-2106-04469"></a>
[1] [Lower Bounds and Optimal Algorithms for Smooth and Strongly Convex Decentralized Optimization Over Time-Varying Networks](https://arxiv.org/abs/2106.04469) [[arXiv]](https://arxiv.org/abs/2106.04469)
Kovalev, Gasanov, Gasnikov, and Richtarik. NeurIPS, 2021.

<a id="ref-arxiv-1806-00291"></a>
[2] [Optimal Algorithms for Non-Smooth Distributed Optimization in Networks](https://arxiv.org/abs/1806.00291) [[arXiv]](https://arxiv.org/abs/1806.00291)
Scaman, Bach, Bubeck, Lee, and Massoulie. arXiv, 2018.

<a id="ref-arxiv-1702-08704"></a>
[3] [Optimal Algorithms for Smooth and Strongly Convex Distributed Optimization in Networks](https://proceedings.mlr.press/v70/scaman17a.html) [[arXiv]](https://arxiv.org/abs/1702.08704)
Scaman, Bach, Bubeck, Lee, and Massoulie. ICML, 2017.

<a id="ref-arxiv-2405-18031"></a>
[4] [Lower Bounds and Optimal Algorithms for Non-Smooth Convex Decentralized Optimization over Time-Varying Networks](https://arxiv.org/abs/2405.18031) [[arXiv]](https://arxiv.org/abs/2405.18031)
Dmitry Kovalev, Ekaterina Borodich, Alexander Gasnikov, and Dmitrii Feoktistov. arXiv, 2024.

<a id="ref-arxiv-1910-10666"></a>
[5] [Accelerated Primal-Dual Algorithms for Distributed Smooth Convex Optimization over Networks](https://arxiv.org/abs/1910.10666) [[arXiv]](https://arxiv.org/abs/1910.10666)
Xu, Tian, Sun, and Scutari. AISTATS, 2020.

<a id="ref-arxiv-2210-07863"></a>
[6] [Revisiting Optimal Convergence Rate for Smooth and Non-convex Stochastic Decentralized Optimization](https://arxiv.org/abs/2210.07863) [[arXiv]](https://arxiv.org/abs/2210.07863)
Yuan et al. NeurIPS, 2022.

<a id="ref-arxiv-2312-04928"></a>
[7] [Understanding the Influence of Digraphs on Decentralized Optimization: Effective Metrics, Lower Bound, and Optimal Algorithm](https://arxiv.org/abs/2312.04928) [[arXiv]](https://arxiv.org/abs/2312.04928)
Liyuan Liang, Xinmeng Huang, Ran Xin, and Kun Yuan. arXiv, 2023.

<a id="ref-arxiv-2211-00533"></a>
[8] [Lower and Upper Bounds for Nonconvex Decentralized Optimization over Time-varying Networks](https://arxiv.org/abs/2211.00533) [[arXiv]](https://arxiv.org/abs/2211.00533)
Huang et al. arXiv, 2022.

<a id="ref-arxiv-2210-13931"></a>
[9] [On the Complexity of Decentralized Smooth Nonconvex Finite-Sum Optimization](https://arxiv.org/abs/2210.13931) [[arXiv]](https://arxiv.org/abs/2210.13931)
Luo Luo, Yunyan Bai, Lesi Chen, Yuxing Liu, and Haishan Ye. arXiv, 2022.

<a id="ref-arxiv-2006-08085"></a>
[10] [Optimal Complexity in Decentralized Training](https://arxiv.org/abs/2006.08085) [[arXiv]](https://arxiv.org/abs/2006.08085)
Lu, Tsaknakis, Hong, and Chen. ICML, 2021.

<a id="ref-arxiv-2606-04757"></a>
[11] [Near-Optimal Decentralized Stochastic Convex Optimization over Networks](https://arxiv.org/abs/2606.04757) [[arXiv]](https://arxiv.org/abs/2606.04757)
Kluger, Attia, and Koren. arXiv, 2026.

<a id="ref-arxiv-2004-02635"></a>
[12] [Dualize, Split, Randomize: Toward Fast Nonsmooth Optimization Algorithms](https://arxiv.org/abs/2004.02635) [[arXiv]](https://arxiv.org/abs/2004.02635)
Adil Salim, Laurent Condat, Konstantin Mishchenko, and Peter Richtárik. arXiv, 2020.

<a id="ref-arxiv-2308-08164"></a>
[13] [Privacy-Preserving Push-Pull Method for Decentralized Optimization via State Decomposition](https://arxiv.org/abs/2308.08164) [[arXiv]](https://arxiv.org/abs/2308.08164)
Huqiang Cheng, Xiaofeng Liao, Huaqing Li, and You Zhao. arXiv, 2023.

<a id="ref-arxiv-1803-09169"></a>
[14] [FROST -- Fast row-stochastic optimization with uncoordinated step-sizes](https://arxiv.org/abs/1803.09169) [[arXiv]](https://arxiv.org/abs/1803.09169)
Ran Xin, Chenguang Xi, and Usman A. Khan. arXiv, 2018.

<a id="ref-arxiv-1810-07393"></a>
[15] [Optimization over time-varying directed graphs with row and column-stochastic matrices](https://arxiv.org/abs/1810.07393) [[arXiv]](https://arxiv.org/abs/1810.07393)
Fakhteh Saadatniaki, Ran Xin, and Usman A. Khan. arXiv, 2018.

<a id="ref-arxiv-1611-00990"></a>
[16] [Geometrical Convergence Rate for Distributed Optimization with Time-Varying Directed Graphs and Uncoordinated Step-Sizes](https://arxiv.org/abs/1611.00990) [[arXiv]](https://arxiv.org/abs/1611.00990)
Qingguo Lü and Huaqing Li. arXiv, 2016.

<a id="ref-arxiv-1911-08527"></a>
[17] [Projected Gradient Method for Decentralized Optimization over Time-Varying Networks](https://arxiv.org/abs/1911.08527) [[arXiv]](https://arxiv.org/abs/1911.08527)
Alexander Rogozin and Alexander Gasnikov. arXiv, 2019.

<a id="ref-arxiv-2309-02626"></a>
[18] [Adaptive Consensus: A network pruning approach for decentralized optimization](https://arxiv.org/abs/2309.02626) [[arXiv]](https://arxiv.org/abs/2309.02626)
Suhail M. Shah, Albert S. Berahas, and Raghu Bollapragada. arXiv, 2023.

<a id="ref-arxiv-2005-07785"></a>
[19] [S-ADDOPT: Decentralized stochastic first-order optimization over directed graphs](https://arxiv.org/abs/2005.07785) [[arXiv]](https://arxiv.org/abs/2005.07785)
Muhammad I. Qureshi, Ran Xin, Soummya Kar, and Usman A. Khan. arXiv, 2020.

<a id="ref-arxiv-2009-04373"></a>
[20] [Variance Reduced EXTRA and DIGing and Their Optimal Acceleration for Strongly Convex Decentralized Optimization](https://arxiv.org/abs/2009.04373) [[arXiv]](https://arxiv.org/abs/2009.04373)
Huan Li, Zhouchen Lin, and Yongchun Fang. arXiv, 2020.

<a id="ref-arxiv-1910-04057"></a>
[21] [Variance-Reduced Decentralized Stochastic Optimization with Gradient Tracking -- Part II: GT-SVRG](https://arxiv.org/abs/1910.04057) [[arXiv]](https://arxiv.org/abs/1910.04057)
Ran Xin, Usman A. Khan, and Soummya Kar. arXiv, 2019.

<a id="ref-arxiv-2402-02490"></a>
[22] [Decentralized Finite-Sum Optimization over Time-Varying Networks](https://arxiv.org/abs/2402.02490) [[arXiv]](https://arxiv.org/abs/2402.02490)
Dmitry Metelev, Savelii Chezhegov, Alexander Rogozin, Aleksandr Beznosikov, Alexander Sholokhov, Alexander Gasnikov, and Dmitry Kovalev. arXiv, 2024.

<a id="ref-arxiv-2312-11861"></a>
[23] [A Proximal Gradient Method With Probabilistic Multi-Gossip Communications for Decentralized Composite Optimization](https://arxiv.org/abs/2312.11861) [[arXiv]](https://arxiv.org/abs/2312.11861)
Luyao Guo, Luqing Wang, Xinli Shi, and Jinde Cao. arXiv, 2023.

<a id="ref-arxiv-1905-07996"></a>
[24] [A Linearly Convergent Proximal Gradient Algorithm for Decentralized Optimization](https://arxiv.org/abs/1905.07996) [[arXiv]](https://arxiv.org/abs/1905.07996)
Sulaiman A. Alghunaim, Kun Yuan, and Ali H. Sayed. arXiv, 2019.

<a id="ref-arxiv-2106-14075"></a>
[25] [Decentralized Composite Optimization in Stochastic Networks: A Dual Averaging Approach with Linear Convergence](https://arxiv.org/abs/2106.14075) [[arXiv]](https://arxiv.org/abs/2106.14075)
Changxin Liu, Zirui Zhou, Jian Pei, Yong Zhang, and Yang Shi. arXiv, 2021.

<a id="ref-arxiv-1904-09015"></a>
[26] [Decentralized and Parallel Primal and Dual Accelerated Methods for Stochastic Convex Programming Problems](https://arxiv.org/abs/1904.09015) [[arXiv]](https://arxiv.org/abs/1904.09015)
Darina Dvinskikh and Alexander Gasnikov. arXiv, 2019.

<a id="ref-arxiv-2210-03232"></a>
[27] [Double Averaging and Gradient Projection: Convergence Guarantees for Decentralized Constrained Optimization](https://arxiv.org/abs/2210.03232) [[arXiv]](https://arxiv.org/abs/2210.03232)
Firooz Shahriari-Mehr and Ashkan Panahi. arXiv, 2022.

<a id="ref-arxiv-2104-02596"></a>
[28] [Accelerated Gradient Tracking over Time-varying Graphs for Decentralized Optimization](https://arxiv.org/abs/2104.02596) [[arXiv]](https://arxiv.org/abs/2104.02596)
Li and Lin. arXiv, 2021.

<a id="ref-arxiv-2312-15845"></a>
[29] [Optimal Decentralized Composite Optimization for Convex Functions](https://arxiv.org/abs/2312.15845) [[arXiv]](https://arxiv.org/abs/2312.15845)
Haishan Ye and Xiangyu Chang. arXiv, 2023.

<a id="ref-arxiv-1005-2012"></a>
[30] [Dual Averaging for Distributed Optimization](https://arxiv.org/abs/1005.2012) [[arXiv]](https://arxiv.org/abs/1005.2012)
Duchi, Agarwal, and Wainwright. IEEE Transactions on Automatic Control, 2012.

<a id="ref-arxiv-2506-12901"></a>
[31] [High-Probability Convergence Theory for Distributed Composite Optimization with Sub-Weibull Noises](https://arxiv.org/abs/2506.12901) [[arXiv]](https://arxiv.org/abs/2506.12901)
Zhan Yu, Zhongjie Shi, and Deming Yuan. arXiv, 2025.

<a id="ref-arxiv-2112-10389"></a>
[32] [Decentralized Stochastic Proximal Gradient Descent with Variance Reduction over Time-varying Networks](https://arxiv.org/abs/2112.10389) [[arXiv]](https://arxiv.org/abs/2112.10389)
Xuanjie Li, Yuedong Xu, Jessie Hui Wang, Xin Wang, and John C. S. Lui. arXiv, 2021.

<a id="ref-arxiv-1411-4186"></a>
[33] [Linear Time Average Consensus on Fixed Graphs and Implications for Decentralized Optimization and Multi-Agent Control](https://arxiv.org/abs/1411.4186) [[arXiv]](https://arxiv.org/abs/1411.4186)
Alex Olshevsky. arXiv, 2014.

<a id="ref-arxiv-1806-07081"></a>
[34] [Distributed Optimization over Directed Graphs with Row Stochasticity and Constraint Regularity](https://arxiv.org/abs/1806.07081) [[arXiv]](https://arxiv.org/abs/1806.07081)
Van Sy Mai and Eyad H. Abed. arXiv, 2018.

<a id="ref-arxiv-1606-08904"></a>
[35] [On the Convergence Rate of Average Consensus and Distributed Optimization over Unreliable Networks](https://arxiv.org/abs/1606.08904) [[arXiv]](https://arxiv.org/abs/1606.08904)
Lili Su. arXiv, 2016.

<a id="ref-arxiv-2601-03566"></a>
[36] [Provably Convergent Decentralized Optimization over Directed Graphs under Generalized Smoothness](https://arxiv.org/abs/2601.03566) [[arXiv]](https://arxiv.org/abs/2601.03566)
Yanan Bo and Yongqiang Wang. arXiv, 2026.

<a id="ref-arxiv-2512-24483"></a>
[37] [Decentralized Optimization over Time-Varying Row-Stochastic Digraphs](https://arxiv.org/abs/2512.24483) [[arXiv]](https://arxiv.org/abs/2512.24483)
Liyuan Liang, Yilong Song, and Kun Yuan. arXiv, 2025.

<a id="ref-arxiv-2211-06914"></a>
[38] [Rate analysis of dual averaging for nonconvex distributed optimization](https://arxiv.org/abs/2211.06914) [[arXiv]](https://arxiv.org/abs/2211.06914)
Changxin Liu, Xuyang Wu, Xinlei Yi, Yang Shi, and Karl H. Johansson. arXiv, 2022.

<a id="ref-arxiv-2510-01377"></a>
[39] [DeMuon: A Decentralized Muon for Matrix Optimization over Graphs](https://arxiv.org/abs/2510.01377) [[arXiv]](https://arxiv.org/abs/2510.01377)
Chuan He, Shuyi Ren, Jingwei Mao, and Erik G. Larsson. arXiv, 2025.

<a id="ref-arxiv-2110-01594"></a>
[40] [A Stochastic Proximal Gradient Framework for Decentralized Non-Convex Composite Optimization: Topology-Independent Sample Complexity and Communication Efficiency](https://arxiv.org/abs/2110.01594) [[arXiv]](https://arxiv.org/abs/2110.01594)
Ran Xin, Subhro Das, Usman A. Khan, and Soummya Kar. arXiv, 2021.

<a id="ref-arxiv-1912-12110"></a>
[41] [Linear Convergence of First- and Zeroth-Order Primal-Dual Algorithms for Distributed Nonconvex Optimization](https://arxiv.org/abs/1912.12110) [[arXiv]](https://arxiv.org/abs/1912.12110)
Xinlei Yi, Shengjun Zhang, Tao Yang, Tianyou Chai, and Karl H. Johansson. arXiv, 2020.

<a id="ref-arxiv-2210-03810"></a>
[42] [Gradient-Type Methods For Decentralized Optimization Problems With Polyak-Lojasiewicz Condition Over Time-Varying Networks](https://arxiv.org/abs/2210.03810) [[arXiv]](https://arxiv.org/abs/2210.03810)
Ilya Kuruzov, Mohammad Alkousa, Fedor Stonyakin, and Alexander Gasnikov. arXiv, 2022.

<a id="ref-arxiv-2109-07981"></a>
[43] [Asymptotic Properties of S-AB Method with Diminishing Stepsize](https://arxiv.org/abs/2109.07981) [[arXiv]](https://arxiv.org/abs/2109.07981)
Shengchao Zhao and Yongchao Liu. arXiv, 2021.

<a id="ref-arxiv-2003-10422"></a>
[44] [A Unified Theory of Decentralized SGD with Changing Topology and Local Updates](https://arxiv.org/abs/2003.10422) [[arXiv]](https://arxiv.org/abs/2003.10422)
Anastasia Koloskova, Nicolas Loizou, Sadra Boreiri, Martin Jaggi, and Sebastian U. Stich. arXiv, 2020.

<a id="ref-arxiv-2508-09029"></a>
[45] [Stochastic Decentralized Optimization of Non-Smooth Convex and Convex-Concave Problems over Time-Varying Networks](https://arxiv.org/abs/2508.09029) [[arXiv]](https://arxiv.org/abs/2508.09029)
Maxim Divilkovskiy and Alexander Gasnikov. arXiv, 2025.

<a id="ref-arxiv-1911-12665"></a>
[46] [D-SPIDER-SFO: A Decentralized Optimization Algorithm with Faster Convergence Rate for Nonconvex Problems](https://arxiv.org/abs/1911.12665) [[arXiv]](https://arxiv.org/abs/1911.12665)
Taoxing Pan, Jun Liu, and Jie Wang. arXiv, 2019.

<a id="ref-arxiv-2506-04600"></a>
[47] [Achieving Linear Speedup and Near-Optimal Complexity for Decentralized Optimization over Row-stochastic Networks](https://arxiv.org/abs/2506.04600) [[arXiv]](https://arxiv.org/abs/2506.04600)
Liyuan Liang, Xinyi Chen, Gan Luo, and Kun Yuan. arXiv, 2025.

<a id="ref-arxiv-2412-07252"></a>
[48] [Adaptive Weighting Push-SUM for Decentralized Optimization with Statistical Diversity](https://arxiv.org/abs/2412.07252) [[arXiv]](https://arxiv.org/abs/2412.07252)
Yiming Zhou, Yifei Cheng, Linli Xu, and Enhong Chen. arXiv, 2024.

<a id="ref-arxiv-2406-01484"></a>
[49] [Online Optimization Perspective on First-Order and Zero-Order Decentralized Nonsmooth Nonconvex Stochastic Optimization](https://arxiv.org/abs/2406.01484) [[arXiv]](https://arxiv.org/abs/2406.01484)
Emre Sahinoglu and Shahin Shahrampour. arXiv, 2024.

<a id="ref-arxiv-2402-02356"></a>
[50] [Decentralized Sum-of-Nonconvex Optimization](https://arxiv.org/abs/2402.02356) [[arXiv]](https://arxiv.org/abs/2402.02356)
Zhuanghua Liu and Bryan Kian Hsiang Low. arXiv, 2024.

<a id="ref-arxiv-1009-0571"></a>
[51] [Information-theoretic lower bounds on the oracle complexity of convex optimization](https://arxiv.org/abs/1009.0571) [[arXiv]](https://arxiv.org/abs/1009.0571)
Alekh Agarwal, Peter L. Bartlett, Pradeep Ravikumar, and Martin J. Wainwright. IEEE Transactions on Information Theory, 2012.

<a id="ref-arxiv-1406-0919"></a>
[52] [Gradient Sliding for Composite Optimization](https://arxiv.org/abs/1406.0919) [[arXiv]](https://arxiv.org/abs/1406.0919)
Guanghui Lan. Mathematical Programming, 2016.
