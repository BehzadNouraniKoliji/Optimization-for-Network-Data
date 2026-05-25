
## Optimization Methods for Network Structure Inference

One important class of optimization problems arising in network data analysis concerns learning the structure of the network that captures the relationships among entities in the data. Learning this structure can substantially improve many downstream tasks such as clustering, semi-supervised learning, recommendation systems, anomaly detection, and influence propagation analysis.

In this file, I present several optimization frameworks for learning different types of network structures from data. The choice of optimization framework depends on the prior knowledge available about the network. For example, if we know that the underlying network is undirected, we employ formulations that explicitly infer symmetric or undirected structures. Such assumptions act as inductive biases that are injected into the network structure learning problem to guide the optimization toward meaningful and interpretable solutions.

### Learning a network structure
For the sake of argument, suppose that the network structure follows a Structural Equation Model (SEM). In this setting, $\mathbf{A}$ denotes the $K \times K$ network structure matrix, where $K$ is the number of entities (nodes) in the network. Moreover, $\mathbf{Y}$ represents the endogenous variables, while $\mathbf{X}$ represents the exogenous inputs of the SEM. Both $\mathbf{Y}$ and $\mathbf{X}$ have dimensions $K \times d$, where $d$ denotes the number of observations collected from the network. In all the following formulas, there is one condition for the optimizations,  

```math
A_m
=
\left\{
A \in \mathbb{R}_{-,+,0}^{m \times m}
\;:\;
\mathrm{diag}(A)=0
\right\}.
```


* The following formulation indicates that we apriori know that the network is directed and the nodes with lower signal values impose the direction of the network edges on their neighboring nodes with higher signal values. This property is imposed on the learning algorithm via the following term  $[\cdot]^+ = \max(\cdot, 0)$. The optimization has a convex but non-smooth form. There are famous solutions such as ISTA and FISTA for it. The CVXPY implementation of the optimization problem is provided in the corresponding Jupyter Notebook. $\gamma$ is found via cross-validation.

```math
\hat{\mathbf{A}}_{\mathrm{Di\text{-}TV}}
=
\arg\min_{\mathbf{A}}
\left\|
\mathbf{Y} - \mathbf{A}\mathbf{Y} - \mathbf{X}
\right\|_2^2
+
\gamma
\sum_{i,j}
|A_{i,j}|
\sum_k
\left[Y_{ik} - Y_{jk}\right]^+
```
* The following formulation assumes that the network is **undirected**, and the regularization term penalizes the presence of an edge $(i,j)$ proportionally to the absolute signal difference between nodes $i$ and $j$, without imposing any directional preference. This is reflected in the use of the absolute value $|\cdot|$ rather than the one-sided operator $[\cdot]^+ = \max(\cdot, 0)$, making the penalty symmetric with respect to the two endpoints of each edge. The optimization retains a **convex but non-smooth** form, and standard proximal algorithms such as ISTA and FISTA remain directly applicable. The CVXPY implementation of the optimization problem is provided in the corresponding Jupyter Notebook. $\gamma$ is found via cross-validation.

```math
\hat{\mathbf{A}}_{\mathrm{TV}}
=
\arg\min_{\mathbf{A}}
\left\|
\mathbf{Y} - \mathbf{A}\mathbf{Y} - \mathbf{X}
\right\|_2^2
+
\gamma
\sum_{i,j}
|A_{i,j}|
\sum_k
|Y_{ik} - Y_{jk}|
```
* The following formulation assumes **no prior knowledge** on the structure of the network. The regularization term penalizes the entries of $\mathbf{A}$ uniformly via an $\ell_1$ norm, promoting sparsity in the estimated connectivity matrix without encoding any preference for directionality or edge weighting based on signal values. This corresponds to the classical **Lasso** regularization, where all edges are penalized equally regardless of the signal difference between nodes. The optimization retains a **convex but non-smooth** form, and standard proximal algorithms such as ISTA and FISTA remain directly applicable. The CVXPY implementation of the optimization problem is provided in the corresponding Jupyter Notebook. $\gamma$ is found via cross-validation.

```math
\hat{\mathbf{A}}_{\mathrm{Lasso}}
=
\arg\min_{\mathbf{A}}
\left\|
\mathbf{Y} - \mathbf{A}\mathbf{Y} - \mathbf{X}
\right\|_2^2
+
\gamma
\|\mathbf{A}\|_1
```
* The following formulation assumes **no prior knowledge** on the structure of the network. The regularization term penalizes the entries of $\mathbf{A}$ uniformly via an $\ell_2$ norm, encouraging small but generally **non-sparse** connectivity weights without encoding any preference for directionality or edge weighting based on signal values. This corresponds to the classical **Ridge** regularization, where all edges are shrunk smoothly toward zero rather than being driven to exact sparsity. Crucially, unlike the Lasso, Di-TV, and TV formulations, the optimization is **convex and smooth**, admitting a **closed-form solution** via the normal equations, and proximal algorithms are not required. The CVXPY implementation of the optimization problem is provided in the corresponding Jupyter Notebook. $\gamma$ is found via cross-validation.

```math
\hat{\mathbf{A}}_{\mathrm{Ridge}}
=
\arg\min_{\mathbf{A}}
\left\|
\mathbf{Y} - \mathbf{A}\mathbf{Y} - \mathbf{X}
\right\|_2^2
+
\gamma
\|\mathbf{A}\|_2
```



