
## Optimization Methods for Network Structure Inference

One important class of optimization problems arising in network data analysis concerns learning the structure of the network that captures the relationships among entities in the data. Learning this structure can substantially improve many downstream tasks such as clustering, semi-supervised learning, recommendation systems, anomaly detection, and influence propagation analysis.

In this file, I present several optimization frameworks for learning different types of network structures from data. The choice of optimization framework depends on the prior knowledge available about the network. For example, if we know that the underlying network is undirected, we employ formulations that explicitly infer symmetric or undirected structures. Such assumptions act as inductive biases that are injected into the network structure learning problem to guide the optimization toward meaningful and interpretable solutions.

### Learning a directed network 
For the sake of argument, suppose that the network structure follows a Structural Equation Model (SEM). In this setting, $\mathbf{A}$ denotes the $K \times K$ network structure matrix, where $K$ is the number of entities (nodes) in the network. Moreover, $\mathbf{Y}$ represents the endogenous variables, while $\mathbf{X}$ represents the exogenous inputs of the SEM. Both $\mathbf{Y}$ and $\mathbf{X}$ have dimensions $K \times d$, where $d$ denotes the number of observations collected from the network. In all the following formulas, there is one condition for the optimizations,  

```math
\mathcal{A}_m
=
\left\{
A \in \mathbb{R}^{m \times m}
\;:\;
A_{ij} \in \{-1,+1,0\},
\operatorname{diag}(A)=0
\right\}
```


* The following formulation indicates that we apriori know that the network is directed and the nodes with higher signal values impose the direction of the network edges on their neighboring nodes with smaller signal values. This property is imposed on the learning algorithm via the following term  $[\cdot]^+ = \max(\cdot, 0)$. The optimization has a convex but non-smooth form. There are famous solutions such as ISTA and FISTA for it. The CVXPY implementation of the optimization problem is provided in the corresponding Jupyter Notebook.

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



