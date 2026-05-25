
## Optimization Methods for Network Structure Inference

One important class of optimization problems arising in network data analysis concerns learning the structure of the network that captures the relationships among entities in the data. Learning this structure can substantially improve many downstream tasks such as clustering, semi-supervised learning, recommendation systems, anomaly detection, and influence propagation analysis.

In this file, I present several optimization frameworks for learning different types of network structures from data. The choice of optimization framework depends on the prior knowledge available about the network. For example, if we know that the underlying network is undirected, we employ formulations that explicitly infer symmetric or undirected structures. Such assumptions act as inductive biases that are injected into the network structure learning problem to guide the optimization toward meaningful and interpretable solutions.

### Learning a directed network 

```math
\hat{\mathbf{A}}_{\mathrm{Di\text{-}TV}}
=
\arg\min_{\mathbf{A}}
\left\|
\mathbf{X} - \mathbf{A}\mathbf{X}
\right\|_2^2
+
\gamma
\sum_{i,j}
|A_{i,j}|
\sum_k
\left[X_{ik} - X_{jk}\right]^+
```

```math
\hat{\mathbf{A}}_{\mathrm{TV}}
=
\arg\min_{\mathbf{A}}
\left\|
\mathbf{X} - \mathbf{A}\mathbf{X}
\right\|_2^2
+
\gamma
\sum_{i,j}
|A_{i,j}|
\sum_k
|X_{ik} - X_{jk}|
```

```math
\hat{\mathbf{A}}_{\mathrm{L1Norm}}
=
\arg\min_{\mathbf{A}}
\left\|
\mathbf{X} - \mathbf{A}\mathbf{X}
\right\|_2^2
+
\gamma
\|\mathbf{A}\|_1
```


