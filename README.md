# 2D_Poisson_FDM_FVM
Solution of 2D Poisson’s equation with Finite Difference Method (FDM) and Finite Volume Method (FVM). The following boundary-value problem is considered:

```math
\begin{equation}
    \begin{dcases} 
        -\nabla \cdot (k \nabla u) = f, \ \ \ \ \ (x, y) \in \Omega = (0, 10) \times (0, 5) \\
        u(x, y)=0, \ \ \ \ \ (x, y) \in \partial \Omega \\
    \end{dcases}
\end{equation}
```

where $`k`$ is a coefficient function and the source function is the following:

```math
\begin{equation}
  f(x,y) = \sum_{i=1}^9 \sum_{j=1}^4 e^{-40(x-i)^2-40(y-j)^2}
\end{equation}
```

For the FDM, only a homogeneous coefficient function is considered, i.e. $`k(x,y)=1`$ whereas, for the FVM, a homogeneous and a non-homogeneous functions, defined as $`k(x,y)=1+0.1(x+y+xy)`$, are considered.

