# Bomze's Regularization of the Motzkin-Straus Formulation

## 1. Classical Motzkin-Straus (1965)

For a graph $G = (V, E)$ with $|V| = n$ and binary adjacency matrix $A$, the classical result states:

$$1 - \frac{1}{\omega(G)} = \max_{x \in \Delta} \; x^T A \, x$$

where $\Delta = \{x \in \mathbb{R}^n : x \geq 0, \; \sum_i x_i = 1\}$ is the standard simplex and $\omega(G)$ is the clique number (size of the maximum clique).

If $S$ is a maximum clique of size $\omega$, the **characteristic vector** of $S$:

$$x^S_i = \begin{cases} 1/\omega & \text{if } i \in S \\ 0 & \text{otherwise} \end{cases}$$

is a global maximizer, and achieves the value:

$$(x^S)^T A \, x^S = \binom{\omega}{2} \cdot \frac{2}{\omega^2} = \frac{\omega - 1}{\omega} = 1 - \frac{1}{\omega}$$

This is elegant: it converts the combinatorial max-clique problem into continuous optimization over a simplex.

---

## 2. The Spurious Solution Problem

The classical formulation has a fundamental flaw: **the global maximizers are not isolated, and many of them do not correspond to cliques.**

### How spurious solutions arise

**Mechanism 1 — Convex envelope of overlapping cliques:**
Suppose $S_1$ and $S_2$ are two distinct maximum cliques of the same size $\omega$. Their characteristic vectors $x^{S_1}$ and $x^{S_2}$ are both global maximizers. Now consider any point $\bar{x}$ on the line segment between them — not necessarily a convex combination in the usual sense, but any feasible point near this segment. Because the objective $x^T A x$ is quadratic (not linear), $\bar{x}$ is not automatically optimal. However, the structure of overlapping cliques creates flat regions in the objective landscape where non-clique solutions achieve the same optimum.

Concretely, if $S_1 \cap S_2 \neq \emptyset$, one can construct solutions supported on $S_1 \cup S_2$ — which is **not** a clique — that still achieve the global maximum value $1 - 1/\omega$.

**Mechanism 2 — Weight spreading to dominated vertices:**
Even with a single maximum clique $S$, if a vertex $v \notin S$ is adjacent to exactly $\omega - 1$ vertices in $S$, there exist global maximizers that place nonzero weight on $v$. The vertex $v$ is "almost" in the clique, and the continuous relaxation cannot distinguish it from a true clique member at the boundary.

### Why this matters

When you solve the continuous optimization and look at the support $\{i : x^*_i > 0\}$, you cannot reliably extract the maximum clique. The support might include extra vertices, miss some clique vertices, or correspond to no clique at all. You need additional combinatorial post-processing to find the actual clique, which partially defeats the purpose of the continuous formulation.

---

## 3. Bomze's Regularization (1997)

Bomze's fix is remarkably simple. Replace the adjacency matrix $A$ with:

$$\bar{A} = A + \tfrac{1}{2} I$$

and solve:

$$\max_{x \in \Delta} \; x^T \bar{A} \, x = \max_{x \in \Delta} \; \left[ x^T A \, x + \tfrac{1}{2} \|x\|^2 \right]$$

### The key theorem

**Theorem (Bomze, 1997):** Let $G$ be a graph with clique number $\omega(G)$. Then:

**(a)** The global maximum of $x^T \bar{A} \, x$ over $\Delta$ is:

$$1 - \frac{1}{2\,\omega(G)}$$

**(b)** $x^*$ is a **global** maximizer if and only if $x^*$ is the characteristic vector of a **maximum** clique.

**(c)** $x^*$ is a **local** maximizer if and only if $x^*$ is the characteristic vector of a **maximal** clique (not necessarily maximum).

**(d)** There are **no other** critical points that are local maximizers. Every local maximizer is isolated and corresponds to exactly one maximal clique.

### Verification of the optimal value

For a maximum clique $S$ of size $\omega$, the characteristic vector $x^S$ (uniform $1/\omega$ on $S$, zero elsewhere) gives:

$$
(x^S)^T \bar{A} \, x^S = (x^S)^T A \, x^S + \tfrac{1}{2} \|x^S\|^2
= \left(1 - \frac{1}{\omega}\right) + \frac{1}{2} \cdot \frac{1}{\omega}
= 1 - \frac{1}{2\omega}
$$

since $\|x^S\|^2 = \omega \cdot (1/\omega)^2 = 1/\omega$.

---

## 4. Why the Regularization Works

### Intuitive explanation

The added term $\frac{1}{2}\|x\|^2$ is a **concentration incentive**. On the simplex (where $\sum x_i = 1$), the squared norm $\|x\|^2 = \sum x_i^2$ is:

- **Maximized** by putting all weight on a single vertex: $\|e_i\|^2 = 1$
- **Minimized** by spreading weight uniformly: $\|(1/n, \ldots, 1/n)\|^2 = 1/n$

So the regularization penalizes spreading weight across many vertices and rewards concentration. Combined with the adjacency term $x^T A x$ which rewards placing weight on mutually adjacent vertices:

- $x^T A x$ alone: "spread weight over vertices with many mutual edges" → cliques are good, but spreading across overlapping cliques is equally good (spurious solutions).
- $\frac{1}{2}\|x\|^2$ alone: "concentrate weight on few vertices" → single vertices are optimal (trivial).
- $x^T A x + \frac{1}{2}\|x\|^2$ together: "concentrate weight on the largest group of mutually adjacent vertices" → exactly the maximum clique, no spurious solutions.

### Technical explanation

The spurious solutions in the original formulation exist because the Hessian of $x^T A x$ restricted to the face of the simplex corresponding to a clique is **singular** — the objective is locally flat, allowing perturbations that move weight to non-clique vertices without changing the objective value.

Adding $\frac{1}{2}\|x\|^2$ makes the restricted Hessian **positive definite** on each clique face. This means:

1. Each critical point becomes a strict local maximum (within its face of the simplex).
2. The characteristic vector of each maximal clique becomes the **unique** local maximum on its corresponding face.
3. Perturbations that spread weight outside the clique now strictly decrease the objective.

The coefficient $\frac{1}{2}$ is the **exact threshold** that achieves this. Any $\alpha \in (0, \frac{1}{2})$ in $A + \alpha I$ still has some spurious solutions. Any $\alpha > \frac{1}{2}$ works but $\alpha = \frac{1}{2}$ is minimal and preserves the tightest relationship between the optimal value and $\omega(G)$.

---

## 5. The Landscape of Local Maxima

One of the most useful consequences of Bomze's result is the **complete characterization** of the optimization landscape:

| Critical point type | Corresponds to | Count |
|---|---|---|
| Global maximizer | Maximum clique (characteristic vector) | One per maximum clique |
| Local maximizer | Maximal clique (characteristic vector) | One per maximal clique |
| Saddle / other | Non-maximal clique or non-clique | — |

This means:

- If your solver finds any local maximum, you get a maximal clique for free — just read off the support.
- If your solver finds the global maximum, you get a maximum clique.
- There is a **discrete set** of local maxima, each isolated, with a clear combinatorial interpretation.
- The objective value at each local maximum directly tells you the clique size: if $x^T \bar{A} x = 1 - \frac{1}{2k}$, the corresponding clique has size $k$.

---

## 6. Extension to the Vertex-Weighted Case

For the vertex-weighted maximum clique problem with weights $w_i > 0$, the weight of a clique $S$ is $W(S) = \sum_{i \in S} w_i$, and we want to maximize this.

### Weighted Motzkin-Straus (Gibbons et al., 1997)

Define the weighted simplex:

$$\Delta_w = \left\{ x \in \mathbb{R}^n : x \geq 0, \; \sum_i \frac{x_i}{w_i} = 1 \right\}$$

Then:

$$1 - \frac{1}{\Omega(G)} = \max_{x \in \Delta_w} \; x^T A \, x$$

where $\Omega(G) = \max_{S \text{ clique}} W(S)$ is the maximum weight clique value.

The characteristic vector of a maximum-weight clique $S$ is:

$$x^S_i = \begin{cases} w_i / W(S) \cdot w_i = w_i^2 / W(S) & \text{... wait} \end{cases}$$

Actually, the optimizer on $\Delta_w$ for clique $S$ is:

$$x^S_i = \begin{cases} w_i / W(S) \cdot w_i \end{cases}$$

Let's be precise. With the substitution $y_i = x_i / w_i$, the constraint becomes $\sum y_i = 1$ (standard simplex), and:

$$x^T A x = \sum_{(i,j) \in E} 2 x_i x_j = \sum_{(i,j) \in E} 2 w_i w_j y_i y_j = y^T (W A W) y$$

where $W = \text{diag}(w_1, \ldots, w_n)$.

So the problem is equivalent to:

$$\max_{y \in \Delta} \; y^T (W A W) \, y$$

### Weighted Bomze regularization

Applying Bomze's regularization to the transformed problem:

$$\max_{y \in \Delta} \; y^T \left( W A W + \tfrac{1}{2} W^2 \right) y = \max_{y \in \Delta} \; y^T W \left( A + \tfrac{1}{2} I \right) W \, y$$

Or equivalently, back in the original $x$ coordinates:

$$\max_{x \in \Delta_w} \; x^T \left( A + \tfrac{1}{2} I \right) x$$

**The regularization is the same:** replace $A$ with $A + \frac{1}{2}I$, regardless of whether the problem is weighted or unweighted.

The global maximum becomes $1 - \frac{1}{2\,\Omega(G)}$, and the same correspondence holds: global maximizers are characteristic vectors of maximum-weight cliques, local maximizers are maximal-weight cliques.

---

## 7. Implications for Your QUBO Pipeline

### The full chain

$$\text{QUBO} \xrightarrow{\text{Encoding 2a}} \text{Vertex-weighted graph } G' \xrightarrow{\text{Bomze}} \max_{x \in \Delta_w} x^T (A' + \tfrac{1}{2}I) x$$

where $G'$ is the conflict graph with $2n + m$ vertices and adjacency matrix $A'$.

### What your solver needs to handle

1. **The matrix is $A' + \frac{1}{2}I$**, not $A'$. The diagonal entries are $\frac{1}{2}$ (instead of 0 in a standard adjacency matrix). If your solver takes an adjacency matrix as input and internally applies the Bomze regularization, you just pass $A'$. If it takes a general matrix, pass $A' + \frac{1}{2}I$.

2. **Vertex weights may be negative** (from negating QUBO coefficients). The standard Gibbons formulation assumes $w_i > 0$. You may need to shift weights as discussed previously, and verify that your solver's convergence guarantees hold for the shifted problem.

3. **Solution extraction is clean.** At the optimum $x^*$, the support directly gives you the clique vertices. Read off the variable assignments from which $v_i^0$ or $v_i^1$ vertices are in the support. The gadget vertices $u_{ij}$ in the support confirm that the corresponding interaction terms are active.

### Computational cost of the Bomze formulation

The continuous optimization over $\Delta_w$ (or $\Delta$ after transformation) has:

- **Dimension** = $|V'| = 2n + m$
- **Matrix entries** in $\bar{A}' = A' + \frac{1}{2}I$: up to $(2n+m)^2$, but $A'$ is structured (it's the adjacency matrix of the conflict graph, which is fairly dense)

For a sparse QUBO ($m = O(n)$): optimization is over an $O(n)$-dimensional simplex with an $O(n) \times O(n)$ matrix. Very tractable.

For a dense QUBO ($m = O(n^2)$): optimization is over an $O(n^2)$-dimensional simplex with an $O(n^2) \times O(n^2)$ matrix. The matrix has $O(n^4)$ entries. This is where the encoding overhead bites.

---

## 8. References

- Motzkin, T.S. and Straus, E.G. (1965). "Maxima for graphs and a new proof of a theorem of Turán." *Canadian Journal of Mathematics*, 17, 533–540.
- Bomze, I.M. (1997). "Evolution towards the maximum clique." *Journal of Global Optimization*, 10, 143–164.
- Gibbons, L.E., Hearn, D.W., Pardalos, P.M., and Ramana, M.V. (1997). "Continuous characterizations of the maximum clique problem." *Mathematics of Operations Research*, 22(3), 754–768.
- Bomze, I.M., Budinich, M., Pardalos, P.M., and Pelillo, M. (1999). "The Maximum Clique Problem." In: *Handbook of Combinatorial Optimization*, Vol. 4, Kluwer.
- Pelillo, M. and Jagota, A. (1995). "Feasible and infeasible maxima in a quadratic program for maximum clique." *Journal of Artificial Neural Networks*, 2, 411–420.
