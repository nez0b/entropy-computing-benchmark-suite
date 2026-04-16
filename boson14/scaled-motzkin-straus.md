# Scaled Motzkin-Straus Formulation for Boson14 Hardware

## 1. Standard Motzkin-Straus Recap

The Motzkin-Straus theorem relates the **maximum clique number** of a graph to a continuous quadratic program on the probability simplex:

$$\max \quad f(x) = \tfrac{1}{2}\, x^\top A\, x \qquad \text{s.t.} \quad \sum_i x_i = 1,\; x \ge 0$$

where $A$ is the adjacency matrix (0/1, zero diagonal).

**Optimal value:** $f^*(\omega) = \tfrac{1}{2}(1 - 1/\omega)$

**Inversion:** $\omega = \operatorname{round}\!\bigl(1 / (1 - 2f^*)\bigr)$

**Optimal solution:** $x_i = 1/\omega$ for $i$ in a maximum clique, $x_i = 0$ otherwise.

## 2. Scaled Simplex Derivation

The standard simplex constraint $\sum x_i = 1$ produces fractional allocations ($x_i = 1/\omega$). Hardware works with larger integers, so we scale by $R$:

**Substitution:** $y = R\,x$, so $\sum y_i = R$.

The scaled objective becomes:
$$g(y) = \tfrac{1}{2}\, y^\top A\, y = R^2 \cdot f(x)$$

**Scaled optimal value:**
$$g^*(\omega) = \frac{R^2}{2}\Bigl(1 - \frac{1}{\omega}\Bigr)$$

**Omega recovery:**
$$\omega = \operatorname{round}\!\Bigl(\frac{R^2}{R^2 - 2\,g^*}\Bigr)$$

**Scaled optimal solution:** $y_i = R/\omega$ for $i$ in a maximum clique, $y_i = 0$ otherwise.

## 3. Integer Coupling Matrix

The adjacency matrix $A$ has entries in $\{0, 1\}$. The hardware coupling matrix is:

$$J = -A$$

which has entries in $\{0, -1\}$ -- **all integers**. No scaling by $\frac{1}{2}$ is needed in the coupling matrix itself; the factor of $\frac{1}{2}$ is absorbed into the interpretation (the objective-to-omega mapping formula).

This is a key advantage over the standard formulation where $J = -\frac{1}{2}A$ would have non-integer entries.

## 4. Hardware Energy Mapping

The hardware minimises the quadratic form:

$$E(y) = y^\top J\, y = -y^\top A\, y = -2\,g(y)$$

At the optimum:
$$E^* = -R^2\Bigl(1 - \frac{1}{\omega}\Bigr)$$

**Omega from energy:**
$$\omega = \operatorname{round}\!\Bigl(\frac{R^2}{R^2 + E^*}\Bigr)$$

## 5. Setting R

Any positive integer $R$ works. The choice affects:

- **Resolution:** Larger $R$ gives finer granularity in the allocation $y_i = R/\omega$. For $R=100$ and $\omega=10$, each clique vertex gets $y_i = 10$.
- **Energy scale:** $|E^*| = R^2(1 - 1/\omega)$. Larger $R$ amplifies the energy gap between different clique sizes.
- **Hardware compatibility:** The boson sampler sum constraint is set to $R$.

**Default:** $R = 100$ for early Boson14 experiments.

## 6. Worked Example

Parameters: $R = 100$, $\omega = 10$ (planted 10-clique)

| Quantity | Formula | Value |
|----------|---------|-------|
| $f^*$ | $\frac{1}{2}(1 - 1/10)$ | $0.45$ |
| $g^*$ | $\frac{100^2}{2}(1 - 1/10)$ | $4500$ |
| $E^*$ | $-100^2(1 - 1/10)$ | $-9000$ |
| $y_i$ (clique) | $100/10$ | $10$ |
| $y_i$ (non-clique) | $0$ | $0$ |

**Verification:**
- $\omega = \operatorname{round}(10000 / (10000 - 9000)) = \operatorname{round}(10) = 10$ ✓
- $\omega = \operatorname{round}(10000 / (10000 + (-9000))) = \operatorname{round}(10) = 10$ ✓
- $\sum y_i = 10 \times 10 = 100 = R$ ✓
