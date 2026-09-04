# Derivation: IFC-aware effective-potential model fitting

Companion to `ifc_fitting_derivation.py` (run it; every equation below is
asserted there and every quoted line is its verbatim output). Story 1 of
`specs/ifc-aware-model-fitting` (memnotes vault).

## D1 — Term energy and conventions

One compiled term, single cell origin $o$:

$$
T_o = \alpha \, S(\eta) \prod_{f=1}^{F} d_f^{\,p_f}, \qquad
d_f = U_{A_f} - U_{B_f}, \qquad
S(\eta) = \prod_t \eta_{v_t}^{\,q_t},
$$

with flat atom-major coordinates $U_{3a+\mu}$; the term energy sums the same
structure over the `ncell` origins (each origin has its own index pair).
$\alpha$ is the coefficient scale of the basis column (in the fitter,
`value * weight`); every formula below is linear in $\alpha$.

## D2 — Product-rule per-term IFC features

For flat coordinates $I, J$ touched by the factors, with
$s_X(f) = +1$ if $X = A_f$, $-1$ if $X = B_f$, else $0$:

$$
\frac{\partial^2 T_o}{\partial U_I \partial U_J}
= \underbrace{\sum_{f} \alpha\, S\, p_f(p_f{-}1)\, d_f^{\,p_f-2}
   \prod_{h \neq f} d_h^{\,p_h}\; s_I(f)\, s_J(f)}_{f = g}
+\; \underbrace{\sum_{f \neq g} \alpha\, S\, p_f p_g\,
   d_f^{\,p_f-1} d_g^{\,p_g-1} \prod_{h \neq f,g} d_h^{\,p_h}\;
   s_I(f)\, s_J(g)}_{\text{ordered pairs}} .
$$

The ordered-pair sum writes both $(f,g)$ and $(g,f)$ (symmetric total); a
single factor's own cross terms appear once ($f{=}g$). Strain powers enter
only through the scalar prefactor $S(\eta)$ — the displacement Hessian of a
strain-decorated term is $S(\eta)$ times the Hessian of its displacement
part. Verified for 3 factors including two sharing an atom, with strain
powers $\eta_1^1 \eta_2^2$, against `Matrix.hessian`, and the origin sum:

```
D2 PASS: scatter-formula Hessian == Matrix.hessian(T, U) (3 factors incl. shared atom, strain powers eta1^1 eta2^2).
D2 PASS: Hessian of origin sum == sum of per-origin Hessians.
```

## D3 — Coefficient linearity and unit-coefficient columns

$$
E = \sum_j c_j E_j
\;\Longrightarrow\;
\frac{\partial^2 E}{\partial \mathbf{u}^2}
= \sum_j c_j \frac{\partial^2 E_j}{\partial \mathbf{u}^2}
= \sum_j c_j X_j,
\qquad
K(\mathbf{c}) = K^{\mathrm{fixed}} + \sum_j c_j X_j .
$$

```
D3 PASS: hessian(c1 E1 + c2 E2) = c1 hessian(E1) + c2 hessian(E2);
   hence X_j = d2 E_j / du^2 (unit-coefficient column) and
   K(c) = K_fixed + sum_j c_j X_j exactly.
```

## D4 — Objective, normal equations, greedy quantities

With targets $k$, flattened residual $r_k = \mathrm{vec}(K^{\mathrm{ref}}_k
- K^{\mathrm{fixed}}_k)$, per-coefficient columns $x_{kj} =
\mathrm{vec}(X_{kj})$, entry count $m_k = (3N_k)^2$, active-target count
$n_{\mathrm{act}}$, global factor $\lambda$, per-target weights $w_k$:

$$
J_{\mathrm{ifc}} = \frac{\lambda}{n_{\mathrm{act}}}
\sum_k \frac{w_k}{m_k} \bigl\lVert r_k - X_k \mathbf{c} \bigr\rVert_F^2 .
$$

Stationarity gives the channel-factor increments used by
`_normal_equations` (goal/solver lockstep):

$$
\frac{\partial J}{\partial \mathbf{c}} = 2(\text{normal}\,\mathbf{c} -
\text{rhs}), \qquad
\text{normal} = \frac{\lambda}{n_{\mathrm{act}}} \sum_k \frac{w_k}{m_k}
X_k^{T} X_k, \qquad
\text{rhs} = \frac{\lambda}{n_{\mathrm{act}}} \sum_k \frac{w_k}{m_k}
X_k^{T} r_k .
$$

Greedy single-column quantities (factor $f_k = \lambda w_k / (n_{\mathrm{act}}
m_k)$ folded in, matching `_greedy_rhs_diagonal_target`):
$\text{rhs}_j = \sum_k f_k\, x_{kj} \cdot r_k$,
$\text{diag}_j = \sum_k f_k \lVert x_{kj} \rVert^2$,
$\text{target\_norm} = \sum_k f_k \lVert r_k \rVert^2$, and with ridge
$\rho$:

$$
\min_{c_j} \Bigl[ \sum_k f_k \lVert r_k - c_j x_{kj} \rVert^2 + \rho\,
c_j^2 \Bigr] = \text{target\_norm} -
\frac{\text{rhs}_j^2}{\text{diag}_j + \rho},
\qquad c_j^{*} = \frac{\text{rhs}_j}{\text{diag}_j + \rho}.
$$

```
D4 PASS: dJ/dc = 2(normal*c - rhs) identically; stationarity is
   normal * c = rhs with normal, rhs as the channel-factor increments
D4 PASS: greedy score identity  min_c [f||r-xc||^2 + rho c^2] =
   target_norm - rhs^2/(diag+ridge)  (closed form exact).
```

## D5 — Unit conversion

CODATA-2014: $1\,\mathrm{Ha} = 27.21138602\;\mathrm{eV}$,
$1\,\mathrm{Bohr} = 0.52917721067\;\mathrm{\AA}$, so

$$
1\;\frac{\mathrm{Ha}}{\mathrm{Bohr}^2} =
\frac{27.21138602}{(0.52917721067)^2}\;\frac{\mathrm{eV}}{\mathrm{\AA}^2}.
$$

```
D5 PASS: Ha/Bohr^2 -> eV/Ang^2 factor = 97.17362357083667
   equals HARTREE_TO_EV/BOHR_TO_ANGSTROM^2 = 97.1736236254206 (rel 5.62e-10).
```

(The $5.6\times10^{-10}$ relative difference is the round-off of the two
published CODATA-2014 values vs the module constants.)

## D6 — Numeric spot-checks

```
D6 PASS: formula == direct hessian numerically (max 0.0e+00);
D6 PASS: central-FD probe matches (max abs 9.9e-12, scale 6.6e-04).

ALL DERIVATION CHECKS PASSED.
```
