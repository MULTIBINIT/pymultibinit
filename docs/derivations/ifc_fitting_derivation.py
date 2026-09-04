#!/usr/bin/env python3
"""Sympy derivation: IFC-aware effective-potential model fitting.

Story 1 of specs/ifc-aware-model-fitting (vault). Companion markdown:
ifc_fitting_derivation.md quotes ONLY output printed by this script.

Equation set (PRD FR-014):
  D1  Term energy and flattening conventions (mirrors
      pymultibinit second_derivatives._fitted_term_blocks).
  D2  Product-rule per-term IFC features (diagonal f==g and mixed f!=g
      ordered-pair factors, strain-power prefactors, +/- scatter signs,
      per-origin sum) -- asserted equal to Matrix.hessian.
  D3  Coefficient linearity: E = sum_j c_j E_j  =>  d2E/du2 = sum_j c_j X_j.
  D4  Mean-normalized weighted Frobenius objective -> exact normal-equation
      increments and greedy quantities (rhs/diagonal/target_norm and the
      closed-form single-coefficient score target_norm - rhs^2/(diag+ridge)).
  D5  Ha/Bohr^2 -> eV/Ang^2 conversion factor vs CODATA-2014 constants.
  D6  Numeric spot-checks: formula Hessian == direct hessian == FD probe.

Run:  source mydev; python ifc_fitting_derivation.py
"""

import numpy as np
import sympy as s

# ----------------------------------------------------------------------
# D1: term energy conventions
# ----------------------------------------------------------------------
# One compiled term (single cell origin o):
#   T_o = alpha * S(eta) * prod_{f=1..F} d_f^{p_f},
#   d_f = U[A_f] - U[B_f]          (flat atom-major coords: 3*atom + dir),
#   S(eta) = prod_t eta_{v_t}^{q_t}.
# Full term energy sums identical structures over the ncell origins o
# (each origin has its own index pair (A_f(o), B_f(o))):
#   E_term = alpha * sum_o T_o-like structures.
# alpha is the coefficient scale of the basis column (in the fitter,
# alpha = value * weight for the fitted value; the derivation is agnostic).

print("D1 conventions: T_o = alpha * prod_t eta_t^q_t * prod_f (U[A_f]-U[B_f])^p_f")
print("   flat index = 3*atom + dir; energy linear in alpha per term.\n")

# ----------------------------------------------------------------------
# D2: product-rule per-term Hessian == direct hessian
# ----------------------------------------------------------------------
# Formula under test (single origin):
#   d2T/du_{I}du_{J} =
#     sum_{f}   alpha S p_f(p_f-1) d_f^{p_f-2} prod_{h!=f} d_h^{p_h}
#               * s_I(f) s_J(f)                      [f == g, I,J in factor f]
#   + sum_{f!=g} alpha S p_f p_g d_f^{p_f-1} d_g^{p_g-1} prod_{h!=f,g} d_h^{p_h}
#               * s_I(f) s_J(g)                      [ordered pairs]
# with s_X(f) = +1 if X is A_f, -1 if X is B_f, 0 otherwise
# (I, J range over the flat coordinates touched by the factors).
# The ordered-pair sum writes both (f,g) and (g,f), giving the symmetric
# total; a single factor's own cross terms appear once (f==g case).

def term_hessian_formula(U, alpha, strain_powers, factors):
    """Assemble H_formula by the implementation scatter rules.

    factors: list of (A_flat, B_flat, p).
    strain_powers: list of (eta_symbol, q).
    """
    F = len(factors)
    n = len(U)
    S = s.Integer(1)
    for e, q in strain_powers:
        S *= e ** q
    d = [U[A] - U[B] for (A, B, p) in factors]
    H = s.zeros(n, n)
    for f in range(F):
        _, _, pf = factors[f]
        for g in range(F):
            _, _, pg = factors[g]
            if f == g:
                if pf < 2:
                    continue
                pref = alpha * S * pf * (pf - 1) * d[f] ** (pf - 2)
                for h in range(F):
                    if h != f:
                        pref = pref * d[h] ** factors[h][2]
                for (i, si) in ((factors[f][0], 1), (factors[f][1], -1)):
                    for (j, sj) in ((factors[f][0], 1), (factors[f][1], -1)):
                        H[i, j] += si * sj * pref
            else:
                pref = (alpha * S * pf * pg
                        * d[f] ** (pf - 1) * d[g] ** (pg - 1))
                for h in range(F):
                    if h not in (f, g):
                        pref = pref * d[h] ** factors[h][2]
                for (i, si) in ((factors[f][0], 1), (factors[f][1], -1)):
                    for (j, sj) in ((factors[g][0], 1), (factors[g][1], -1)):
                        H[i, j] += si * sj * pref
    return H


def term_expr(U, alpha, strain_powers, factors):
    T = alpha
    for e, q in strain_powers:
        T *= e ** q
    for (A, B, p) in factors:
        T *= (U[A] - U[B]) ** p
    return T


# Concrete case: 6 flat coords, 3 factors (two share an atom), strain powers.
U = s.symbols("U0:6", real=True)
e1, e2 = s.symbols("eta1 eta2", real=True)
alpha = s.Symbol("alpha", real=True)
strain_powers = [(e1, 1), (e2, 2)]
factors = [(0, 1, 3), (2, 3, 4), (0, 4, 2)]  # factor 3 shares U0 with factor 1

T = term_expr(U, alpha, strain_powers, factors)
H_direct = s.hessian(T, U)
H_formula = term_hessian_formula(U, alpha, strain_powers, factors)
assert H_direct.shape == (6, 6)
resid = s.simplify(H_direct - H_formula)
assert resid == s.zeros(6, 6), "D2 FAILED: formula != direct hessian"
print("D2 PASS: scatter-formula Hessian == Matrix.hessian(T, U) "
      "(3 factors incl. shared atom, strain powers eta1^1 eta2^2).")

# Multi-origin: Hessian of the origin sum == sum of per-origin Hessians.
T2 = T + term_expr(U, alpha, strain_powers, [(3, 5, 3)])
H2_direct = s.hessian(T2, U)
H2_sum = (H_formula + term_hessian_formula(
    U, alpha, strain_powers, [(3, 5, 3)]))
assert s.simplify(H2_direct - H2_sum) == s.zeros(6, 6)
print("D2 PASS: Hessian of origin sum == sum of per-origin Hessians.\n")

# ----------------------------------------------------------------------
# D3: coefficient linearity -> unit-coefficient columns
# ----------------------------------------------------------------------
c1, c2 = s.symbols("c1 c2", real=True)
T1 = term_expr(U, s.Integer(1), strain_powers, factors)
T2b = term_expr(U, s.Integer(1), [(e1, 2)], [(5, 2, 3)])
E = c1 * T1 + c2 * T2b
H_E = s.hessian(E, U)
H_lin = c1 * s.hessian(T1, U) + c2 * s.hessian(T2b, U)
assert s.simplify(H_E - H_lin) == s.zeros(6, 6)
print("D3 PASS: hessian(c1 E1 + c2 E2) = c1 hessian(E1) + c2 hessian(E2);")
print("   hence X_j = d2 E_j / du^2 (unit-coefficient column) and")
print("   K(c) = K_fixed + sum_j c_j X_j exactly.\n")

# ----------------------------------------------------------------------
# D4: Frobenius objective -> normal equations + greedy quantities
# ----------------------------------------------------------------------
# J_ifc = lambda/n_act * sum_k w_k/m_k * || r_k - X_k c ||_F^2,
#   m_k = (3 N_k)^2 (matrix-entry mean), n_act = number of active targets.
# Normal equations (channel-factor form used by _normal_equations):
lam, w1, w2 = s.symbols("lambda w1 w2", positive=True)
m1, m2 = s.symbols("m1 m2", positive=True)
r1 = s.Matrix(s.symbols("r1_0:4", real=True))
r2 = s.Matrix(s.symbols("r2_0:9", real=True))
X1 = s.Matrix(4, 2, s.symbols("x1_:8", real=True))
X2 = s.Matrix(9, 2, s.symbols("x2_:18", real=True))
cv = s.Matrix([c1, c2])
n_act = s.Integer(2)
J = lam / n_act * (w1 / m1 * (r1 - X1 * cv).dot(r1 - X1 * cv)
                   + w2 / m2 * (r2 - X2 * cv).dot(r2 - X2 * cv))
normal = lam / n_act * (w1 / m1 * X1.T * X1 + w2 / m2 * X2.T * X2)
rhs = lam / n_act * (w1 / m1 * X1.T * r1 + w2 / m2 * X2.T * r2)
gradJ = s.Matrix([s.simplify(s.diff(J, c)) for c in cv])
lhs = s.simplify(2 * (normal * cv - rhs))
assert s.simplify(gradJ - lhs) == s.zeros(2, 1), "D4 FAILED: gradient"
print("D4 PASS: dJ/dc = 2(normal*c - rhs) identically; stationarity is")
print("   normal * c = rhs with normal, rhs as the channel-factor increments")

# Greedy quantities for a single column j (all quantities carry the factor
# f = lambda w/m folded in, matching _greedy_rhs_diagonal_target):
#   rhs_j = sum_k f_k x_kj . r_k,  diag_j = sum_k f_k ||x_kj||^2,
#   target_norm = sum_k f_k ||r_k||^2;
#   ridge-regularized single-coefficient fit: c* = rhs_j/(diag_j + rho),
#   minimal lambda-weighted value = target_norm - rhs_j^2/(diag_j + rho).
rho = s.Symbol("rho", nonnegative=True)
x = s.Matrix(s.symbols("x_0:3", real=True))
r = s.Matrix(s.symbols("q_0:3", real=True))
f = s.Symbol("f", positive=True)
rhs_j = f * (x.T * r)[0]
diag_j = f * (x.T * x)[0]
tnorm = f * (r.T * r)[0]
cstar = rhs_j / (diag_j + rho)
V = f * (r - cstar * x).dot(r - cstar * x) + rho * cstar ** 2
assert s.simplify(V - (tnorm - rhs_j ** 2 / (diag_j + rho))) == 0
print("D4 PASS: greedy score identity  min_c [f||r-xc||^2 + rho c^2] =")
print("   target_norm - rhs^2/(diag+ridge)  (closed form exact).\n")
T_fn = s.lambdify((list(U), e1, e2, alpha), T, "math")


def T_num(uvec):
    return T_fn(list(uvec), subs[e1], subs[e2], subs[alpha])
HA_EV_2014 = s.Float("27.21138602", 20)     # CODATA-2014: 1 Hartree in eV
BOHR_ANG_2014 = s.Float("0.52917721067", 20)  # CODATA-2014: 1 Bohr in Angstrom
factor = s.nsimplify(HA_EV_2014 / BOHR_ANG_2014 ** 2)
from pymultibinit.potential import BOHR_TO_ANGSTROM, HARTREE_TO_EV  # noqa: E402

impl = HARTREE_TO_EV / BOHR_TO_ANGSTROM ** 2
rel = abs(float(factor) / impl - 1.0)
assert rel < 1e-9, f"D5 FAILED: rel={rel}"
print(f"D5 PASS: Ha/Bohr^2 -> eV/Ang^2 factor = {float(factor)!r}")
print(f"   equals HARTREE_TO_EV/BOHR_TO_ANGSTROM^2 = {impl!r} (rel {rel:.2e}).\n")

# ----------------------------------------------------------------------
# D6: numeric spot-checks (formula vs direct vs central FD)
# ----------------------------------------------------------------------
rng = np.random.default_rng(20260904)
subs = {u: rng.normal() for u in U}
subs[e1] = 0.03
subs[e2] = -0.01
subs[alpha] = 0.7
H_num_formula = np.array(H_formula.subs(subs).evalf(20), dtype=float)
H_num_direct = np.array(H_direct.subs(subs).evalf(20), dtype=float)
err = np.abs(H_num_formula - H_num_direct).max()
assert err < 1e-12, f"D6 FAILED: {err}"
assert np.abs(H_num_formula - H_num_formula.T).max() < 1e-12
print(f"D6 PASS: formula == direct hessian numerically (max {err:.1e});")


u0 = np.array([subs[u] for u in U], dtype=float)
h = 1e-4
H_fd = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        e_i = np.eye(6)[i]; e_j = np.eye(6)[j]
        H_fd[i, j] = (T_num(u0 + h * e_i + h * e_j) - T_num(u0 + h * e_i - h * e_j)
                      - T_num(u0 - h * e_i + h * e_j) + T_num(u0 - h * e_i - h * e_j)) / (4 * h * h)
fd_err = np.abs(H_num_formula - H_fd).max()
scale = np.abs(H_num_formula).max()
assert fd_err < 5e-4 * max(scale, 1.0), f"D6 FAILED: fd {fd_err}"
print(f"D6 PASS: central-FD probe matches (max abs {fd_err:.1e}, scale {scale:.1e}).")
print("\nALL DERIVATION CHECKS PASSED.")
