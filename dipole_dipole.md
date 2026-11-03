# Dipole–dipole IFCs in `pymultibinit`

## Bottom line

`pymultibinit` evaluates dipole–dipole (DD) interactions as a **frozen, harmonic linear-response contribution** to the force-constant matrix. The matrix is built from reference-structure Born effective charges and the high-frequency dielectric tensor, with a periodic Ewald-style lattice sum. Given this matrix, the DD energy is exactly quadratic and its forces exactly linear in the code's displacement coordinates; the routine does **not** recompute dipoles, Born charges, or screening as atoms move. Thus it is exact only for the chosen harmonic point-dipole model (subject to numerical summation cutoffs), not for full material electrostatics or large-amplitude motion.

The main DDB route is not a finite-displacement calculation: it reads q-space dynamical matrices, subtracts the DD model before Fourier transforming the residual, and stores the DD part separately. This decomposition follows the same general procedure documented for ABINIT's `dipdip` option. [ABINIT `dipdip` documentation](https://docs.abinit.org/variables/anaddb/#dipdip); code: `src/pymultibinit/pyeffpot/supercell_builder.py:619–638, 698–727, 790–824`.

## Harmonic IFCs and finite displacements

A second-order IFC is a Hessian of the potential energy,

\[
\Phi_{\alpha\beta}=\frac{\partial^2 V}{\partial r_\alpha\partial r_\beta}
=-\frac{\partial F_\beta}{\partial r_\alpha}.
\]

A finite-displacement method estimates this derivative from force differences at displaced supercell geometries, so its force constants are numerical estimates of the harmonic curvature. The official [Phonopy formulation](https://phonopy.github.io/phonopy/formulation.html#second-order-force-constants) gives this definition and the finite-difference expression; the original [Parlinski–Li–Kawazoe paper](https://doi.org/10.1103/PhysRevLett.78.4063) reports force constants determined from Hellmann–Feynman forces induced by an atomic displacement in a periodic supercell.

That is background for IFCs generally, not what the normal `pyeffpot` DDB path does at evaluation time. The parser reads DDB `d2E_ns`/`d2E_st` blocks into q-dependent dynamical matrices (`src/pymultibinit/pyeffpot/ddb_parser_complete.py:571–583`); the builder then expands/interpolates the q-grid and Fourier-transforms to real space (`src/pymultibinit/pyeffpot/supercell_builder.py:619–727`). The DD term it removes and adds is independently calculated from the DDB's linear-response electric data. A finite displacement performed externally to produce data would not make the *runtime DD evaluation* nonlinear.

## What the DD model contains

The physical construction is well established for polar insulating crystals: the leading long-range harmonic term is an anisotropic dipole–dipole interaction parameterized by Born effective-charge tensors and electronic dielectric screening. Gonze and Lee's DFPT paper identifies dynamical matrices, Born effective charges, and electronic dielectric tensors as second-order response quantities, and describes obtaining IFCs by Fourier-transforming dynamical matrices with separate treatment of the long-range DD interaction ([Phys. Rev. B 55, 10355 (1997)](https://doi.org/10.1103/PhysRevB.55.10355), [APS abstract](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.55.10355)). Gonze et al. describe explicit inclusion of the long-ranged anisotropic DD interaction for crystalline insulators ([Phys. Rev. B 50, 13035(R) (1994)](https://doi.org/10.1103/PhysRevB.50.13035)).

In this code, q=0 DDB entries provide `zeff` and `epsilon_inf`; the parser adds ionic charges to the Born tensors and applies a charge-acoustic-sum correction before storing the data (`src/pymultibinit/pyeffpot/ddb_parser_complete.py:593–649`). The DD routines use the Born tensors and dielectric tensor in their Ewald kernels (`src/pymultibinit/pyeffpot/dipdip.py:261–419, 588–735`). ABINIT's official documentation describes the same decomposition: DD is subtracted before Fourier transform and reintroduced analytically, with Ewald reciprocal-space and real-space sums ([ABINIT `dipdip`](https://docs.abinit.org/variables/anaddb/#dipdip)).

“Ewald” names the periodic summation method for this specified dipole kernel; it does not make the kernel a complete theory of electrostatics. The implementation has only the DD term. The original Gonze et al. discussion notes that induced quadrupoles/octupoles can produce additional long-range IFC tails even when effective charges vanish, and distinguishes the electrostatic insulating case from metallic Friedel tails (same [1994 paper](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.50.13035)). Higher multipoles, nonlinear polarization/charges, and changing dielectric response are not evaluated as part of this DD term.

There is also a numerical cutoff: `compute_dipdip_dynmat` defaults to `nreal=5`, `nrecip=8` and loops over finite integer-cell and reciprocal-index windows (`dipdip.py:588–608, 635–675`; the batched routine has the same defaults at `dipdip.py:994–1018`). Consequently the actual result is a finite approximation to the converged Ewald sum. The builder does not perform a convergence study at these call sites; quantitative accuracy should be checked by varying Ewald cutoffs/parameters where exposed.

## What happens at large displacement

`EffectivePotential` builds fixed `phi_local` and `phi_dipdip` matrices from the stored IFC arrays (`src/pymultibinit/pyeffpot/potential.py:91–113`). It evaluates each harmonic component with

\[
E_{\rm dd}=\tfrac12 u^T\Phi_{\rm dd}u, \qquad F_{\rm dd}=-\Phi_{\rm dd}u,
\]

as implemented in `potential.py:555–580`; the combined harmonic path uses the same evaluator (`potential.py:582–591`). Therefore, before displacement wrapping, doubling a displacement doubles the DD force and quadruples the DD energy. There is no amplitude threshold that switches to a geometry-dependent DD calculation. Optional XML anharmonic terms can add nonlinear contributions to the *total* potential (`potential.py:268–273`), but they do not update the harmonic DD matrix.

In the normal backend, displacements are computed relative to the reference positions and wrapped by subtracting the nearest integer from each fractional component (`potential.py:447–482`). This is componentwise fractional wrapping, not necessarily the shortest Cartesian image in a skew cell. Large internal motion can therefore alias to another displacement branch; crossing a half-cell branch can cause a switch. The standalone-compatibility path instead caches initial image shifts. [Inference from the fixed quadratic IFC and wrapping implementation.] A nonlinear replacement must also reconcile the harmonic DD baseline already in the IFCs.

For an illustration, take two collinear point dipoles in vacuum with \(p_1=qu\), \(p_2=2qu\), and separation \(R+u\). Ignoring the common Coulomb prefactor, their exact pair energy is \(-4q^2u^2/(R+u)^3\). Freezing the separation at its reference value gives \(-4q^2u^2/R^3\), the quadratic term; the difference begins at order \(u^3\). This is an illustrative point-dipole model, not a quantitative error estimate for the periodic, screened implementation. If dipole centers and response tensors were genuinely fixed and dipoles linear in \(u\), the quadratic model would instead be exact at any amplitude within that idealized model.

The matrix is also not rebuilt for a changed simulation cell. `potential.py:544–579` applies the code's displacement/strain stress correction, but this is not a fresh Ewald calculation of the strain dependence of the DD kernel, Born charges, or dielectric tensor. The reciprocal sum omits the zero-wavevector term through the \(K\cdot\epsilon_\infty K>10^{-12}\) mask (`dipdip.py:640–651`); comparisons with a different macroscopic electric boundary condition require matching that convention.

## What to do in practice

1. **Do not automatically replace the matrix merely because displacements are large.** First validate the *total* effective potential, including its fitted anharmonic terms, against first-principles energies and forces on representative distorted structures, strains and configurations from the intended simulation. Check errors against the accuracy needed for the observable; no universal displacement threshold follows from this code.
2. Check displacement scaling and symmetry along physically relevant modes, inspect errors versus amplitude, and validate on configurations not used in fitting. Compare stresses too when the cell changes. Also converge the q-grid, simulation cell and Ewald sums for the target property. Agreement of two implementations of the same harmonic model is not proof of finite-displacement accuracy.
3. If the total model fails, first consider refitting or extending the anharmonic model over the relevant configuration range. A nonlinear long-range model is warranted only if tests show that its missing nonlocal physics matters; even then, geometry-dependent point dipoles with fixed Born charges and \(\epsilon_\infty\) remain an approximation, not full electronic electrostatics.
4. For such a replacement, define the electrostatic energy and periodic boundary convention explicitly; derive consistent forces and stress including the dependence of dipole moments, separations and screening on coordinates and strain. Match its quadratic Hessian to the existing DD baseline or subtract that baseline from the remainder, and refit/revalidate anharmonic coefficients so terms already learned from first-principles data are not counted twice. This is a model change, not just a different matrix multiplication.

## Observed runtime check

Using `examples/BaHfO3_example/BaHfO3_DDB`, `ncell=(2,2,2)` and the constructed `_phi_dipdip`, a displacement of 0.05 Bohr in one Cartesian component produced \(E_{\rm dd}=3.102807997760443\times10^{-5}\) Ha. Doubling it produced \(1.2411231991041772\times10^{-4}\) Ha (exactly four times as printed); the maximum residual of \(F(2u)-2F(u)\) was zero. This exercises the actual matrix evaluator; it does not establish agreement with first-principles forces or converge the Ewald sum.

## Path-specific uncertainty

There are two supercell-builder routes. The ordinary q-point/DDB route keeps the short-range and Ewald arrays separate, and `EffectivePotential` adds them as separate harmonic parts (`supercell_builder.py:619–638, 788–900`; `potential.py:91–113`). In the IFC-only fallback, `_compute_dipole_dipole` adds the computed DD matrix to `atmfrc` and also stores it in `ewald_atmfrc` (`supercell_builder.py:1068–1100`); the evaluator then adds both arrays. **This appears to count the added DD contribution twice**, subject to the separate acoustic-sum-rule corrections. Any DD already present in the input IFCs is a further bookkeeping question. This source-level finding concerns the fallback, not the normal q-point DDB route; it has not been separately reproduced numerically.

## Sources

- A. Togo et al., [Phonopy formulation: second-order force constants and finite displacement](https://phonopy.github.io/phonopy/formulation.html#second-order-force-constants).
- K. Parlinski, Z. Q. Li, and Y. Kawazoe, [“First-Principles Determination of the Soft Mode in Cubic ZrO₂,” Phys. Rev. Lett. 78, 4063 (1997)](https://doi.org/10.1103/PhysRevLett.78.4063); [APS abstract](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.78.4063).
- X. Gonze and C. Lee, [“Dynamical matrices, Born effective charges, dielectric permittivity tensors, and interatomic force constants from density-functional perturbation theory,” Phys. Rev. B 55, 10355 (1997)](https://doi.org/10.1103/PhysRevB.55.10355); [APS abstract](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.55.10355).
- X. Gonze et al., [“Interatomic force constants from first principles: The case of α-quartz,” Phys. Rev. B 50, 13035(R) (1994)](https://doi.org/10.1103/PhysRevB.50.13035); [APS abstract](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.50.13035).
- [ABINIT ANADDB `dipdip` variable](https://docs.abinit.org/variables/anaddb/#dipdip), documenting subtraction/reintroduction of the DD term and the Ewald real-/reciprocal-space sums.
