# BaTiO3 Real Training Run Notes

This directory records a real BaTiO3 training run using generated BaTiO3 artifacts.

## Inputs

- DDB: `/home/hexu/projects/atomchain_dev/atomchain/.tmp/batio3_ddb_example/BaTiO3_stress.ddb`
- HIST: `BaTiO3_multibinit_HIST.nc`
- MULTIBINIT input: `BaTiO3_fit.abi`
- Binary: `/home/hexu/projects/abibuildbot_dev/abibuildbot/docker-worker-ubuntu22-gcc-openmpi-openblas/state/docker_ubuntu22.04_gnu_openmpi_openblas/abinit_master/src/98_main/multibinit`

## Important Corrections

Generated DDB files must not be hand-edited. The DDB marker, complete ABINIT text header, fixed-width numeric fields, block count, q-point records, and block summary are emitted by `atomchain.ddb.writer`; the DDB was regenerated through `write_ddb_from_finite_difference()`.

The generated DDB includes a 2x2x2 phonon q-grid, real spglib symmetry metadata (`nsym`, `symafm`, `symrel`, `tnons`), Gamma elastic strain-strain blocks, and Gamma displacement-strain internal-strain blocks from finite differences of forces under homogeneous strain.

It does not encode finite-q `dPhi(q)/dstrain` terms as second-order DDB blocks. Those terms are third-order responses and require a separate third-order representation.

## Current Status

`multibinit BaTiO3_fit.abi --dry-run` recognizes the generated file as a DDB and exits successfully in dry-run mode.

The real `pymultibinit.train_multibinit_model()` wrapper run completes with return code 0 and writes outputs under:

```text
wrapper_run_qgrid/
```

Key outputs:

- `wrapper_run_qgrid/multibinit.stdout.log`
- `wrapper_run_qgrid/multibinit.stderr.log`
- `wrapper_run_qgrid/pymultibinit_training_result.json`
- `wrapper_run_qgrid/BaTiO3_fit_coeffs.xml`
- `wrapper_run_qgrid/TRS_fit_diff_energy.dat`
- `wrapper_run_qgrid/TRS_fit_diff_stress.dat`

`model_config` is currently `null` because this MULTIBINIT input writes XML coefficients and diagnostics, not a `.conf`/`.ini` model config file.
