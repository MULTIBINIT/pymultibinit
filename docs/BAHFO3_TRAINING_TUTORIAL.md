# BaHfO3 MULTIBINIT Training Tutorial

This tutorial explains how to prepare BaHfO3 DDB + HIST inputs and call the `multibinit` binary through `pymultibinit`.

## Scope

`pymultibinit` owns the binary invocation. It does not implement ABINIT library-mode model building. The Python API validates paths, runs the `multibinit` executable, captures logs, and records metadata.

## 1. Prepare Inputs

You need these files:

- A BaHfO3 DDB file with harmonic/response data.
- A BaHfO3 ABINIT `HIST.nc` trajectory with training structures and labels.
- A MULTIBINIT training input file accepted by your ABINIT/MULTIBINIT build.
- A working `multibinit` executable.

This repository already includes an example DDB at:

```text
examples/BaHfO3_example/BaHfO3_DDB
```

The HIST file is not committed because it is normally generated from a training trajectory. You can generate it with ABINIT or with atomchain's HIST workflow.

## 2. Generate HIST With Atomchain

One practical route is to use atomchain to generate a training bundle from structures sampled with ML potentials, phonon modes, MD, or metastable distortions.

Example command shape:

```bash
mltraining artifacts \
  --structure BaHfO3.vasp \
  --ddb BaHfO3_DDB \
  --hist BaHfO3_HIST.nc \
  --output-dir BaHfO3_training_bundle
```

Use the exact atomchain options needed for your dataset and calculator. The important output for this tutorial is `BaHfO3_HIST.nc`.

## 3. Check The Training Template

The example template is:

```text
examples/BaHfO3_training/BaHfO3_train.abi
```

Adapt it to the exact `multibinit` training syntax in your ABINIT build. `pymultibinit` passes this file as the first positional argument to the executable and exposes resolved input paths as environment variables:

- `PYMULTIBINIT_DDB`
- `PYMULTIBINIT_HIST`
- `PYMULTIBINIT_CONFIG`
- `PYMULTIBINIT_OUTPUT_DIR`

## 4. Run A Dry-Run Demonstration

Before using a real binary, run the dry-run example to verify the Python side:

```bash
uv run python examples/BaHfO3_training/01_train_bahfo3.py --dry-run
```

This creates `examples/BaHfO3_training/training_output/` with logs, metadata, and fake model files. It is not a physical training result.

## 5. Run Real Training

Set the binary path and pass a real HIST file:

```bash
MULTIBINIT_BINARY=/path/to/multibinit \
uv run python examples/BaHfO3_training/01_train_bahfo3.py \
  --hist /path/to/BaHfO3_HIST.nc \
  --output-dir examples/BaHfO3_training/real_training_output
```

Equivalent CLI form:

```bash
mbtools train \
  examples/BaHfO3_example/BaHfO3_DDB \
  /path/to/BaHfO3_HIST.nc \
  --config examples/BaHfO3_training/BaHfO3_train.abi \
  --output-dir examples/BaHfO3_training/real_training_output \
  --executable /path/to/multibinit
```

Pass binary-specific flags with repeated `--binary-arg` options:

```bash
mbtools train examples/BaHfO3_example/BaHfO3_DDB /path/to/BaHfO3_HIST.nc \
  --config examples/BaHfO3_training/BaHfO3_train.abi \
  --executable /path/to/multibinit \
  --binary-arg=-v
```

## 6. Inspect Outputs

The output directory contains:

- `multibinit.stdout.log` - captured standard output.
- `multibinit.stderr.log` - captured standard error.
- `pymultibinit_training_result.json` - command, resolved paths, return code, logs, and discovered artifacts.
- Any model/configuration files written by `multibinit`.

`pymultibinit` attempts to identify a model config named `model.conf`, `multibinit.conf`, `trained_model.conf`, or another `.conf`/`.ini` artifact.

## 7. Verified Local Run

A real local run was completed with the ABINIT buildbot `multibinit` binary and atomchain-generated BaHfO3 artifacts:

```text
examples/BaHfO3_training/real_training_run/
```

The run used:

- DDB: `/home/hexu/projects/atomchain_dev/atomchain/.tmp/batio3_ddb_example/BaTiO3_stress.ddb` (historical path; the source data is BaHfO3)
- HIST: `examples/BaHfO3_training/real_training_run/BaTiO3_multibinit_HIST.nc` (historical filename kept as audit trail)
- input: `examples/BaHfO3_training/real_training_run/BaTiO3_fit.abi` (historical filename kept as audit trail)
- binary: `/home/hexu/projects/abibuildbot_dev/abibuildbot/docker-worker-ubuntu22-gcc-openmpi-openblas/state/docker_ubuntu22.04_gnu_openmpi_openblas/abinit_master/src/98_main/multibinit`

Successful outputs include `BaTiO3_fit_coeffs.xml`, `TRS_fit_diff_energy.dat`, `TRS_fit_diff_stress.dat`, and `pymultibinit_training_result.json` in `real_training_run/wrapper_run_qgrid/`. (Historical filenames inside `real_training_run/` are intentionally preserved as an audit trail.)

The atomchain DDB writer must emit the complete ABINIT text header, `d22.14` header floats, padded integer arrays, fixed-format `qpt` records, exact block counts, block characteristics, and ABINIT column-major `symrel` records. The verified DDB uses a 2x2x2 q-grid with real spglib symmetry plus Gamma elastic and Gamma internal-strain response blocks.

Finite-q `dPhi(q)/dstrain` is a third-order response and is not encoded as a second-order DDB strain perturbation.

## 8. Use The Trained Model

After training, initialize the calculator from the generated model configuration:

```python
from pymultibinit import MultibinitCalculator

calc = MultibinitCalculator.from_config_file("examples/BaHfO3_training/real_training_output/model.conf")
```

The resulting config must reference any generated DDB/XML/coefficient files using paths valid from the config location.

Some MULTIBINIT inputs write coefficient XML rather than a `.conf` or `.ini` file. In that case `pymultibinit_training_result.json` will have `model_config: null`; use the generated XML according to the consuming MULTIBINIT workflow.

## Pure-Python Fitting Path

For fitting XML coefficient values without calling the external `multibinit` binary, use `mbtools train-python` or `fit_multibinit_model_python()`. The pure-Python workflow reads DDB/HIST/XML inputs, evaluates basis features, optionally performs greedy term selection, and writes fitted XML plus diagnostics JSON.

Example BaHfO3 pure-Python training run:

```bash
mbtools train-python \
  examples/BaHfO3_example/BaHfO3_DDB \
  examples/BaHfO3_training/real_training_run/BaTiO3_multibinit_HIST.nc \
  --basis-xml examples/BaHfO3_training/real_training_run/wrapper_run_qgrid/BaTiO3_fit_coeffs.xml \
  --output-xml BaHfO3_fit_python.xml \
  --diagnostics-json BaHfO3_fit_python_diagnostics.json \
  --ncell 2 2 2 \
  --selection greedy \
  --ncoeff 3 \
  --regularization 1e-8
```

The fitted XML can be tested without `libabinit` through the pure-Python backend:

```python
import numpy as np
from pymultibinit import MultibinitPotential

pot = MultibinitPotential.from_pyeffpot(
    "examples/BaHfO3_example/BaHfO3_DDB",
    xml_file="BaHfO3_fit_python.xml",
    ncell=(2, 2, 2),
)
positions, lattice, _ = pot.get_supercell_structure()
energy, forces, stress = pot.evaluate(positions, lattice, skip_atom_matching=True)
assert np.isfinite(energy)
assert np.linalg.norm(forces) < 1e-8
```

See `PURE_PYTHON_TRAINING.md` for the detailed procedure and the logic behind term generation and term selection.
