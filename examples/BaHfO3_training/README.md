# BaHfO3 MULTIBINIT Training Example

This example shows how to call the `multibinit` binary through `pymultibinit` to build a BaHfO3 model from a DDB file and an ABINIT `HIST.nc` training trajectory.

## Files

- `01_train_bahfo3.py` - Python example using `pymultibinit.train_multibinit_model()`.
- `BaHfO3_train.abi` - template input passed to the `multibinit` executable.
- `../BaHfO3_example/BaHfO3_DDB` - BaHfO3 DDB input reused by this example.

## Dry Run

The dry run does not train a physical model. It writes a placeholder HIST file and uses a fake executable so you can inspect the output layout.

```bash
uv run python examples/BaHfO3_training/01_train_bahfo3.py --dry-run
```

Expected outputs in `examples/BaHfO3_training/training_output/`:

- `multibinit.stdout.log`
- `multibinit.stderr.log`
- `pymultibinit_training_result.json`
- `model.conf`
- `training_summary.json`

## Real Training

For real training, generate or provide a valid BaHfO3 `HIST.nc`, then run:

```bash
MULTIBINIT_BINARY=/path/to/multibinit \
uv run python examples/BaHfO3_training/01_train_bahfo3.py \
  --hist /path/to/BaHfO3_HIST.nc \
  --output-dir examples/BaHfO3_training/real_training_output
```

You can pass additional binary arguments with repeated `--binary-arg` options:

```bash
uv run python examples/BaHfO3_training/01_train_bahfo3.py \
  --hist /path/to/BaHfO3_HIST.nc \
  --executable /path/to/multibinit \
  --binary-arg=-v
```

The runner exposes input paths to the binary through `PYMULTIBINIT_DDB`, `PYMULTIBINIT_HIST`, `PYMULTIBINIT_CONFIG`, and `PYMULTIBINIT_OUTPUT_DIR`.
