#!/usr/bin/env python3
"""BaHfO3 MULTIBINIT model-building example.

This example demonstrates the pymultibinit training runner. Use ``--dry-run``
to exercise the workflow without a real MULTIBINIT executable. For real model
building, provide a real HIST.nc file and a ``multibinit`` executable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import stat

from pymultibinit import train_multibinit_model


EXAMPLE_DIR = Path(__file__).resolve().parent
PACKAGE_EXAMPLES = EXAMPLE_DIR.parent
BAHFO3_ASSETS = PACKAGE_EXAMPLES / "BaHfO3_example"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a BaHfO3 MULTIBINIT model from DDB + HIST inputs.")
    parser.add_argument(
        "--ddb",
        default=str(BAHFO3_ASSETS / "BaHfO3_DDB"),
        help="BaHfO3 DDB input file. Defaults to examples/BaHfO3_example/BaHfO3_DDB.",
    )
    parser.add_argument(
        "--hist",
        default=None,
        help="ABINIT HIST.nc training trajectory. Required unless --dry-run is used.",
    )
    parser.add_argument(
        "--config",
        default=str(EXAMPLE_DIR / "BaHfO3_train.abi"),
        help="MULTIBINIT training input file passed as the first binary argument.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(EXAMPLE_DIR / "training_output"),
        help="Directory for logs, metadata, and generated model files.",
    )
    parser.add_argument(
        "--executable",
        default=None,
        help="Path to the multibinit executable. Defaults to MULTIBINIT_BINARY/PATH.",
    )
    parser.add_argument(
        "--binary-arg",
        action="append",
        default=[],
        help="Extra argument passed to the multibinit executable. Repeat as needed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a fake binary and placeholder HIST to demonstrate the runner without ABINIT.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hist = Path(args.hist).resolve() if args.hist else None
    executable = args.executable
    if args.dry_run:
        hist = _write_placeholder_hist(output_dir / "BaHfO3_demo_HIST.nc")
        executable = str(_write_fake_multibinit(output_dir / "fake_multibinit.py"))
        print("Dry-run mode: using a placeholder HIST file and fake multibinit executable.")
    elif hist is None:
        parser.error("--hist is required for real training. Use --dry-run for a runnable demonstration.")

    result = train_multibinit_model(
        ddb=args.ddb,
        hist=hist,
        config=args.config,
        output_dir=output_dir,
        executable=executable,
        extra_args=args.binary_arg,
    )

    print("BaHfO3 training runner completed.")
    print(f"Output directory: {result.output_dir}")
    print(f"Metadata: {result.metadata_file}")
    print(f"stdout log: {result.log_file}")
    print(f"stderr log: {result.stderr_file}")
    print(f"Model config: {result.model_config or '(not reported)'}")
    print("Artifacts:")
    for name, path in result.artifacts.items():
        print(f"  {name}: {path}")
    return 0


def _write_placeholder_hist(path: Path) -> Path:
    path.write_text(
        "This is a dry-run placeholder, not a valid ABINIT HIST.nc file.\n"
        "Generate a real HIST.nc with atomchain or ABINIT for actual training.\n",
        encoding="utf-8",
    )
    return path


def _write_fake_multibinit(path: Path) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import os\n"
        "from pathlib import Path\n"
        "summary = {\n"
        "    'argv': __import__('sys').argv,\n"
        "    'ddb': os.environ.get('PYMULTIBINIT_DDB'),\n"
        "    'hist': os.environ.get('PYMULTIBINIT_HIST'),\n"
        "    'config': os.environ.get('PYMULTIBINIT_CONFIG'),\n"
        "    'output_dir': os.environ.get('PYMULTIBINIT_OUTPUT_DIR'),\n"
        "}\n"
        "Path('model.conf').write_text('ddb_file = BaHfO3_DDB\\nsys_file = BaHfO3.xml\\n', encoding='utf-8')\n"
        "Path('training_summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True), encoding='utf-8')\n"
        "print('fake multibinit completed')\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
