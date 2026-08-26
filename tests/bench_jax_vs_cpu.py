#!/usr/bin/env python
"""Benchmark JAX GPU vs NumPy CPU for the pymultibinit pure-Python backend.

Story 8 of the JAX GPU training spec. Measures the three PRD targets:

  B1  Feature evaluation   -- pool x frames feature matrices (features.py)
  B2  Model prediction     -- fitted 40-term model over N configs (jax_eval)
  B3  Full greedy training -- screened_greedy end-to-end (training.py)

Usage (needs a CUDA JAX install; run with the interpreter that has jax[cuda]):

    python tests/bench_jax_vs_cpu.py \
        --ddb   .../batio3_npt_md/ddb/model.ddb \
        --hist  .../batio3_npt_md/training/train_HIST.nc \
        --basis .../batio3_npt_md/model/basis.nc \
        --fitted .../batio3_npt_md/model/fitted.nc \
        --output-dir .tmp/bench_jax_vs_cpu

Outputs ``bench_report.md`` (human report) and ``bench_results.json``
(machine-readable results + verdicts) into --output-dir.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

# PRD performance targets, seconds.
TARGETS = {
    "b1_feature_evaluation": 60.0,
    "b2_model_prediction": 1.0,
    "b3_full_training": 300.0,
}

# AC canonical workloads the targets above are defined for. Verdicts are
# inapplicable (pass=None) when a run deviates from these.
AC_WORKLOAD = {
    "b2_model_prediction": {"ncoeff": 40, "nconfigs": 1000},
    "b3_full_training": {"ncoeff": 40},
}

# --------------------------------------------------------------------------- #
# Report helpers (unit-tested in tests/test_bench_jax_vs_cpu.py)
# --------------------------------------------------------------------------- #
def _seconds(section: dict, backend: str) -> float | None:
    entry = section.get("backends", {}).get(backend)
    if entry is None:
        return None
    if "extrapolated_50k_1000_s" in entry:  # B1 AC is a 50k-term build
        return entry["extrapolated_50k_1000_s"]
    return entry.get("seconds")


def _workload_applicable(section: str, body: dict) -> bool:
    """True when the run's workload matches the AC the target is defined for."""
    for key, expected in AC_WORKLOAD.get(section, {}).items():
        if body.get(key) != expected:
            return False
    return True


def check_targets(results: dict) -> dict:
    """Map each section to {backend: {limit_s, seconds, pass}}.

    ``pass`` is ``None`` when the backend was not run (absent result) or when
    the run's workload deviates from the AC workload the target is defined
    for (e.g. ``--ncoeff``/``--configs`` overrides).
    """
    verdicts: dict = {}
    for section, limit in TARGETS.items():
        body = results.get(section, {})
        applicable = _workload_applicable(section, body)
        verdicts[section] = {}
        for backend in ("numpy", "jax"):
            seconds = _seconds(body, backend)
            verdicts[section][backend] = {
                "limit_s": limit,
                "seconds": seconds,
                "pass": (
                    None
                    if seconds is None or not applicable
                    else bool(seconds < limit)
                ),
            }
    return verdicts


def _fmt(seconds):
    return "-" if seconds is None else f"{seconds:.2f}"


def _verdict(verdicts: dict, section: str, backend: str) -> str:
    v = verdicts.get(section, {}).get(backend, {}).get("pass")
    return "-" if v is None else ("PASS" if v else "FAIL")


def render_markdown(results: dict, verdicts: dict) -> str:
    env = results.get("environment", {})
    lines = [
        "# JAX GPU vs CPU Benchmark Report",
        "",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- jax: {env.get('jax_version', '?')} on {env.get('devices', '?')}",
        f"- numpy: {env.get('numpy_version', '?')}",
        f"- python: {env.get('python', '?')}",
        f"- data: {env.get('data', '?')}",
        "",
    ]

    b1 = results.get("b1_feature_evaluation", {})
    if b1:
        lines += [
            "## B1 Feature evaluation",
            "",
            f"- candidate pool: {b1.get('pool_size')} x {b1.get('screen_frames')} screening "
            f"frames; {b1.get('pool_subset', '?')} x {b1.get('design_frames')} design frames",
            f"- target (jax): < {TARGETS['b1_feature_evaluation']:.0f} s for a 50k-term x "
            f"1000-frame build, judged on the throughput extrapolation",
            "",
            "| backend | screening (s) | design build (s) | extrapolated 50k x 1000 (s) | verdict |",
            "|---|---|---|---|---|",
        ]
        for backend in ("numpy", "jax"):
            entry = b1.get("backends", {}).get(backend, {})
            lines.append(
                f"| {backend} | {_fmt(entry.get('screen_seconds'))} "
                f"| {_fmt(entry.get('design_seconds'))} "
                f"| {_fmt(entry.get('extrapolated_50k_1000_s'))} "
                f"| {_verdict(verdicts, 'b1_feature_evaluation', backend)} |"
            )
        if b1.get("note"):
            lines += ["", f"- note: {b1['note']}"]
        lines.append("")

    b2 = results.get("b2_model_prediction", {})
    if b2:
        lines += [
            "## B2 Model prediction",
            "",
            f"- {b2.get('ncoeff')}-term fitted model over {b2.get('nconfigs')} configurations",
            f"- target (jax): < {TARGETS['b2_model_prediction']:.0f} s",
            "",
            "| backend | warm | total (s) | per config (ms) | verdict |",
            "|---|---|---|---|---|",
        ]
        for backend in ("numpy", "jax"):
            entry = b2.get("backends", {}).get(backend, {})
            total = entry.get("seconds")
            per = None if total is None else total / max(b2.get("nconfigs", 1), 1) * 1e3
            lines.append(
                f"| {backend} | {entry.get('warmup', '-')} | {_fmt(total)} "
                f"| {_fmt(per)} | {_verdict(verdicts, 'b2_model_prediction', backend)} |"
            )
        lines.append("")
        if b2.get("note"):
            lines += [f"- note: {b2['note']}"]
    b3 = results.get("b3_full_training", {})
    if b3:
        lines += [
            "## B3 Full greedy training",
            "",
            f"- selection: {b3.get('selection')}, ncoeff={b3.get('ncoeff')}, "
            f"frames={b3.get('nframes', '?')}",
            f"- target (jax): < {TARGETS['b3_full_training']:.0f} s",
            "",
            "| backend | end-to-end (s) | verdict |",
            "|---|---|---|",
        ]
        for backend in ("numpy", "jax"):
            entry = b3.get("backends", {}).get(backend, {})
            lines.append(
                f"| {backend} | {_fmt(entry.get('seconds'))} "
                f"| {_verdict(verdicts, 'b3_full_training', backend)} |"
            )
        lines.append("")
        if b3.get("note"):
            lines += [f"- note: {b3['note']}"]

    return "\n".join(lines)


def write_outputs(results: dict, verdicts: dict, output_dir: str) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "bench_results.json"
    md_path = out / "bench_report.md"
    json_path.write_text(
        json.dumps({"results": results, "verdicts": verdicts}, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(results, verdicts), encoding="utf-8")
    return {"bench_results.json": str(json_path), "bench_report.md": str(md_path)}


# --------------------------------------------------------------------------- #
# Benchmark bodies
# --------------------------------------------------------------------------- #
def _environment(data_label: str) -> dict:
    env = {
        "python": platform.python_version(),
        "numpy_version": np.__version__,
        "devices": "cpu",
        "jax_version": None,
        "data": data_label,
    }
    try:
        import jax

        env["jax_version"] = jax.__version__
        env["devices"] = ",".join(str(d) for d in jax.devices())
    except Exception:
        pass
    return env

def bench_b1(basis, dataset, ncell, natom_uc, backends, screen_frames, design_frames, pool_subset):
    """Time full-pool screening and pool_subset design-build feature matrices."""
    from pymultibinit.features import evaluate_basis_features_auto

    n_available = dataset.displacement.shape[0]
    if not basis or n_available == 0:
        raise ValueError(
            f"cannot benchmark B1 with pool={len(basis)} candidates and "
            f"{n_available} frames"
        )

    section = {
        "pool_size": len(basis),
        "screen_frames": min(screen_frames, n_available),
        "design_frames": min(design_frames, n_available),
        "pool_subset": min(pool_subset, len(basis)),
        "backends": {},
        "note": (
            "The 'jax' feature backend evaluates terms with NumPy loops and only "
            "jits the stress finalization; it is NOT a JAX feature kernel."
        ),
    }
    screen_disp = dataset.displacement[:section["screen_frames"]]
    screen_strain = dataset.strain[:section["screen_frames"]]
    screen_du = dataset.du_delta[:section["screen_frames"]]
    screen_ucvol = dataset.ucvol[:section["screen_frames"]]
    sub_basis = basis[:section["pool_subset"]]
    design_disp = dataset.displacement[:section["design_frames"]]
    design_strain = dataset.strain[:section["design_frames"]]
    design_du = dataset.du_delta[:section["design_frames"]]
    design_ucvol = dataset.ucvol[:section["design_frames"]]

    for backend in backends:
        t0 = time.perf_counter()
        evaluate_basis_features_auto(
            basis, screen_disp, screen_strain, screen_du, screen_ucvol,
            ncell, natom_uc, backend=backend,
        )
        screen_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        evaluate_basis_features_auto(
            sub_basis, design_disp, design_strain, design_du, design_ucvol,
            ncell, natom_uc, backend=backend,
        )
        design_s = time.perf_counter() - t0
        # Honest throughput from the ACTUAL measured sizes; extrapolate to the
        # 50k-term x 1000-frame AC scale for the verdict.
        actual_terms = section["pool_subset"] * section["design_frames"]
        throughput = actual_terms / design_s
        section["backends"][backend] = {
            "screen_seconds": screen_s,
            "design_seconds": design_s,
            "throughput_terms_per_s": throughput,
            "extrapolated_50k_1000_s": 50000 * 1000 / throughput,
        }
    return section




def _compat_coefficients(basis):
    """Adapt NetCDF basis dicts to the attribute protocol compile_terms expects."""
    from types import SimpleNamespace

    adapted = []
    for coeff in basis:
        terms = []
        for term in getattr(coeff, "terms", ()) or ():
            if isinstance(term, dict):
                terms.append(SimpleNamespace(
                    displacements=term.get("displacements", []),
                    strains=term.get("strains", []),
                    weight=term.get("weight", 1.0),
                ))
            else:
                terms.append(term)
        if isinstance(coeff, dict):
            adapted.append(SimpleNamespace(value=coeff.get("value", 0.0), terms=terms))
        else:
            adapted.append(SimpleNamespace(value=getattr(coeff, "value", 0.0), terms=terms))
    return adapted

def bench_b2(model_basis, dataset, ncell, natom_uc, backends, nconfigs):
    """Time a fitted model over nconfigs configurations with both kernels."""
    from pymultibinit.pyeffpot.jax_eval import compile_terms, evaluate_jax, evaluate_numpy

    compiled = compile_terms(_compat_coefficients(model_basis), ncell, natom_uc)
    n_available = dataset.displacement.shape[0]
    if n_available < nconfigs:
        raise ValueError(
            f"B2 requires {nconfigs} configurations but the dataset has only "
            f"{n_available}; the <1 s target is not applicable to a shorter run"
        )
    n = nconfigs
    disp = dataset.displacement[:n]
    strain = dataset.strain[:n]

    section = {
        "ncoeff": len(model_basis),
        "nconfigs": n,
        "nconfigs_requested": nconfigs,
        "backends": {},
        "note": (
            "evaluate_jax is an EAGER kernel (no jax.jit; it re-stages jnp arrays "
            "and calls float()/np.asarray per config). The warm call below "
            "excludes first-call staging only, not a JIT compilation."
        ),
    }

    # NumPy kernel (with one warm call so allocation caches are hot).
    if "numpy" in backends:
        evaluate_numpy(compiled, disp[0], strain[0])
        t0 = time.perf_counter()
        for t in range(n):
            evaluate_numpy(compiled, disp[t], strain[t])
        section["backends"]["numpy"] = {
            "seconds": time.perf_counter() - t0, "warmup": "1 config"
        }

    # JAX kernel, eager per-config dispatch (the production prediction path).
    if "jax" in backends:
        t0 = time.perf_counter()
        evaluate_jax(compiled, disp[0], strain[0])  # first-call staging
        warm = time.perf_counter() - t0
        t0 = time.perf_counter()
        for t in range(n):
            evaluate_jax(compiled, disp[t], strain[t])
        total = time.perf_counter() - t0
        section["backends"]["jax"] = {
            "seconds": total, "warmup": f"{warm:.2f} s (first-call staging, excluded)"
        }
    return section


def bench_b3(ddb, hist, basis, ncell, backends, ncoeff, pool_size, screen_frames):
    """End-to-end screened-greedy fit, timed per feature backend."""
    from pymultibinit.training import PythonFitConfig, fit_multibinit_model_python

    section = {
        "selection": "screened_greedy",
        "ncoeff": ncoeff,
        "backends": {},
        "note": (
            "training._fit_screened_greedy hardwires candidate scoring to the "
            "numpy backend; the jax label covers the pool-fit stage only, "
            "screening is always CPU."
        ),
    }
    for backend in backends:
        cfg = PythonFitConfig(
            ncell=ncell,
            selection="screened_greedy",
            ncoeff=ncoeff,
            regularization=1e-8,
            feature_backend=backend,
            candidate_pool_size=pool_size,
            screening_frame_count=screen_frames,
        )
        t0 = time.perf_counter()
        result = fit_multibinit_model_python(
            ddb=str(ddb), hist=str(hist), basis_xml=str(basis), config=cfg
        )
        section["backends"][backend] = {"seconds": time.perf_counter() - t0}
        section["nframes"] = result.nframes
    return section
def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return ivalue




# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def _load_inputs(ddb: Path, hist: Path, basis: Path, ncell):
    from pymultibinit.training import (
        _reference_frame_from_ddb,
        build_training_dataset,
        load_xml_basis,
        read_basis_netcdf,
        read_hist_frames,
    )

    reference = _reference_frame_from_ddb(str(ddb), ncell)
    frames = read_hist_frames(str(hist))
    dataset = build_training_dataset(reference, frames)
    if str(basis).endswith(".nc"):
        pool = read_basis_netcdf(str(basis))
    else:
        pool = load_xml_basis(str(basis))
    return pool, dataset


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ddb", required=True, type=Path)
    parser.add_argument("--hist", required=True, type=Path)
    parser.add_argument("--basis", required=True, type=Path, help="candidate pool (.xml or .nc)")
    parser.add_argument("--fitted", required=True, type=Path,
                        help="fitted model (.nc) providing the B2 model coefficients")
    parser.add_argument("--backends", default="numpy,jax")
    parser.add_argument("--skip-b3", action="store_true")
    parser.add_argument("--output-dir", default=".tmp/bench_jax_vs_cpu")
    parser.add_argument("--ncell", nargs=3, type=_positive_int, default=(2, 2, 2))
    parser.add_argument("--frames", type=_positive_int, default=1000)
    parser.add_argument("--screen-frames", type=_positive_int, default=8)
    parser.add_argument("--pool-subset", type=_positive_int, default=200)
    parser.add_argument("--ncoeff", type=_positive_int, default=40)
    parser.add_argument("--configs", type=_positive_int, default=1000)

    args = parser.parse_args(argv)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if not backends:
        parser.error("--backends must name at least one of: numpy, jax")
    unknown = [b for b in backends if b not in ("numpy", "jax")]
    if unknown:
        parser.error(f"unknown backend(s) {unknown}; expected numpy and/or jax")
    ncell = tuple(args.ncell)

    print(f"[bench] loading {args.hist.name} and {args.basis.name} ...", flush=True)
    pool, dataset = _load_inputs(args.ddb, args.hist, args.basis, ncell)
    natom_uc = dataset.displacement.shape[1] // (ncell[0] * ncell[1] * ncell[2])
    print(f"[bench] pool={len(pool)} frames={dataset.displacement.shape[0]} "
          f"natom_sc={dataset.displacement.shape[1]}", flush=True)

    results = {"environment": _environment(str(args.hist))}

    def _checkpoint():
        write_outputs(results, check_targets(results), args.output_dir)

    print("[bench] B1 feature evaluation ...", flush=True)
    results["b1_feature_evaluation"] = bench_b1(
        pool, dataset, ncell, natom_uc, backends,
        args.screen_frames, args.frames, args.pool_subset,
    )
    _checkpoint()

    # B2: honest model = the largest-|value| coefficients of the fitted model,
    # exactly --ncoeff of them (fail loudly if the fit has fewer).
    from pymultibinit.training import read_basis_netcdf

    fitted = read_basis_netcdf(str(args.fitted))
    ranked = sorted(fitted, key=lambda c: abs(getattr(c, "value", 0.0) or 0.0), reverse=True)
    model_basis = ranked[: args.ncoeff]
    if len(model_basis) != args.ncoeff:
        raise ValueError(
            f"fitted model has only {len(model_basis)} coefficients; "
            f"B2 requires exactly {args.ncoeff}"
        )
    print(f"[bench] B2 model: top-{len(model_basis)} |value| terms from {args.fitted.name}",
          flush=True)
    print("[bench] B2 model prediction ...", flush=True)
    results["b2_model_prediction"] = bench_b2(
        model_basis, dataset, ncell, natom_uc, backends, args.configs,
    )
    _checkpoint()

    if not args.skip_b3:
        print("[bench] B3 full screened-greedy training ...", flush=True)
        results["b3_full_training"] = bench_b3(
            args.ddb, args.hist, args.basis, ncell, backends,
            args.ncoeff, args.pool_subset, args.screen_frames,
        )
    _checkpoint()

    verdicts = check_targets(results)
    paths = write_outputs(results, verdicts, args.output_dir)
    print(render_markdown(results, verdicts))
    print(f"[bench] wrote {paths['bench_report.md']} and {paths['bench_results.json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
