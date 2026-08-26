"""Tests for the JAX-vs-CPU benchmark script helpers (story-008).

Covers the load-bearing logic that does not need real DDB/HIST fixtures:
target evaluation, markdown rendering, and output writing. The full data
pipeline is exercised by running ``tests/bench_jax_vs_cpu.py`` itself.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent / "bench_jax_vs_cpu.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("bench_jax_vs_cpu", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["bench_jax_vs_cpu"] = module
    spec.loader.exec_module(module)
    return module


def _sample_results():
    return {
        "environment": {
            "jax_version": "0.10.1",
            "devices": "gpu",
            "numpy_version": "2.2.6",
        },
        "b1_feature_evaluation": {
            "pool_size": 25790,
            "screen_frames": 8,
            "design_frames": 1000,
            "backends": {
                "numpy": {
                    "screen_seconds": 120.0,
                    "design_seconds": 300.0,
                    "extrapolated_50k_1000_s": 3000.0,
                },
                "jax": {
                    "screen_seconds": 60.0,
                    "design_seconds": 50.0,
                    "extrapolated_50k_1000_s": 50.0,
                },
            },
        },
        "b2_model_prediction": {
            "ncoeff": 40,
            "nconfigs": 1000,
            "backends": {
                "numpy": {"seconds": 4.0},
                "jax": {"seconds": 0.5},
            },
        },
        "b3_full_training": {
            "selection": "screened_greedy",
            "ncoeff": 40,
            "backends": {
                "numpy": {"seconds": 420.0},
                "jax": {"seconds": 240.0},
            },
        },
    }


class TestCheckTargets:
    def test_pass_when_all_targets_met(self, bench):
        verdicts = bench.check_targets(_sample_results())
        assert verdicts["b1_feature_evaluation"]["jax"]["pass"] is True
        assert verdicts["b2_model_prediction"]["jax"]["pass"] is True
        assert verdicts["b3_full_training"]["jax"]["pass"] is True  # 240 < 300

    def test_b1_threshold_is_60s(self, bench):
        results = _sample_results()
        entry = results["b1_feature_evaluation"]["backends"]["jax"]
        entry["extrapolated_50k_1000_s"] = 59.9
        assert bench.check_targets(results)["b1_feature_evaluation"]["jax"]["pass"] is True
        entry["extrapolated_50k_1000_s"] = 60.1
        assert bench.check_targets(results)["b1_feature_evaluation"]["jax"]["pass"] is False

    def test_b1_verdict_ignores_subset_design_time(self, bench):
        """A fast 200-subset build must not mask a slow 50k extrapolation."""
        results = _sample_results()
        entry = results["b1_feature_evaluation"]["backends"]["jax"]
        entry["design_seconds"] = 10.0
        entry["extrapolated_50k_1000_s"] = 3900.0
        assert bench.check_targets(results)["b1_feature_evaluation"]["jax"]["pass"] is False


    def test_b2_threshold_is_1s(self, bench):
        results = _sample_results()
        results["b2_model_prediction"]["backends"]["jax"]["seconds"] = 0.99
        assert bench.check_targets(results)["b2_model_prediction"]["jax"]["pass"] is True
        results["b2_model_prediction"]["backends"]["jax"]["seconds"] = 1.01
        assert bench.check_targets(results)["b2_model_prediction"]["jax"]["pass"] is False


    def test_non_ac_workload_makes_verdict_inapplicable(self, bench):
        results = _sample_results()
        results["b2_model_prediction"]["ncoeff"] = 5
        results["b2_model_prediction"]["nconfigs"] = 10
        results["b3_full_training"]["ncoeff"] = 12
        verdicts = bench.check_targets(results)
        assert verdicts["b2_model_prediction"]["jax"]["pass"] is None
        assert verdicts["b3_full_training"]["jax"]["pass"] is None
        # B1 has no workload pin (extrapolation-based) and stays applicable.
        assert verdicts["b1_feature_evaluation"]["jax"]["pass"] is True

    def test_b3_threshold_is_300s(self, bench):
        results = _sample_results()
        results["b3_full_training"]["backends"]["jax"]["seconds"] = 299.9
        assert bench.check_targets(results)["b3_full_training"]["jax"]["pass"] is True
        results["b3_full_training"]["backends"]["jax"]["seconds"] = 300.1
        assert bench.check_targets(results)["b3_full_training"]["jax"]["pass"] is False

    def test_missing_backend_reports_absent_not_crash(self, bench):
        results = _sample_results()
        del results["b3_full_training"]["backends"]["jax"]
        verdicts = bench.check_targets(results)
        assert verdicts["b3_full_training"]["jax"]["pass"] is None


class TestRenderMarkdown:
    def test_report_contains_all_sections_and_verdicts(self, bench):
        results = _sample_results()
        verdicts = bench.check_targets(results)
        report = bench.render_markdown(results, verdicts)
        assert "# JAX GPU vs CPU Benchmark Report" in report
        assert "## B1" in report and "## B2" in report and "## B3" in report
        assert "PASS" in report and "FAIL" in report
        assert "60" in report and "25790" in report


class TestWriteOutputs:
    def test_writes_json_and_markdown(self, bench, tmp_path):
        results = _sample_results()
        verdicts = bench.check_targets(results)
        paths = bench.write_outputs(results, verdicts, output_dir=str(tmp_path))
        for name in ("bench_results.json", "bench_report.md"):
            assert Path(paths[name]).exists()
        loaded = json.loads(Path(paths["bench_results.json"]).read_text())
        assert loaded["results"]["b2_model_prediction"]["ncoeff"] == 40
        assert "verdicts" in loaded
