"""Tests for scripts/run_bomze_dirac.py helper functions.

Tests the offline/unit-testable parts: solution extraction, sample analysis,
graph building, and JSON output — without calling real Dirac hardware.
"""

import json
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from dirac_bench.problems import bomze_regularize
from dirac_bench.test_graphs import make_overlapping_k4, make_erdos_renyi

# Import script functions under test
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "run_bomze_dirac",
    Path(__file__).resolve().parent.parent / "scripts" / "run_bomze_dirac.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

extract_solutions_from_response = _mod.extract_solutions_from_response
analyze_samples = _mod.analyze_samples
build_graph_list = _mod.build_graph_list
save_analysis = _mod.save_analysis
BACKEND_CLOUD = _mod.BACKEND_CLOUD
BACKEND_DIRECT = _mod.BACKEND_DIRECT


# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def k4_matrices():
    """Return (A, A_bar) for overlapping K4 graph."""
    A, omega, name = make_overlapping_k4()
    return A, bomze_regularize(A)


def _make_cloud_response(solutions: list[list[float]]) -> dict:
    """Build a mock cloud solver result dict."""
    return {
        "raw_response": {"results": {"solutions": solutions}},
        "solve_time": 1.0,
    }


def _make_direct_response(solutions: np.ndarray) -> dict:
    """Build a mock direct solver result dict."""
    return {
        "raw_response": {"solution": solutions},
        "solve_time": 1.0,
    }


# ── TestExtractSolutions ─────────────────────────────────────────────────

class TestExtractSolutions:
    def test_cloud_basic(self):
        sol = [[0.5, 0.5, 0.0], [0.3, 0.3, 0.4]]
        result = _make_cloud_response(sol)
        extracted = extract_solutions_from_response(result, BACKEND_CLOUD)
        assert len(extracted) == 2
        np.testing.assert_allclose(extracted[0], [0.5, 0.5, 0.0])
        np.testing.assert_allclose(extracted[1], [0.3, 0.3, 0.4])

    def test_cloud_empty_raises(self):
        result = {"raw_response": {"results": {"solutions": []}}}
        with pytest.raises(RuntimeError, match="no solutions"):
            extract_solutions_from_response(result, BACKEND_CLOUD)

    def test_direct_2d(self):
        arr = np.array([[0.5, 0.5, 0.0], [0.3, 0.3, 0.4]])
        result = _make_direct_response(arr)
        extracted = extract_solutions_from_response(result, BACKEND_DIRECT)
        assert len(extracted) == 2
        np.testing.assert_allclose(extracted[0], [0.5, 0.5, 0.0])

    def test_direct_1d_reshapes(self):
        arr = np.array([0.5, 0.3, 0.2])
        result = _make_direct_response(arr)
        extracted = extract_solutions_from_response(result, BACKEND_DIRECT)
        assert len(extracted) == 1
        np.testing.assert_allclose(extracted[0], [0.5, 0.3, 0.2])

    def test_unknown_backend_raises(self):
        result = {"raw_response": {}}
        with pytest.raises(ValueError, match="Unknown backend"):
            extract_solutions_from_response(result, "quantum_unicorn")


# ── TestAnalyzeSamples ───────────────────────────────────────────────────

class TestAnalyzeSamples:
    def test_clique_solution_detected(self, k4_matrices):
        """Uniform weight on first K4 {0,1,2,3} should be a clique."""
        A, A_bar = k4_matrices
        x = np.zeros(6)
        x[:4] = 0.25  # uniform on {0,1,2,3} — a K4 clique
        samples, summary = analyze_samples([x], A, A_bar)
        assert summary["clique_count"] == 1
        assert summary["spurious_count"] == 0
        assert samples[0]["is_clique"] is True
        assert samples[0]["support"] == [0, 1, 2, 3]

    def test_spurious_solution_detected(self, k4_matrices):
        """Weight spread across both K4s (non-clique support) is spurious."""
        A, A_bar = k4_matrices
        x = np.ones(6) / 6  # uniform on all 6 — support is NOT a clique
        samples, summary = analyze_samples([x], A, A_bar)
        assert summary["spurious_count"] == 1
        assert summary["clique_count"] == 0
        assert samples[0]["is_clique"] is False

    def test_multiple_samples_counts(self, k4_matrices):
        A, A_bar = k4_matrices
        x_clique = np.zeros(6)
        x_clique[:4] = 0.25
        x_spurious = np.ones(6) / 6
        samples, summary = analyze_samples(
            [x_clique, x_spurious, x_clique], A, A_bar
        )
        assert summary["total_samples"] == 3
        assert summary["clique_count"] == 2
        assert summary["spurious_count"] == 1
        assert summary["spurious_pct"] == pytest.approx(33.3, abs=0.1)

    def test_objectives_computed(self, k4_matrices):
        A, A_bar = k4_matrices
        x = np.zeros(6)
        x[:4] = 0.25
        samples, summary = analyze_samples([x], A, A_bar)
        assert samples[0]["obj_standard"] > 0
        assert samples[0]["obj_bomze"] > samples[0]["obj_standard"]

    def test_support_size_stats(self, k4_matrices):
        A, A_bar = k4_matrices
        x1 = np.zeros(6)
        x1[:4] = 0.25  # support size 4
        x2 = np.zeros(6)
        x2[:3] = 1.0 / 3  # support size 3
        samples, summary = analyze_samples([x1, x2], A, A_bar)
        assert summary["support_size_min"] == 3
        assert summary["support_size_max"] == 4
        assert summary["support_size_mean"] == 3.5

    def test_empty_solutions(self, k4_matrices):
        A, A_bar = k4_matrices
        samples, summary = analyze_samples([], A, A_bar)
        assert summary["total_samples"] == 0
        assert summary["clique_count"] == 0

    def test_threshold_parameter(self, k4_matrices):
        """Near-zero weight should be excluded at appropriate threshold."""
        A, A_bar = k4_matrices
        x = np.zeros(6)
        x[:4] = 0.2499
        x[4] = 0.0004  # above default 1e-4, below 1e-3
        samples_default, _ = analyze_samples([x], A, A_bar, threshold=1e-4)
        samples_strict, _ = analyze_samples([x], A, A_bar, threshold=1e-3)
        assert 4 in samples_default[0]["support"]
        assert 4 not in samples_strict[0]["support"]


# ── TestBuildGraphList ───────────────────────────────────────────────────

class TestBuildGraphList:
    def _make_args(self, **kwargs):
        defaults = {
            "graph": None, "inline_graph": None,
            "dimacs_dir": "problems/dimacs", "max_nodes": None,
        }
        defaults.update(kwargs)
        return Namespace(**defaults)

    def test_default_returns_four_graphs(self):
        args = self._make_args()
        graphs = build_graph_list(args)
        assert len(graphs) == 4
        names = [g[2] for g in graphs]
        assert "overlap_2xK4" in names
        assert "ER(20,0.7)" in names
        assert "ER(30,0.5)" in names
        assert "ER(50,0.9)" in names

    def test_inline_graph_filter(self):
        args = self._make_args(inline_graph="ER(20,0.7)")
        graphs = build_graph_list(args)
        assert len(graphs) == 1
        assert graphs[0][2] == "ER(20,0.7)"

    def test_inline_graph_invalid_exits(self):
        args = self._make_args(inline_graph="NONEXISTENT")
        with pytest.raises(SystemExit):
            build_graph_list(args)

    def test_overlap_k4_has_known_omega(self):
        args = self._make_args(inline_graph="overlap_2xK4")
        graphs = build_graph_list(args)
        A, omega, name = graphs[0]
        assert omega == 4
        assert A.shape == (6, 6)

    def test_er_graphs_have_no_known_omega(self):
        args = self._make_args(inline_graph="ER(50,0.9)")
        graphs = build_graph_list(args)
        assert graphs[0][1] is None


# ── TestSaveAnalysis ─────────────────────────────────────────────────────

class TestSaveAnalysis:
    def _make_args(self, results_dir, no_save=False):
        return Namespace(
            no_save=no_save, results_dir=results_dir,
            num_samples=10, relaxation_schedule=2,
        )

    def test_writes_json(self, k4_matrices):
        A, A_bar = k4_matrices
        x = np.zeros(6)
        x[:4] = 0.25
        samples, summary = analyze_samples([x], A, A_bar)

        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._make_args(tmpdir)
            path = save_analysis(
                "test_graph", 6, BACKEND_CLOUD, "bomze",
                args, samples, summary, 1.5, "20260323T120000",
            )
            assert path is not None
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["graph_name"] == "test_graph"
            assert data["summary"]["clique_count"] == 1
            assert len(data["samples"]) == 1

    def test_no_save_flag(self, k4_matrices):
        A, A_bar = k4_matrices
        samples, summary = analyze_samples([], A, A_bar)

        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._make_args(tmpdir, no_save=True)
            path = save_analysis(
                "test_graph", 6, BACKEND_CLOUD, "standard",
                args, samples, summary, 1.0, "20260323T120000",
            )
            assert path is None

    def test_filename_sanitization(self, k4_matrices):
        A, A_bar = k4_matrices
        samples, summary = analyze_samples([], A, A_bar)

        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._make_args(tmpdir)
            path = save_analysis(
                "ER(50,0.9)", 50, BACKEND_CLOUD, "bomze",
                args, samples, summary, 1.0, "20260323T120000",
            )
            assert "(" not in path.name
            assert ")" not in path.name


# ── TestSharedGraphBuilders ──────────────────────────────────────────────

class TestSharedGraphBuilders:
    def test_overlapping_k4_shape(self):
        A, omega, name = make_overlapping_k4()
        assert A.shape == (6, 6)
        assert omega == 4
        assert name == "overlap_2xK4"

    def test_overlapping_k4_symmetric(self):
        A, _, _ = make_overlapping_k4()
        np.testing.assert_array_equal(A, A.T)

    def test_overlapping_k4_zero_diagonal(self):
        A, _, _ = make_overlapping_k4()
        np.testing.assert_array_equal(np.diag(A), np.zeros(6))

    def test_erdos_renyi_returns_3_tuple(self):
        A, omega, name = make_erdos_renyi(10, 0.5, seed=123)
        assert A.shape == (10, 10)
        assert omega is None
        assert name == "ER(10,0.5)"

    def test_erdos_renyi_deterministic(self):
        A1, _, _ = make_erdos_renyi(20, 0.7, seed=42)
        A2, _, _ = make_erdos_renyi(20, 0.7, seed=42)
        np.testing.assert_array_equal(A1, A2)

    def test_erdos_renyi_different_seeds(self):
        A1, _, _ = make_erdos_renyi(20, 0.7, seed=1)
        A2, _, _ = make_erdos_renyi(20, 0.7, seed=2)
        assert not np.array_equal(A1, A2)
