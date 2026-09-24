"""tests/test_pipeline.py — placeholder test suite (round 5)."""

import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable
import numpy as np
import pytest
import src
import src.config as config
import src.data as data
import src.evaluator as evaluator
import src.models as models
import src.pipeline as pipeline
import src.router as router


# ──────────────────────────────────────────────────────────────────────────────
# Existing round-0 tests (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

class TestSkeletonImports:
    """Verify that all modules import cleanly and key constants are correct."""

    def test_model_registry_has_both_scales(self) -> None:
        """MODEL_REGISTRY must contain exactly the two planned parameter scales."""
        assert "1.5b" in config.MODEL_REGISTRY
        assert "0.5b" in config.MODEL_REGISTRY

    def test_roles_order(self) -> None:
        """ROLES must be ['general', 'math', 'code'] in that order."""
        assert config.ROLES == ["general", "math", "code"]

    def test_version_string_exists(self) -> None:
        """src package must expose a __version__ string."""
        assert isinstance(src.__version__, str)
        assert src.__version__ == "0.0.1"


# ──────────────────────────────────────────────────────────────────────────────
# Round-0.5: signature assertions (inspect only, no stubs called)
# ──────────────────────────────────────────────────────────────────────────────

class TestSignatures:
    """Assert that round-0.5 signatures match the spec (parameter names + defaults)."""

    # ── models.py ─────────────────────────────────────────────────────────────

    def test_raw_generate_signature(self) -> None:
        params = list(inspect.signature(models.HuggingFaceRunner._raw_generate).parameters)
        assert params == ["self", "query", "system_prompt", "role"]

    # ── router.py ─────────────────────────────────────────────────────────────

    def test_spherical_kmeans_init(self) -> None:
        sig = inspect.signature(router.SphericalKMeans.__init__)
        params = list(sig.parameters)
        assert params == ["self", "n_clusters", "random_state"]
        assert sig.parameters["n_clusters"].default == 3
        assert sig.parameters["random_state"].default == 42

    def test_spherical_kmeans_has_predict_one(self) -> None:
        params = list(inspect.signature(router.SphericalKMeans.predict_one).parameters)
        assert params == ["self", "x"]

    def test_embedding_router_init(self) -> None:
        sig = inspect.signature(router.EmbeddingRouter.__init__)
        params = list(sig.parameters)
        assert params == ["self", "embedding_model_name", "n_clusters", "seed"]
        assert sig.parameters["n_clusters"].default == 3
        assert sig.parameters["seed"].default == 42

    def test_calibrate_signature(self) -> None:
        sig = inspect.signature(router.EmbeddingRouter.calibrate)
        params = list(sig.parameters)
        assert params == ["self", "val_items", "score_fn", "min_cluster_warn"]
        assert sig.parameters["min_cluster_warn"].default == 10

    def test_route_with_cluster_exists(self) -> None:
        params = list(inspect.signature(router.EmbeddingRouter.route_with_cluster).parameters)
        assert params == ["self", "query"]

    def test_load_is_classmethod(self) -> None:
        assert isinstance(
            inspect.getattr_static(router.EmbeddingRouter, "load"),
            classmethod,
        )

    # ── evaluator.py ──────────────────────────────────────────────────────────

    def test_verify_python_exec_signature(self) -> None:
        sig = inspect.signature(evaluator.verify_python_exec)
        params = list(sig.parameters)
        assert params == ["response", "test_assertions", "timeout"]
        assert sig.parameters["timeout"].default == 5

    def test_verify_numeric_signature(self) -> None:
        sig = inspect.signature(evaluator.verify_numeric)
        params = list(sig.parameters)
        assert params == ["response", "answer", "rel_tol", "abs_tol"]
        assert sig.parameters["rel_tol"].default == 1e-3
        assert sig.parameters["abs_tol"].default == 1e-6

    def test_wilson_ci_signature(self) -> None:
        sig = inspect.signature(evaluator.wilson_ci)
        params = list(sig.parameters)
        assert params == ["successes", "total", "z"]
        assert sig.parameters["z"].default == 1.96

    def test_helper_stubs_exist(self) -> None:
        for name in ("extract_last_boxed", "safe_eval_arithmetic", "extract_number"):
            assert callable(getattr(evaluator, name)), f"Missing helper: {name}"


# ──────────────────────────────────────────────────────────────────────────────
# Round-1: ModelRunner behavior, factory, and mock path assertions
# ──────────────────────────────────────────────────────────────────────────────

class TestModelRunnerRound1:
    """Verify Round 1 ModelRunner behavior, mock assertions, and config integration."""

    def test_mock_runner_generate_contains_placeholder_and_no_digits(self) -> None:
        """MockRunner.generate must contain 'MOCK_PLACEHOLDER' and NO digits for all roles."""
        for role in config.ROLES:
            runner = models.MockRunner(model_id="mock-id", role=role, scale="1.5b")
            response = runner.generate(query="Calculate 42 + 100", role=role)
            assert "MOCK_PLACEHOLDER" in response
            assert not any(ch.isdigit() for ch in response), f"Found digits in {response!r}"

    def test_generate_reads_prompts_from_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """generate() must read SYSTEM_PROMPTS from config and pass it to _raw_generate."""
        sentinel_prompt = "SENTINEL_PROMPT_XYZ"
        monkeypatch.setitem(config.SYSTEM_PROMPTS, "general", sentinel_prompt)

        spy_calls: dict[str, str] = {}
        runner = models.MockRunner(model_id="mock-id", role="general", scale="1.5b")

        def spy_raw(query: str, system_prompt: str, role: str) -> str:
            spy_calls["query"] = query
            spy_calls["system_prompt"] = system_prompt
            spy_calls["role"] = role
            return "ok"

        monkeypatch.setattr(runner, "_raw_generate", spy_raw)
        result = runner.generate(query="test question", role="general")

        assert result == "ok"
        assert spy_calls["system_prompt"] == sentinel_prompt
        assert spy_calls["role"] == "general"
        assert spy_calls["query"] == "test question"

    def test_get_runner_factory(self) -> None:
        """get_runner returns MockRunner when is_mock=True, correct runner otherwise, raises on unknown."""
        r1 = models.get_runner("huggingface", "id1", "general", "1.5b", is_mock=True)
        assert isinstance(r1, models.MockRunner)
        assert r1.is_mock() is True

        r2 = models.get_runner("ollama", "id2", "math", "0.5b", is_mock=True)
        assert isinstance(r2, models.MockRunner)
        assert r2.is_mock() is True

        r3 = models.get_runner("huggingface", "id3", "general", "1.5b", is_mock=False)
        assert isinstance(r3, models.HuggingFaceRunner)
        assert r3.is_mock() is False

        r4 = models.get_runner("ollama", "id4", "code", "1.5b", is_mock=False)
        assert isinstance(r4, models.OllamaRunner)
        assert r4.is_mock() is False

        with pytest.raises(ValueError):
            models.get_runner("unsupported", "id5", "general", "1.5b", is_mock=False)

    def test_assert_mock_path(self) -> None:
        """assert_mock_path raises for results/ and artifacts/, allows results_mock/ and artifacts_mock/."""
        with pytest.raises(RuntimeError):
            models.MockRunner.assert_mock_path("/x/results/run.json")
        with pytest.raises(RuntimeError):
            models.MockRunner.assert_mock_path("/x/artifacts/run.json")

        models.MockRunner.assert_mock_path("/x/results_mock/run.json")
        models.MockRunner.assert_mock_path("/x/artifacts_mock/run.json")

    def test_real_runners_generate_raises_not_implemented(self) -> None:
        """Calling generate() on OllamaRunner raises NotImplementedError."""
        ol = models.OllamaRunner("ol-id", "math", "0.5b")
        with pytest.raises(NotImplementedError):
            ol.generate("query", "math")


# ──────────────────────────────────────────────────────────────────────────────
# Round-2: Evaluator and Metrics Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluatorRound2:
    """Verify Round 2 extraction helpers, verifiers, scoring, and metrics."""

    def test_extract_last_boxed(self) -> None:
        # plain
        assert evaluator.extract_last_boxed(r"The answer is \boxed{42}.") == "42"
        # nested braces
        assert evaluator.extract_last_boxed(r"\boxed{a + {b \times c}}") == r"a + {b \times c}"
        # multiple boxes (last wins)
        assert evaluator.extract_last_boxed(r"Step: \boxed{10}, final: \boxed{42}") == "42"
        # none / unbalanced
        assert evaluator.extract_last_boxed("no box here") is None
        assert evaluator.extract_last_boxed(r"\boxed{unclosed") is None

    def test_safe_eval_arithmetic(self) -> None:
        assert evaluator.safe_eval_arithmetic("2+3*4") == 14.0
        assert evaluator.safe_eval_arithmetic("2**10") == 1024.0
        assert evaluator.safe_eval_arithmetic("1/0") is None
        assert evaluator.safe_eval_arithmetic("__import__('os')") is None
        assert evaluator.safe_eval_arithmetic("9**9**9") is None

    def test_extract_number(self) -> None:
        # boxed
        assert evaluator.extract_number(r"\boxed{42}") == 42.0
        # boxed fraction
        assert evaluator.extract_number(r"\boxed{\frac{1}{4}}") == pytest.approx(0.25)
        assert evaluator.extract_number(r"\boxed{\dfrac{3}{4}}") == pytest.approx(0.75)
        # comma number
        assert evaluator.extract_number("The diameter is 12,742 km.") == 12742.0
        # negative
        assert evaluator.extract_number("Temperature is -15.5 C") == -15.5
        # last bare number wins
        assert evaluator.extract_number("Step 1: 10, Step 2: 42") == 42.0
        # no number -> None
        assert evaluator.extract_number("No digits here at all") is None

    def test_verify_contains(self) -> None:
        # whole-word only
        assert evaluator.verify_contains("The concatenate function", ["cat"]) is False
        assert evaluator.verify_contains("The cat is sleeping", ["cat"]) is True
        # case-insensitive
        assert evaluator.verify_contains("The CAT is here", ["cat"]) is True

    def test_verify_numeric(self) -> None:
        assert evaluator.verify_numeric("Result: 42.0", 42.0) is True
        assert evaluator.verify_numeric("Result: 42.0001", 42.0, rel_tol=1e-3) is True
        assert evaluator.verify_numeric("Result: 42.1", 42.0, rel_tol=1e-3, abs_tol=1e-6) is False
        assert evaluator.verify_numeric("No number", 42.0) is False

    def test_verify_mcq_letter(self) -> None:
        assert evaluator.verify_mcq_letter("The answer is B.", "B") is True
        assert evaluator.verify_mcq_letter(r"\boxed{C}", "C") is True
        assert evaluator.verify_mcq_letter("(B)", "B") is True
        # Article regression: "A" as a word should not trigger bare \b[A-D]\b
        assert evaluator.verify_mcq_letter("A planet orbits a star.", "A") is False
        # Wrong letter
        assert evaluator.verify_mcq_letter("The answer is B.", "C") is False

        # Additional phrase tests & regressions:
        assert evaluator.verify_mcq_letter("The answer is a planet.", "A") is False
        assert evaluator.verify_mcq_letter("The answer is dog.", "D") is False
        assert evaluator.verify_mcq_letter("The answer is a mammal", "A") is False
        assert evaluator.verify_mcq_letter("The answer is (b)", "B") is True
        assert evaluator.verify_mcq_letter("The answer is B.", "B") is True
        assert evaluator.verify_mcq_letter("The answer is B.", "A") is False
        assert evaluator.verify_mcq_letter("**Answer: C**", "C") is True

        # Boxed LaTeX wrappers:
        assert evaluator.verify_mcq_letter(r"\boxed{\textbf{D}}", "D") is True
        assert evaluator.verify_mcq_letter(r"\boxed{\textbf{D}}", "B") is False
        assert evaluator.verify_mcq_letter(r"\boxed{\mathrm{C}}", "C") is True
        assert evaluator.verify_mcq_letter(r"\boxed{\mathrm{C}}", "A") is False

    def test_verify_python_exec_regressions(self) -> None:
        # Multi-line function with passing assertions
        code_pass = "```python\ndef add(a: int, b: int) -> int:\n    result = a + b\n    return result\n```"
        passed, err = evaluator.verify_python_exec(code_pass, "assert add(2, 3) == 5\nassert add(-1, 1) == 0")
        assert passed is True
        assert err == ""

        # Failing assertion
        passed, err = evaluator.verify_python_exec(code_pass, "assert add(2, 3) == 6")
        assert passed is False
        assert "AssertionError" in err

        # Syntax error
        code_syntax = "```python\ndef broken(\n```"
        passed, err = evaluator.verify_python_exec(code_syntax, "pass")
        assert passed is False
        assert len(err) > 0

        # sys.exit(0)
        code_sysexit = "```python\nimport sys\nsys.exit(0)\n```"
        passed, _ = evaluator.verify_python_exec(code_sysexit, "pass")
        assert passed is False

        # os._exit(0)
        code_osexit = "```python\nimport os\nos._exit(0)\n```"
        passed, _ = evaluator.verify_python_exec(code_osexit, "pass")
        assert passed is False

        # while True: pass with timeout=1
        code_loop = "```python\nwhile True:\n    pass\n```"
        passed, err = evaluator.verify_python_exec(code_loop, "pass", timeout=1)
        assert passed is False
        assert err == "timeout"

        # no code block
        passed, err = evaluator.verify_python_exec("No code block here", "pass")
        assert passed is False
        assert err == "no code block"

        # Correct code whose last output is print('abc', end='') -> True
        code_no_newline = "```python\ndef solve():\n    print('abc', end='')\n    return 42\n```"
        passed, err = evaluator.verify_python_exec(code_no_newline, "assert solve() == 42")
        assert passed is True
        assert err == ""

        # Code calling input() -> False, finishing well under timeout
        import time
        code_input = "```python\ndef solve():\n    x = input()\n    return x\n```"
        t0 = time.time()
        passed, err = evaluator.verify_python_exec(code_input, "solve()", timeout=5)
        elapsed = time.time() - t0
        assert passed is False
        assert elapsed < 2.0

    def test_score_response_dispatch(self) -> None:
        item_contains = {"verifier": "contains", "answer": ["Tokyo"]}
        assert evaluator.score_response(item_contains, "The capital is Tokyo.") == 1.0
        assert evaluator.score_response(item_contains, "The capital is Paris.") == 0.0

        item_numeric = {"verifier": "numeric", "answer": 42.0}
        assert evaluator.score_response(item_numeric, "Answer: \\boxed{42}") == 1.0
        assert evaluator.score_response(item_numeric, "Answer: \\boxed{100}") == 0.0

        item_mcq = {"verifier": "mcq_letter", "answer": "C"}
        assert evaluator.score_response(item_mcq, "The answer is C.") == 1.0
        assert evaluator.score_response(item_mcq, "The answer is D.") == 0.0

        item_py = {"verifier": "python_exec", "test_assertions": "assert f(3) == 9"}
        assert evaluator.score_response(item_py, "```python\ndef f(x):\n    return x*x\n```") == 1.0
        assert evaluator.score_response(item_py, "```python\ndef f(x):\n    return x+1\n```") == 0.0

        with pytest.raises(ValueError):
            evaluator.score_response({"verifier": "unknown_verifier"}, "response")

    def test_wilson_ci(self) -> None:
        lo, hi = evaluator.wilson_ci(8, 10)
        assert 0.48 < lo < 0.50
        assert 0.93 < hi < 0.95

        assert evaluator.wilson_ci(0, 0) == (0.0, 1.0)
        assert evaluator.wilson_ci(0, 10)[0] == 0.0
        assert evaluator.wilson_ci(10, 10)[1] == 1.0

        with pytest.raises(ValueError):
            evaluator.wilson_ci(-1, 10)
        with pytest.raises(ValueError):
            evaluator.wilson_ci(11, 10)

    def test_mcnemar_exact_pvalue(self) -> None:
        assert evaluator.mcnemar_exact_pvalue(0, 0) == 1.0
        assert evaluator.mcnemar_exact_pvalue(5, 5) == 1.0
        assert abs(evaluator.mcnemar_exact_pvalue(0, 10) - 0.001953) < 1e-4

    def test_compute_metrics(self) -> None:
        records = [
            {
                "item": {"id": "1", "category": "math"},
                "routed_specialist": "math",
                "scores": {"general": 0.0, "math": 1.0, "code": 0.0},
                "baseline_score": 0.0,
            },
            {
                "item": {"id": "2", "category": "general"},
                "routed_specialist": "general",
                "scores": {"general": 1.0, "math": 0.0, "code": 0.0},
                "baseline_score": 1.0,
            },
            {
                "item": {"id": "3", "category": "code"},
                "routed_specialist": "code",
                "scores": {"general": 1.0, "math": 0.0, "code": 0.0},
                "baseline_score": 1.0,
            },
        ]

        m = evaluator.compute_metrics(records)
        assert m["n"] == 3
        assert m["routed"]["k"] == 2
        assert m["routed"]["acc"] == pytest.approx(2 / 3)
        assert m["baseline"]["k"] == 2
        assert m["baseline"]["acc"] == pytest.approx(2 / 3)
        assert m["oracle"]["k"] == 3
        assert m["oracle"]["acc"] == 1.0
        assert m["delta"] == pytest.approx(0.0)

        # McNemar b and c
        assert m["mcnemar"]["b"] == 1
        assert m["mcnemar"]["c"] == 1
        assert m["mcnemar"]["p"] == 1.0

        # Routing counts
        assert m["routing_counts"] == {"general": 1, "math": 1, "code": 1}

        # Per category
        assert m["per_category"]["math"]["n"] == 1
        assert m["per_category"]["math"]["routed_acc"] == 1.0
        assert m["per_category"]["math"]["baseline_acc"] == 0.0

        assert m["per_category"]["general"]["n"] == 1
        assert m["per_category"]["general"]["routed_acc"] == 1.0
        assert m["per_category"]["general"]["baseline_acc"] == 1.0

        assert m["per_category"]["code"]["n"] == 1
        assert m["per_category"]["code"]["routed_acc"] == 0.0
        assert m["per_category"]["code"]["baseline_acc"] == 1.0

        # ValueError cases
        # 1. Unknown specialist
        bad_spec = [dict(records[0], routed_specialist="unknown")]
        with pytest.raises(ValueError):
            evaluator.compute_metrics(bad_spec)

        # 2. Baseline mismatch
        bad_base = [dict(records[0], baseline_score=1.0)]
        with pytest.raises(ValueError):
            evaluator.compute_metrics(bad_base)

        # 3. Empty list
        with pytest.raises(ValueError):
            evaluator.compute_metrics([])


# ──────────────────────────────────────────────────────────────────────────────
# Round-3: SphericalKMeans and EmbeddingRouter Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSphericalKMeansRound3:
    """Verify Round 3 SphericalKMeans clustering and mathematical invariants."""

    def test_clustering_well_separated_groups(self) -> None:
        rng = np.random.RandomState(42)
        # 3 groups around unit coordinate axes
        g1 = np.array([1.0, 0.0, 0.0]) + rng.randn(15, 3) * 0.05
        g2 = np.array([0.0, 1.0, 0.0]) + rng.randn(15, 3) * 0.05
        g3 = np.array([0.0, 0.0, 1.0]) + rng.randn(15, 3) * 0.05
        X = np.vstack([g1, g2, g3])

        km = router.SphericalKMeans(n_clusters=3, random_state=42).fit(X)
        preds = km.predict(X)

        # Centroids must have unit norm
        centroid_norms = np.linalg.norm(km.centroids_, axis=1)
        assert np.allclose(centroid_norms, 1.0)

        # 3 well-separated groups are recovered (assignment matches true grouping up to permutation)
        assert len(set(preds[:15])) == 1
        assert len(set(preds[15:30])) == 1
        assert len(set(preds[30:45])) == 1
        assert len({preds[0], preds[15], preds[30]}) == 3

        # predict_one == predict[0]
        for i in range(len(X)):
            assert km.predict_one(X[i]) == preds[i]

    def test_reproducibility_and_data_ordering(self) -> None:
        rng = np.random.RandomState(123)
        g1 = np.array([1.0, 0.0, 0.0]) + rng.randn(10, 3) * 0.05
        g2 = np.array([0.0, 1.0, 0.0]) + rng.randn(10, 3) * 0.05
        g3 = np.array([0.0, 0.0, 1.0]) + rng.randn(10, 3) * 0.05
        X = np.vstack([g1, g2, g3])

        # Same random_state gives identical centroids
        km1 = router.SphericalKMeans(n_clusters=3, random_state=42).fit(X)
        km2 = router.SphericalKMeans(n_clusters=3, random_state=42).fit(X)
        assert np.allclose(km1.centroids_, km2.centroids_)

        # Different data orderings still recover the groups
        perm = np.random.RandomState(99).permutation(len(X))
        X_shuffled = X[perm]
        true_labels = np.array([0] * 10 + [1] * 10 + [2] * 10)[perm]

        km_shuffled = router.SphericalKMeans(n_clusters=3, random_state=42).fit(X_shuffled)
        preds_shuffled = km_shuffled.predict(X_shuffled)

        for true_group in [0, 1, 2]:
            assigned_labels = preds_shuffled[true_labels == true_group]
            assert len(set(assigned_labels)) == 1

    def test_validation_errors(self) -> None:
        km = router.SphericalKMeans(n_clusters=3, random_state=42)

        # RuntimeError when predicting before fit
        with pytest.raises(RuntimeError):
            km.predict(np.ones((5, 3)))
        with pytest.raises(RuntimeError):
            km.predict_one(np.ones(3))

        # ValueError for 1-D X
        with pytest.raises(ValueError):
            km.fit(np.array([1.0, 2.0, 3.0]))

        # ValueError for n_samples < n_clusters
        with pytest.raises(ValueError):
            km.fit(np.ones((2, 3)))

        # ValueError for zero-norm rows
        X_zero = np.array([[1.0, 0.0], [0.0, 0.0], [0.5, 0.5]])
        with pytest.raises(ValueError):
            km.fit(X_zero)


class TestEmbeddingRouterRound3:
    """Verify Round 3 EmbeddingRouter calibration, routing, serialization, and error handling."""

    @staticmethod
    def _fake_embed(texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            if t.startswith("math:"):
                base = np.array([1.0, 0.0, 0.0])
            elif t.startswith("code:"):
                base = np.array([0.0, 1.0, 0.0])
            elif t.startswith("gen:"):
                base = np.array([0.0, 0.0, 1.0])
            else:
                base = np.array([0.577, 0.577, 0.577])
            h = sum(ord(c) for c in t)
            noise = np.array([np.sin(h), np.cos(h), np.sin(h * 2)]) * 0.02
            vec = base + noise
            vec /= np.linalg.norm(vec)
            vecs.append(vec)
        return np.array(vecs, dtype=np.float64)

    def test_calibration_and_routing(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        items = (
            [{"id": f"m{i}", "question": f"math: problem {i}", "category": "math"} for i in range(12)]
            + [{"id": f"c{i}", "question": f"code: task {i}", "category": "code"} for i in range(12)]
            + [{"id": f"g{i}", "question": f"gen: query {i}", "category": "general"} for i in range(12)]
        )

        score_calls = [0]

        def fake_score_fn(item: dict, role: str) -> float:
            score_calls[0] += 1
            if item["category"] == "math":
                return 1.0 if role == "math" else (0.5 if role == "general" else 0.0)
            elif item["category"] == "code":
                return 1.0 if role == "code" else (0.5 if role == "general" else 0.0)
            else:
                return 1.0 if role == "general" else 0.0

        r = router.EmbeddingRouter("mock-model", n_clusters=3, seed=42)
        monkeypatch.setattr(r, "_embed", self._fake_embed)

        # min_cluster_warn=15 ensures warning is emitted and recorded since each cluster has 12 items
        with pytest.warns(UserWarning):
            calib = r.calibrate(items, fake_score_fn, min_cluster_warn=15)

        # Documented keys in returned dict
        expected_keys = {
            "n_items",
            "cluster_sizes",
            "capability_matrix",
            "assignment",
            "cluster_category_counts",
            "item_scores",
            "warnings",
        }
        assert set(calib.keys()) == expected_keys
        assert calib["n_items"] == len(items)
        assert sum(calib["cluster_sizes"]) == len(items)
        assert len(calib["warnings"]) > 0

        # score_fn called exactly len(items) * len(ROLES) times
        assert score_calls[0] == len(items) * len(config.ROLES)

        # assignment maps math cluster to math, code to code, gen to general
        assigned_roles = set(calib["assignment"].values())
        assert assigned_roles == {"math", "code", "general"}

        # route and route_with_cluster
        q_math = "math: what is 2+2"
        q_code = "code: write a quicksort"
        q_gen = "gen: capital of France"
        assert r.route(q_math) == "math"
        assert r.route(q_code) == "code"
        assert r.route(q_gen) == "general"

        assert r.route(q_math) == r.route_with_cluster(q_math)[0]
        assert r.route(q_code) == r.route_with_cluster(q_code)[0]
        assert r.route(q_gen) == r.route_with_cluster(q_gen)[0]

        # Save and load roundtrip
        save_file = tmp_path / "router_state" / "router.json"
        r.save(save_file)

        # Verify saved file parses with json.loads and has no pickle artifacts
        raw_content = save_file.read_text(encoding="utf-8")
        parsed_json = json.loads(raw_content)
        assert parsed_json["schema_version"] == 1
        assert "centroids" in parsed_json

        # Load router
        loaded_router = router.EmbeddingRouter.load(save_file)
        monkeypatch.setattr(loaded_router, "_embed", self._fake_embed)

        assert loaded_router.assignment_ == r.assignment_
        assert loaded_router.route(q_math) == "math"
        assert loaded_router.route(q_code) == "code"
        assert loaded_router.route(q_gen) == "general"

    def test_tie_break_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When specialist and generalist have equal scores, generalist must win."""
        items = (
            [{"id": f"m{i}", "question": f"math: problem {i}", "category": "math"} for i in range(12)]
            + [{"id": f"c{i}", "question": f"code: task {i}", "category": "code"} for i in range(12)]
            + [{"id": f"g{i}", "question": f"gen: query {i}", "category": "general"} for i in range(12)]
        )

        def tie_score_fn(item: dict, role: str) -> float:
            # Everyone scores 0.8 everywhere -> tie between general and specialists
            return 0.8

        r = router.EmbeddingRouter("mock-model", n_clusters=3, seed=42)
        monkeypatch.setattr(r, "_embed", self._fake_embed)
        calib = r.calibrate(items, tie_score_fn, min_cluster_warn=10)

        # All clusters should fall back to general
        for c in range(3):
            assert calib["assignment"][c] == "general"

    def test_errors_and_edge_cases(self, tmp_path: Path) -> None:
        uncalibrated = router.EmbeddingRouter("mock-model", n_clusters=3, seed=42)

        # route/save before calibrate raise RuntimeError
        with pytest.raises(RuntimeError):
            uncalibrated.route("query")
        with pytest.raises(RuntimeError):
            uncalibrated.route_with_cluster("query")
        with pytest.raises(RuntimeError):
            uncalibrated.save(tmp_path / "never.json")

        # calibrate with empty or too short val_items raises ValueError
        with pytest.raises(ValueError):
            uncalibrated.calibrate([], lambda it, ro: 1.0)
        with pytest.raises(ValueError):
            uncalibrated.calibrate([{"id": "1", "question": "q"}], lambda it, ro: 1.0)

        # load with wrong schema_version raises ValueError
        bad_version_file = tmp_path / "bad_version.json"
        bad_version_file.write_text(json.dumps({"schema_version": 999}), encoding="utf-8")
        with pytest.raises(ValueError):
            router.EmbeddingRouter.load(bad_version_file)

    def test_calibrate_duplicate_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calibrating with duplicate item IDs must raise ValueError and never call score_fn."""
        items = [
            {"id": "item1", "question": "math: 1+1", "category": "math"},
            {"id": "item1", "question": "math: 2+2", "category": "math"},
            {"id": "item2", "question": "code: pass", "category": "code"},
        ]
        score_calls = [0]

        def fake_score_fn(item: dict, role: str) -> float:
            score_calls[0] += 1
            return 1.0

        r = router.EmbeddingRouter("mock-model", n_clusters=3, seed=42)
        monkeypatch.setattr(r, "_embed", self._fake_embed)

        with pytest.raises(ValueError, match="Duplicate item id"):
            r.calibrate(items, fake_score_fn)

        assert score_calls[0] == 0

    def test_import_hygiene(self) -> None:
        """Verify that importing src.router loads neither torch, transformers, nor sklearn."""
        code = (
            "import sys; import src.router; "
            "print('sentence_transformers' in sys.modules, 'torch' in sys.modules, 'sklearn' in sys.modules)"
        )
        out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        assert out == "False False False"

    def test_pipeline_import_hygiene(self) -> None:
        """Verify that importing src.pipeline loads neither sentence_transformers, torch, nor sklearn."""
        code = (
            "import sys; import src.pipeline; "
            "print('sentence_transformers' in sys.modules, 'torch' in sys.modules, 'sklearn' in sys.modules)"
        )
        out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        assert out == "False False False"


# ──────────────────────────────────────────────────────────────────────────────
# Round 4 tests: data, hash_embed, pipeline execution, CLI
# ──────────────────────────────────────────────────────────────────────────────

class TestRound4Data:
    """Verify data loading, validation, and query construction."""

    def test_smoke_files_load_and_disjoint(self) -> None:
        val_path, test_path = data.dataset_paths("smoke")
        val_items = data.load_items(val_path)
        test_items = data.load_items(test_path)
        data.assert_disjoint(val_items, test_items)

        assert len(val_items) == 12
        assert len(test_items) == 9

        val_cats = [it["category"] for it in val_items]
        test_cats = [it["category"] for it in test_items]
        assert [val_cats.count(c) for c in ["general", "math", "code"]] == [4, 4, 4]
        assert [test_cats.count(c) for c in ["general", "math", "code"]] == [3, 3, 3]

        for it in val_items + test_items:
            if it["verifier"] == "python_exec":
                assert isinstance(it.get("test_assertions"), str)
                assert it["test_assertions"].strip()
            elif it["verifier"] == "numeric":
                assert not isinstance(it.get("answer"), bool)
                assert isinstance(it.get("answer"), (int, float))

    def test_load_items_errors(self, tmp_path: Path) -> None:
        # Unknown verifier
        p_unknown = tmp_path / "unknown_verifier.json"
        p_unknown.write_text(
            json.dumps([{"id": "uv1", "category": "math", "question": "q?", "verifier": "magic", "answer": 1}]),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unknown verifier"):
            data.load_items(p_unknown)

        # Missing answer
        p_missing_ans = tmp_path / "missing_answer.json"
        p_missing_ans.write_text(
            json.dumps([{"id": "ma1", "category": "math", "question": "q?", "verifier": "numeric"}]),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="missing 'answer'"):
            data.load_items(p_missing_ans)

        # Bad category
        p_bad_cat = tmp_path / "bad_cat.json"
        p_bad_cat.write_text(
            json.dumps([{"id": "bc1", "category": "history", "question": "q?", "verifier": "numeric", "answer": 1}]),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="invalid category"):
            data.load_items(p_bad_cat)

        # Duplicate ids inside file
        p_dup = tmp_path / "dup_id.json"
        p_dup.write_text(
            json.dumps([
                {"id": "dup_1", "category": "general", "question": "q1", "verifier": "contains", "answer": ["a"]},
                {"id": "dup_1", "category": "general", "question": "q2", "verifier": "contains", "answer": ["b"]},
            ]),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Duplicate item id"):
            data.load_items(p_dup)

        # assert_disjoint raises on overlap
        items_val = [{"id": "shared_01", "category": "general", "question": "q1", "verifier": "contains", "answer": ["a"]}]
        items_test = [{"id": "shared_01", "category": "math", "question": "q2", "verifier": "numeric", "answer": 2}]
        with pytest.raises(ValueError, match="share ids"):
            data.assert_disjoint(items_val, items_test)

    def test_build_query(self) -> None:
        numeric_item = {
            "id": "m1",
            "category": "math",
            "question": "What is 2+2?",
            "verifier": "numeric",
            "answer": 4,
        }
        assert data.build_query(numeric_item) == f"What is 2+2?{config.NUMERIC_SUFFIX}"

        other_item = {
            "id": "g1",
            "category": "general",
            "question": "Capital of France?",
            "verifier": "mcq_letter",
            "answer": "B",
        }
        assert data.build_query(other_item) == "Capital of France?"


class TestRound4HashEmbed:
    """Verify deterministic bag-of-words hash embedding."""

    def test_hash_embed_properties(self) -> None:
        texts = ["hello world", "test query text"]
        dim = 64
        res = pipeline.hash_embed(texts, dim=dim)

        assert res.shape == (2, dim)
        assert res.dtype == np.float64
        norms = np.linalg.norm(res, axis=1)
        assert np.allclose(norms, 1.0)

        # Deterministic across two calls
        res_again = pipeline.hash_embed(texts, dim=dim)
        assert np.array_equal(res, res_again)

        # Identical texts give identical vectors
        res_dup = pipeline.hash_embed(["same sentence", "same sentence"], dim=dim)
        assert np.array_equal(res_dup[0], res_dup[1])

        # Empty string still has unit norm
        empty_res = pipeline.hash_embed([""], dim=config.EMBEDDING_DIM)
        assert empty_res.shape == (1, config.EMBEDDING_DIM)
        assert np.isclose(np.linalg.norm(empty_res[0]), 1.0)


class TestRound4PipelineExecution:
    """End-to-end mock execution, error handling, determinism, and CLI integration."""

    @pytest.fixture(autouse=True)
    def setup_dirs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.results_mock = tmp_path / "results_mock"
        self.artifacts_mock = tmp_path / "artifacts_mock"
        self.results_prod = tmp_path / "results"
        self.artifacts_prod = tmp_path / "artifacts"
        self.router_state_file = self.artifacts_prod / "router_state.json"

        monkeypatch.setattr(config, "RESULTS_MOCK_DIR", self.results_mock)
        monkeypatch.setattr(config, "ARTIFACTS_MOCK_DIR", self.artifacts_mock)
        monkeypatch.setattr(config, "RESULTS_DIR", self.results_prod)
        monkeypatch.setattr(config, "ARTIFACTS_DIR", self.artifacts_prod)
        monkeypatch.setattr(config, "ROUTER_STATE_FILE", self.router_state_file)

    def _default_args(self, **kwargs: Any) -> argparse.Namespace:
        defaults: dict[str, Any] = {
            "scale": "0.5b",
            "backend": "huggingface",
            "mock": True,
            "limit": None,
            "skip_calibration": False,
            "seed": 7,
            "benchmark": "smoke",
            "device": "cpu",
            "val_limit": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_full_mock_run_05b(self) -> None:
        args = self._default_args(scale="0.5b")
        ret = pipeline.run(args)
        assert ret == 0

        run_dir = self.results_mock / "0.5b_smoke_seed7"
        assert run_dir.is_dir()

        # All 5 required files in run_dir + timing.json + generation_cache.jsonl + router_state.json
        for fname in ["metrics.json", "records.json", "calibration.json", "generations.json",
                      "run_config.json", "timing.json", "generation_cache.jsonl"]:
            assert (run_dir / fname).is_file(), f"Missing: {fname}"
        assert (self.artifacts_mock / "router_state.json").is_file()

        # Records check
        records = json.loads((run_dir / "records.json").read_text(encoding="utf-8"))
        assert len(records) == 9
        for r in records:
            assert r["baseline_score"] == r["scores"]["general"]

        # Run config checks (0.5b contains note, 1.5b does not)
        cfg_05 = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        assert "size_confound_note" in cfg_05
        assert cfg_05["size_confound_note"] == config.SIZE_CONFOUND_NOTE

        args_15 = self._default_args(scale="1.5b")
        ret_15 = pipeline.run(args_15)
        assert ret_15 == 0
        cfg_15 = json.loads((self.results_mock / "1.5b_smoke_seed7" / "run_config.json").read_text(encoding="utf-8"))
        assert "size_confound_note" not in cfg_15
        assert config.SIZE_CONFOUND_NOTE not in str(cfg_15)

        # Nothing was written under non-mock dirs
        assert not self.results_prod.exists()
        assert not self.artifacts_prod.exists()

    def test_determinism(self) -> None:
        args1 = self._default_args(seed=7)
        assert pipeline.run(args1) == 0
        metrics1 = (self.results_mock / "0.5b_smoke_seed7" / "metrics.json").read_text(encoding="utf-8")

        # Run again with same seed
        args2 = self._default_args(seed=7)
        assert pipeline.run(args2) == 0
        metrics2 = (self.results_mock / "0.5b_smoke_seed7" / "metrics.json").read_text(encoding="utf-8")

        assert metrics1 == metrics2

    def test_limit(self) -> None:
        args = self._default_args(limit=4)
        assert pipeline.run(args) == 0
        records = json.loads((self.results_mock / "0.5b_smoke_seed7" / "records.json").read_text(encoding="utf-8"))
        assert len(records) == 4

    def test_skip_calibration(self) -> None:
        # First normal run to produce router_state.json
        args_init = self._default_args()
        assert pipeline.run(args_init) == 0
        rec_init = json.loads((self.results_mock / "0.5b_smoke_seed7" / "records.json").read_text(encoding="utf-8"))
        routes_init = [r["routed_specialist"] for r in rec_init]

        # Second run with skip_calibration=True
        args_skip = self._default_args(skip_calibration=True)
        assert pipeline.run(args_skip) == 0
        rec_skip = json.loads((self.results_mock / "0.5b_smoke_seed7" / "records.json").read_text(encoding="utf-8"))
        routes_skip = [r["routed_specialist"] for r in rec_skip]

        assert routes_skip == routes_init

        # Without saved state returns 2
        (self.artifacts_mock / "router_state.json").unlink()
        assert pipeline.run(args_skip) == 2

    def test_oracle_style_runner_factory(self) -> None:
        val_items = data.load_items(data.dataset_paths("smoke")[0])
        test_items = data.load_items(data.dataset_paths("smoke")[1])
        all_items = {data.build_query(it): it for it in val_items + test_items}

        class OracleRunner(models.ModelRunner):
            def __init__(self, role: str) -> None:
                self.role = role

            def _raw_generate(self, query: str, system_prompt: str, role: str) -> str:
                it = all_items.get(query)
                if it is None:
                    return "wrong"
                if self.role == it["category"] or self.role == "general":
                    verifier = it["verifier"]
                    if verifier == "mcq_letter":
                        return str(it["answer"])
                    elif verifier == "numeric":
                        return f"\\boxed{{{it['answer']}}}"
                    elif verifier == "contains":
                        return it["answer"][0]
                    elif verifier == "python_exec":
                        q = it["question"]
                        if "add" in q:
                            return "def add(a, b): return a + b"
                        elif "is_even" in q:
                            return "def is_even(n): return n % 2 == 0"
                        elif "reverse_string" in q:
                            return "def reverse_string(s): return s[::-1]"
                        elif "square" in q:
                            return "def square(x): return x * x"
                        elif "multiply" in q:
                            return "def multiply(a, b): return a * b"
                        elif "is_positive" in q:
                            return "def is_positive(n): return n > 0"
                        elif "double" in q:
                            return "def double(n): return n * 2"
                        return "def func(*args): pass"
                return "wrong"

            def is_mock(self) -> bool:
                return True

        def oracle_factory(backend: str, model_id: str, role: str, scale: str, is_mock: bool, device: str = "cpu") -> models.ModelRunner:
            return OracleRunner(role=role)

        args = self._default_args()
        ret = pipeline.run(args, runner_factory=oracle_factory, embed_fn=pipeline.hash_embed)
        assert ret == 0

        metrics = json.loads((self.results_mock / "0.5b_smoke_seed7" / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["oracle"]["acc"] >= metrics["routed"]["acc"]
        assert metrics["baseline"]["acc"] == metrics["always"]["general"]["acc"]

    def test_overlapping_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        val_f = self.tmp_path / "val_overlap.json"
        test_f = self.tmp_path / "test_overlap.json"
        shared_item = [{"id": "shared_01", "category": "general", "question": "q?", "verifier": "contains", "answer": ["a"]}]
        val_f.write_text(json.dumps(shared_item), encoding="utf-8")
        test_f.write_text(json.dumps(shared_item), encoding="utf-8")

        monkeypatch.setattr(data, "dataset_paths", lambda b: (val_f, test_f))
        args = self._default_args()
        assert pipeline.run(args) == 2

    def test_benchmark_real_missing_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        empty_dir = self.tmp_path / "empty_data_dir"
        empty_dir.mkdir()
        monkeypatch.setattr(config, "DATA_DIR", empty_dir)

        args = self._default_args(benchmark="real")
        assert pipeline.run(args) == 2

    def test_mock_false_preflight_ollama(self) -> None:
        """mock=False with backend 'ollama' returns 2 at preflight; embed_fn must never be called."""
        embed_calls = [0]

        def counting_embed(texts: list[str]) -> np.ndarray:
            embed_calls[0] += 1
            return pipeline.hash_embed(texts)

        args = self._default_args(mock=False, backend="ollama")
        ret = pipeline.run(args, embed_fn=counting_embed)
        assert ret == 2
        assert embed_calls[0] == 0

    def test_assert_mock_path_root_relative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(config, "ROOT_DIR", Path("/home/u/results/proj"))

        # Root-relative mock dir must not raise
        models.MockRunner.assert_mock_path("/home/u/results/proj/results_mock/x.json")

        # Root-relative prod dir must raise
        with pytest.raises(RuntimeError):
            models.MockRunner.assert_mock_path("/home/u/results/proj/results/x.json")

        # Outside ROOT_DIR: existing behavior
        with pytest.raises(RuntimeError):
            models.MockRunner.assert_mock_path("/x/results/run.json")
        models.MockRunner.assert_mock_path("/x/results_mock/run.json")

    def test_cli_execution(self) -> None:
        res = subprocess.run(
            [sys.executable, "run_pipeline.py", "--scale", "0.5b", "--mock", "--seed", "7"],
            cwd=str(config.ROOT_DIR),
            capture_output=True,
            text=True,
        )
        assert res.returncode == 0
        assert "Routed accuracy" in res.stdout

        res_real = subprocess.run(
            [sys.executable, "run_pipeline.py", "--scale", "0.5b", "--mock", "--seed", "7", "--benchmark", "real"],
            cwd=str(config.ROOT_DIR),
            capture_output=True,
            text=True,
        )
        assert res_real.returncode == 2
        assert len(res_real.stderr.strip()) > 0


# ──────────────────────────────────────────────────────────────────────────────
# Round 5 tests: models, cache, pipeline behaviour
# ──────────────────────────────────────────────────────────────────────────────

import src.cache as cache_mod


class TestRound5Models:
    """Verify HuggingFaceRunner helpers, cache, and role-major pipeline behaviour."""

    def test_resolve_dtype_name(self) -> None:
        hf = models.HuggingFaceRunner("m", "general", "1.5b", device="cpu")
        assert hf.resolve_dtype_name("cpu", None) == "float32"
        assert hf.resolve_dtype_name("cuda", None) == "float16"
        assert hf.resolve_dtype_name("cpu", "bfloat16") == "bfloat16"
        assert hf.resolve_dtype_name("cuda", "float32") == "float32"

    def test_build_generation_kwargs(self) -> None:
        for role, expected_max in [("general", 256), ("math", 512), ("code", 256)]:
            kw = models.build_generation_kwargs(role)
            assert kw["max_new_tokens"] == expected_max
            assert kw["do_sample"] is False
            assert "repetition_penalty" in kw
            assert "temperature" not in kw

    def test_hf_runner_stores_device_dtype(self) -> None:
        hf = models.HuggingFaceRunner("m", "general", "1.5b", device="cpu", dtype="float32")
        assert hf.device == "cpu"
        assert hf.dtype == "float32"
        assert hf._model is None
        assert hf._tokenizer is None
        assert hf.last_new_tokens is None

    def test_hf_runner_bad_device_dtype(self) -> None:
        with pytest.raises(ValueError, match="device"):
            models.HuggingFaceRunner("m", "general", "1.5b", device="tpu")
        with pytest.raises(ValueError, match="dtype"):
            models.HuggingFaceRunner("m", "general", "1.5b", dtype="int8")

    def test_hf_runner_no_torch_at_import(self) -> None:
        """Importing src.models must not load torch."""
        code = (
            "import sys; import src.models; "
            "print('torch' in sys.modules)"
        )
        out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        assert out == "False"

    def test_get_runner_passes_device(self) -> None:
        """get_runner passes device to HuggingFaceRunner."""
        r = models.get_runner("huggingface", "m", "general", "1.5b", is_mock=False, device="cpu")
        assert isinstance(r, models.HuggingFaceRunner)
        assert r.device == "cpu"

        # mock ignores device
        rm = models.get_runner("huggingface", "m", "general", "1.5b", is_mock=True, device="cpu")
        assert isinstance(rm, models.MockRunner)

    def test_get_runner_signature(self) -> None:
        sig = inspect.signature(models.get_runner)
        params = list(sig.parameters)
        assert params == ["backend", "model_id", "role", "scale", "is_mock", "device"]
        assert sig.parameters["device"].default == "cpu"

    def test_load_unload_describe_noop_on_mock(self) -> None:
        """MockRunner.load/unload/describe are inherited no-ops from ModelRunner."""
        r = models.MockRunner("m", "general", "1.5b")
        r.load()   # must not raise
        r.unload()  # must not raise
        assert r.describe() == {}


class TestRound5Cache:
    """Verify GenerationCache and make_key."""

    def test_put_get_roundtrip(self, tmp_path: Path) -> None:
        c = cache_mod.GenerationCache(tmp_path / "cache.jsonl")
        c.put("k1", "response one")
        assert c.get("k1") == "response one"
        assert c.get("k_missing") is None

    def test_second_instance_sees_data(self, tmp_path: Path) -> None:
        path = tmp_path / "cache.jsonl"
        c1 = cache_mod.GenerationCache(path)
        c1.put("k1", "hello")
        c2 = cache_mod.GenerationCache(path)
        assert c2.get("k1") == "hello"

    def test_truncated_last_line_skipped(self, tmp_path: Path) -> None:
        import json as _json
        path = tmp_path / "cache.jsonl"
        # Write a valid line then a truncated/malformed line
        path.write_text(
            '{"key": "k1", "response": "good"}\n{"key": "k2',
            encoding="utf-8",
        )
        c = cache_mod.GenerationCache(path)
        assert c.get("k1") == "good"
        assert c.get("k2") is None  # truncated line skipped

    def test_make_key_differs_on_inputs(self) -> None:
        runner = models.MockRunner("model-a", "general", "1.5b")
        base_kw = {"max_new_tokens": 256, "do_sample": False, "repetition_penalty": 1.05}

        key_base = cache_mod.make_key(runner, "general", "sys-prompt", base_kw, "query")

        # Different system prompt
        assert cache_mod.make_key(runner, "general", "other-prompt", base_kw, "query") != key_base
        # Different query
        assert cache_mod.make_key(runner, "general", "sys-prompt", base_kw, "other-query") != key_base
        # Different role
        assert cache_mod.make_key(runner, "math", "sys-prompt", base_kw, "query") != key_base
        # Different gen_settings
        assert cache_mod.make_key(runner, "general", "sys-prompt", {**base_kw, "max_new_tokens": 512}, "query") != key_base

        # device / dtype: use a HuggingFaceRunner and check different device gives different key
        hf_cpu = models.HuggingFaceRunner("model-a", "general", "1.5b", device="cpu")
        hf_cuda_sim = models.HuggingFaceRunner("model-a", "general", "1.5b", device="cpu", dtype="float16")
        key_cpu = cache_mod.make_key(hf_cpu, "general", "sys-prompt", base_kw, "query")
        key_cuda = cache_mod.make_key(hf_cuda_sim, "general", "sys-prompt", base_kw, "query")
        assert key_cpu != key_cuda


class TestRound5PipelineBehaviour:
    """Role-major, cache, resume, val_limit, timing, and RuntimeError tests."""

    @pytest.fixture(autouse=True)
    def setup_dirs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.results_mock = tmp_path / "results_mock"
        self.artifacts_mock = tmp_path / "artifacts_mock"
        self.results_prod = tmp_path / "results"
        self.artifacts_prod = tmp_path / "artifacts"
        self.router_state_file = self.artifacts_prod / "router_state.json"

        monkeypatch.setattr(config, "RESULTS_MOCK_DIR", self.results_mock)
        monkeypatch.setattr(config, "ARTIFACTS_MOCK_DIR", self.artifacts_mock)
        monkeypatch.setattr(config, "RESULTS_DIR", self.results_prod)
        monkeypatch.setattr(config, "ARTIFACTS_DIR", self.artifacts_prod)
        monkeypatch.setattr(config, "ROUTER_STATE_FILE", self.router_state_file)

    def _default_args(self, **kwargs: Any) -> argparse.Namespace:
        defaults: dict[str, Any] = {
            "scale": "0.5b",
            "backend": "huggingface",
            "mock": True,
            "limit": None,
            "skip_calibration": False,
            "seed": 7,
            "benchmark": "smoke",
            "device": "cpu",
            "val_limit": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def _spy_factory(self) -> tuple[Callable[..., models.ModelRunner], list]:
        """Return (factory, event_log). factory builds spy runners that log load/generate/unload."""
        event_log: list[tuple[str, str]] = []

        class SpyRunner(models.ModelRunner):
            def __init__(self, role: str) -> None:
                self.role = role
                self.model_id = f"spy-{role}"

            def load(self) -> None:
                event_log.append(("load", self.role))

            def unload(self) -> None:
                event_log.append(("unload", self.role))

            def _raw_generate(self, query: str, system_prompt: str, role: str) -> str:
                event_log.append(("generate", role))
                return f"MOCK_PLACEHOLDER [role={role}] not answered"

            def is_mock(self) -> bool:
                return True

        def factory(backend: str, model_id: str, role: str, scale: str, is_mock: bool, device: str = "cpu") -> models.ModelRunner:
            return SpyRunner(role=role)

        return factory, event_log

    def test_role_major_ordering(self) -> None:
        """Each role is loaded once; all generates for that role are between load and unload."""
        factory, events = self._spy_factory()
        args = self._default_args()
        ret = pipeline.run(args, runner_factory=factory, embed_fn=pipeline.hash_embed)
        assert ret == 0

        # Exactly 3 loads/unloads (one per role)
        loads = [(e, r) for e, r in events if e == "load"]
        unloads = [(e, r) for e, r in events if e == "unload"]
        assert len(loads) == 3, f"Expected 3 loads, got: {events}"
        assert len(unloads) == 3
        assert [r for _, r in loads] == config.ROLES
        assert [r for _, r in unloads] == config.ROLES

        # No two roles loaded at the same time
        loaded = set()
        for ev, role in events:
            if ev == "load":
                assert role not in loaded, f"Role {role!r} loaded twice without unload"
                loaded.add(role)
            elif ev == "unload":
                loaded.discard(role)

        # Total generate calls == 3 * (n_val + n_test)
        val_items = data.load_items(data.dataset_paths("smoke")[0])
        test_items = data.load_items(data.dataset_paths("smoke")[1])
        expected_gens = 3 * (len(val_items) + len(test_items))
        actual_gens = sum(1 for e, _ in events if e == "generate")
        assert actual_gens == expected_gens

        # All generates for a role fall between that role's load and unload
        for role in config.ROLES:
            load_idx = next(i for i, (e, r) in enumerate(events) if e == "load" and r == role)
            unload_idx = next(i for i, (e, r) in enumerate(events) if e == "unload" and r == role)
            for i, (ev, r) in enumerate(events):
                if ev == "generate" and r == role:
                    assert load_idx < i < unload_idx, (
                        f"Generate for {role!r} at index {i} is outside load({load_idx})/unload({unload_idx}) window"
                    )

    def test_resume_zero_generates_on_second_run(self) -> None:
        """Second identical run makes 0 generate calls and produces identical metrics."""
        factory, events = self._spy_factory()
        args = self._default_args()

        ret = pipeline.run(args, runner_factory=factory, embed_fn=pipeline.hash_embed)
        assert ret == 0
        gen_count_first = sum(1 for e, _ in events if e == "generate")
        assert gen_count_first > 0
        metrics_first = (self.results_mock / "0.5b_smoke_seed7" / "metrics.json").read_text(encoding="utf-8")

        # Second run
        factory2, events2 = self._spy_factory()
        ret2 = pipeline.run(args, runner_factory=factory2, embed_fn=pipeline.hash_embed)
        assert ret2 == 0
        gen_count_second = sum(1 for e, _ in events2 if e == "generate")
        assert gen_count_second == 0, f"Expected 0 generates on second run, got {gen_count_second}"

        metrics_second = (self.results_mock / "0.5b_smoke_seed7" / "metrics.json").read_text(encoding="utf-8")
        assert metrics_first == metrics_second

        timing = json.loads((self.results_mock / "0.5b_smoke_seed7" / "timing.json").read_text(encoding="utf-8"))
        for role in config.ROLES:
            assert timing[role]["n_cached"] > 0
            assert timing[role]["n_generated"] == 0

    def test_resume_repromt_regenerates_changed_role(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Changing system prompt for one role causes only that role to regenerate."""
        factory, events = self._spy_factory()
        args = self._default_args()
        assert pipeline.run(args, runner_factory=factory, embed_fn=pipeline.hash_embed) == 0

        # Change system prompt for 'math' only
        original_math_prompt = config.SYSTEM_PROMPTS["math"]
        monkeypatch.setitem(config.SYSTEM_PROMPTS, "math", "CHANGED MATH PROMPT")

        factory2, events2 = self._spy_factory()
        assert pipeline.run(args, runner_factory=factory2, embed_fn=pipeline.hash_embed) == 0

        generates_by_role: dict[str, int] = {}
        for ev, r in events2:
            if ev == "generate":
                generates_by_role[r] = generates_by_role.get(r, 0) + 1

        assert generates_by_role.get("math", 0) > 0, "math should regenerate after prompt change"
        assert generates_by_role.get("general", 0) == 0
        assert generates_by_role.get("code", 0) == 0

    def test_skip_calibration_warm_cache_no_generates(self) -> None:
        """skip_calibration with warm cache generates nothing."""
        factory, events = self._spy_factory()
        args_init = self._default_args()
        assert pipeline.run(args_init, runner_factory=factory, embed_fn=pipeline.hash_embed) == 0

        factory2, events2 = self._spy_factory()
        args_skip = self._default_args(skip_calibration=True)
        assert pipeline.run(args_skip, runner_factory=factory2, embed_fn=pipeline.hash_embed) == 0
        assert sum(1 for e, _ in events2 if e == "generate") == 0

    def test_skip_calibration_cold_cache_generates_test_only(self) -> None:
        """skip_calibration with NO cache: only test items are generated (no val items)."""
        # First do a normal run to get the router state
        factory_init, _ = self._spy_factory()
        args_init = self._default_args()
        assert pipeline.run(args_init, runner_factory=factory_init, embed_fn=pipeline.hash_embed) == 0

        # Delete the cache so the next run has a cold cache
        cache_file = self.results_mock / "0.5b_smoke_seed7" / "generation_cache.jsonl"
        cache_file.unlink()

        factory2, events2 = self._spy_factory()
        args_skip = self._default_args(skip_calibration=True)
        assert pipeline.run(args_skip, runner_factory=factory2, embed_fn=pipeline.hash_embed) == 0

        gen_count = sum(1 for e, _ in events2 if e == "generate")
        test_items = data.load_items(data.dataset_paths("smoke")[1])
        # Only test items × 3 roles should have been generated
        assert gen_count == 3 * len(test_items), f"Expected {3 * len(test_items)}, got {gen_count}"

    def test_val_limit_5(self) -> None:
        """val_limit=5 uses only 5 val items for calibration."""
        args = self._default_args(val_limit=5)
        assert pipeline.run(args, embed_fn=pipeline.hash_embed) == 0
        calibration = json.loads(
            (self.results_mock / "0.5b_smoke_seed7" / "calibration.json").read_text(encoding="utf-8")
        )
        assert calibration["n_items"] == 5

    def test_val_limit_too_small_returns_2(self) -> None:
        """val_limit < N_CLUSTERS returns exit code 2."""
        args = self._default_args(val_limit=1)
        assert pipeline.run(args) == 2

    def test_timing_json_fields(self) -> None:
        """timing.json contains the expected fields per role."""
        args = self._default_args()
        assert pipeline.run(args, embed_fn=pipeline.hash_embed) == 0
        timing = json.loads(
            (self.results_mock / "0.5b_smoke_seed7" / "timing.json").read_text(encoding="utf-8")
        )
        for role in config.ROLES:
            assert role in timing
            t = timing[role]
            for field in ["n_generated", "n_cached", "seconds", "sec_per_generation",
                          "new_tokens", "tokens_per_sec"]:
                assert field in t, f"Missing field {field!r} in timing[{role!r}]"

    def test_run_config_new_fields(self) -> None:
        """run_config.json contains device, val_limit, and runner_describe."""
        args = self._default_args()
        assert pipeline.run(args, embed_fn=pipeline.hash_embed) == 0
        cfg = json.loads(
            (self.results_mock / "0.5b_smoke_seed7" / "run_config.json").read_text(encoding="utf-8")
        )
        assert "device" in cfg
        assert "val_limit" in cfg
        assert "runner_describe" in cfg

    def test_runtime_error_from_load_returns_2(self) -> None:
        """RuntimeError raised by runner.load() during the generation phase returns exit code 2."""
        class CrashRunner(models.ModelRunner):
            def __init__(self, role: str) -> None:
                self.role = role
                self.model_id = "crash-model"

            def load(self) -> None:
                raise RuntimeError("Simulated CUDA OOM")

            def _raw_generate(self, query: str, system_prompt: str, role: str) -> str:
                return "never"

            def is_mock(self) -> bool:
                return True

        def crash_factory(backend: str, model_id: str, role: str, scale: str, is_mock: bool, device: str = "cpu") -> models.ModelRunner:
            return CrashRunner(role=role)

        args = self._default_args()
        assert pipeline.run(args, runner_factory=crash_factory, embed_fn=pipeline.hash_embed) == 2

    def test_import_hygiene_round5(self) -> None:
        """import src.models, src.pipeline leaves torch/transformers/sentence_transformers out of sys.modules."""
        code = (
            "import sys; import src.models, src.pipeline; "
            "print('torch' in sys.modules, 'transformers' in sys.modules, "
            "'sentence_transformers' in sys.modules)"
        )
        out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        assert out == "False False False"

    @pytest.mark.skipif(
        not os.environ.get("RUN_REAL_MODEL_TESTS"),
        reason="Set RUN_REAL_MODEL_TESTS=1 to run real model tests",
    )
    def test_hf_runner_real_model(self) -> None:  # pragma: no cover
        """Opt-in: verify HuggingFaceRunner can generate and unload a real 0.5B model."""
        hf = models.HuggingFaceRunner(
            "Qwen/Qwen2.5-0.5B-Instruct", "general", "0.5b", device="cpu"
        )
        response = hf.generate("What is 2+2?", "general")
        assert isinstance(response, str) and len(response.strip()) > 0
        hf.unload()
        assert hf._model is None
        assert hf._tokenizer is None
