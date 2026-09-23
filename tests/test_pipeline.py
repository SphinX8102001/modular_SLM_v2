"""tests/test_pipeline.py — placeholder test suite (round 2)."""

import inspect
import pytest
import src
import src.config as config
import src.models as models
import src.router as router
import src.evaluator as evaluator


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
        """Calling generate() on HuggingFaceRunner or OllamaRunner raises NotImplementedError."""
        hf = models.HuggingFaceRunner("hf-id", "general", "1.5b")
        with pytest.raises(NotImplementedError):
            hf.generate("query", "general")

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
