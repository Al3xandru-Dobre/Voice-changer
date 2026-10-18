#!/usr/bin/env python3
"""Model contract v1 tests (Milestone 2 / Stage 02), Python side.

Runs the shared cases in tests/data/contracts/ (the C++ twin is
tests/test_model_contract.cpp) and checks the committed fixture manifest.

    uv run pytest tests/test_contract.py          # from python_training/
    python -I python_training/tests/test_contract.py   # any Python 3, no deps (CTest)

Exit status (script mode): 0 = all passed, 1 = a check failed.
Standard library only.
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

from fvml.contract import (  # noqa: E402
    ContractError, ModelContract, TensorSpec, algorithmic_latency, check_model,
    validate_io, validate_pair, validate_voices,
)

CASES = REPO / "tests" / "data" / "contracts"
FIXTURES = REPO / "tests" / "data" / "fixtures"

# Hand-computed in Documents/Milestone 2/02_model_contract_fixtures.md
# (Verification): hop + (L_enc + L_voc) x frame_samples + overlap_samples.
FIXTURE_LATENCY = {"identity_s1": 320, "identity_s2": 640, "identity_s4": 1280, "delay_s1": 640}


def load_cases(kind):
    paths = sorted((CASES / kind).glob("*.json"))
    assert paths, f"no {kind} cases under {CASES}"
    return [(p.stem, json.loads(p.read_text(encoding="utf-8"))) for p in paths]


def specs(items):
    return [TensorSpec.from_dict(d) for d in items]


def run_case(case):
    """Parse every model, cross-check graphs, validate the pair. Returns the
    pair's algorithmic latency (None for a single model)."""
    contracts = []
    for m in case["models"]:
        raw = m["contract"]
        c = ModelContract.from_json(raw if isinstance(raw, str) else json.dumps(raw))
        if "graph" in m:
            validate_io(c, specs(m["graph"]["inputs"]), specs(m["graph"]["outputs"]))
        contracts.append(c)
    if len(contracts) == 2:
        validate_pair(*contracts)
        return algorithmic_latency(*contracts)
    return None


def test_case_files_are_well_formed():
    for kind in ("valid", "invalid"):
        for name, case in load_cases(kind):
            assert isinstance(case.get("description"), str) and case["description"], name
            assert 1 <= len(case["models"]) <= 2, name
            assert ("error" in case) == (kind == "invalid"), name


def test_valid_cases_pass():
    for name, case in load_cases("valid"):
        latency = run_case(case)
        if "algorithmic_latency_samples" in case:
            assert latency == case["algorithmic_latency_samples"], (name, latency)
        for m in case["models"]:
            c = ModelContract.from_json(json.dumps(m["contract"]))
            assert ModelContract.from_json(c.to_json()) == c, name  # canonical round trip


def test_invalid_cases_are_rejected_with_their_message():
    for name, case in load_cases("invalid"):
        try:
            run_case(case)
        except ContractError as e:
            assert case["error"] in str(e), f"{name}: got {e!r}, want {case['error']!r}"
        else:
            raise AssertionError(f"{name}: accepted, want {case['error']!r}")


def test_contracts_built_in_code_are_validated():
    good = ModelContract(role="encoder", sample_rate=16000, frame_samples=320, frames_per_step=1,
                         lookahead_frames=0, content_kind="discrete", vocab=100, license="x")
    assert good.hop == 320 and good.content_name == "units"
    assert good.to_json().startswith('{"content_kind":"discrete","contract_version":1,')
    soft = {**good.__dict__, "content_kind": "soft", "vocab": 0, "content_dim": 8}
    for fields, message in (
        ({"sample_rate": "16000"}, "key 'sample_rate' must be an integer"),
        ({"frames_per_step": True}, "key 'frames_per_step' must be an integer"),
        ({"states": "conv"}, "key 'states' must be a list of strings"),
        ({"states": None}, "key 'states' must be a list of strings"),
        ({"cond_inputs": 5}, "key 'cond_inputs' must be a list of strings"),
        ({"license": "\ud800"}, "is not valid UTF-8"),
        ({"overlap_samples": 160}, "key 'overlap_samples' is not valid for role 'encoder'"),
        ({"content_dim": 8}, "key 'content_dim' is not valid for content_kind 'discrete'"),
        ({**soft, "vocab": 100}, "key 'vocab' is not valid for content_kind 'soft'"),
    ):
        contract = ModelContract(**{**good.__dict__, **fields})
        for check in (contract.validate, contract.to_json):    # to_json validates first
            try:
                check()
            except ContractError as e:
                assert message in str(e), (fields, str(e))
            else:
                raise AssertionError(f"accepted {fields}")
    # Lists from code become tuples: round trip and hashing work.
    listed = ModelContract(**{**good.__dict__, "states": ["a", "b"]})
    assert listed.states == ("a", "b") and ModelContract.from_json(listed.to_json()) == listed
    assert hash(listed) == hash(ModelContract.from_json(listed.to_json()))


def test_malformed_input_raises_only_contract_error():
    for fn in (lambda: ModelContract.from_json("[" * 100000),           # RecursionError inside json
               lambda: TensorSpec.from_dict({"name": "a", "shape": [1]}),
               lambda: TensorSpec.from_dict({"name": "a", "dtype": "f32"}),   # shape is required
               lambda: TensorSpec.from_dict({"name": 3, "dtype": "f32", "shape": [1]}),
               lambda: TensorSpec.from_dict({"name": "a", "dtype": "f32", "shape": [1.5]}),
               lambda: TensorSpec.from_dict(["audio"])):
        try:
            fn()
        except ContractError:
            pass
        else:
            raise AssertionError("accepted malformed input")
    assert TensorSpec.from_dict({"name": "s", "dtype": "f32", "shape": None}).shape is None


def test_check_model_reports_missing_metadata():
    try:
        check_model(None, [], [])
    except ContractError as e:
        assert str(e) == "model has no 'fluxvoice.contract' metadata"
    else:
        raise AssertionError("accepted a model without metadata")


def test_fixture_manifest():
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    by_file = {}
    for m in manifest["models"]:
        assert (FIXTURES / m["file"]).is_file(), m["file"]
        try:
            c = check_model(m["contract"], specs(m["graph"]["inputs"]), specs(m["graph"]["outputs"]))
        except ContractError as e:
            assert m["error"] and m["error"] in str(e), (m["file"], str(e))
            continue
        assert m["error"] is None, (m["file"], "accepted")
        by_file[m["file"]] = c
    for p in manifest["pairs"]:
        enc, voc = by_file[p["encoder"]], by_file[p["vocoder"]]
        validate_pair(enc, voc)
        latency = algorithmic_latency(enc, voc)
        assert latency == p["algorithmic_latency_samples"] == FIXTURE_LATENCY[p["name"]], p
    assert sorted(p["name"] for p in manifest["pairs"]) == sorted(FIXTURE_LATENCY)
    for bundle in manifest.get("bundles", []):
        voices = json.loads((FIXTURES / bundle["voices"]).read_text(encoding="utf-8"))
        validate_voices(voices, by_file[bundle["vocoder"]])


def test_voices_validation():
    voc = ModelContract(role="vocoder", sample_rate=16000, frame_samples=320, frames_per_step=1,
                        lookahead_frames=0, content_kind="discrete", vocab=100,
                        cond_frame_samples=320, cond_inputs=("logf0", "spk_emb"), spk_emb_dim=2,
                        license="x")
    voice = {"name": "lj", "display": "LJ", "embedding": [0.5, -0.5], "logf0_mean": 5.3,
             "logf0_std": 0.2, "energy_mean": -3.0, "energy_std": 1.1, "license": "PD",
             "source": "LJSpeech 1.1"}
    validate_voices({"schema": 1, "voices": [voice]}, voc)
    validate_voices({"schema": 1, "voices": [{**voice, "attribution": "LJSpeech, public domain"}]}, voc)
    without_source = {k: v for k, v in voice.items() if k != "source"}
    one = lambda **kw: {"schema": 1, "voices": [{**voice, **kw}]}  # noqa: E731
    for data, message in (
        ({"schema": 2, "voices": [voice]}, "unsupported voices.json schema 2"),
        ({"schema": True, "voices": [voice]}, "unsupported voices.json schema True"),
        ({"schema": 1.0, "voices": [voice]}, "unsupported voices.json schema 1.0"),
        ({"schema": 1, "voices": [voice], "bundle": "x"}, "voices.json: unknown key 'bundle'"),
        ([voice], "voices.json must be a JSON object"),
        ({"schema": 1, "voices": []}, "'voices' must be a non-empty list"),
        ({"schema": 1, "voices": ["lj"]}, "voices[0] must be an object"),
        ({"schema": 1, "voices": [voice, voice]}, "duplicate voice 'lj'"),
        ({"schema": 1, "voices": [without_source]}, "voices[0]: missing key 'source'"),
        (one(colour="red"), "unknown key 'colour'"),
        (one(embedding=[1.0]), "embedding has 1 values"),
        (one(embedding=[1.0, "x"]), "embedding must be a list of numbers"),
        (one(name="LJ Speech"), "name must be lowercase"),
        (one(name="LJ"), "name must be lowercase"),
        (one(name="lé"), "name must be lowercase"),
        (one(logf0_std=0), "logf0_std must be > 0"),
        (one(energy_std=0), "energy_std must be > 0"),
        (one(logf0_mean="5.3"), "logf0_mean must be a finite number"),
        (one(energy_mean=float("nan")), "energy_mean must be a finite number"),
        (one(display=3), "display must be a non-empty string"),
        (one(license=""), "license must be a non-empty string"),
        (one(source=None), "source must be a non-empty string"),
        (one(attribution=["x"]), "attribution must be a non-empty string"),
    ):
        try:
            validate_voices(data, voc)
        except ContractError as e:
            assert message in str(e), (message, str(e))
        else:
            raise AssertionError(f"accepted: {message}")


def main():
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # report every failure, then exit non-zero
            failed += 1
            print(f"FAIL  {name} -> {type(e).__name__}: {e}")
    print(f"[contract_py] {'all checks passed' if not failed else f'{failed} test(s) FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
