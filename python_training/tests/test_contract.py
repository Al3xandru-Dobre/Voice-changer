#!/usr/bin/env python3
"""Model contract v1 & v2 tests (Milestone 2 / Stage 02 & Milestone 3 / Stage 02).

Runs the shared cases in tests/data/contracts/ (the C++ twin is
tests/test_model_contract.cpp), checks the committed fixture manifest (Mode A
pairs + Mode B S2UT quartets + langs.json), and verifies langs.json parity with
frontend/assets/langs.default.json.

    uv run pytest tests/test_contract.py               # from python_training/
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
    ContractError, FrozenMap, ModelContract, TensorSpec, algorithmic_latency,
    check_model, s2ut_onset_latency_samples, validate_io, validate_langs,
    validate_pair, validate_s2ut_bundle, validate_s2ut_shared_io, validate_voices,
)

CASES = REPO / "tests" / "data" / "contracts"
FIXTURES = REPO / "tests" / "data" / "fixtures"
DEFAULT_LANGS = REPO / "frontend" / "assets" / "langs.default.json"

FIXTURE_LATENCY = {"identity_s1": 320, "identity_s2": 640, "identity_s4": 1280, "delay_s1": 640}
FIXTURE_S2UT_ONSET = {1: 1280, 3: 3840, 5: 6400, 10: 12800}


def load_cases(kind):
    paths = sorted((CASES / kind).glob("*.json"))
    assert paths, f"no {kind} cases under {CASES}"
    return [(p.stem, json.loads(p.read_text(encoding="utf-8"))) for p in paths]


def specs(items):
    return [TensorSpec.from_dict(d) for d in items]


def run_case(case):
    """Parse every model, cross-check graphs, validate Mode A pair (2 models)
    or Mode B quartet (4 models). Returns (mode_a_latency, s2ut_onset)."""
    contracts, graphs = [], []
    for m in case["models"]:
        raw = m["contract"]
        c = ModelContract.from_json(raw if isinstance(raw, str) else json.dumps(raw))
        if "graph" in m:
            ins, outs = specs(m["graph"]["inputs"]), specs(m["graph"]["outputs"])
            validate_io(c, ins, outs)
            graphs.append((ins, outs))
        else:
            graphs.append(None)
        contracts.append(c)
    if len(contracts) == 2:
        validate_pair(*contracts)
        return algorithmic_latency(*contracts), None
    if len(contracts) == 4:
        validate_s2ut_bundle(*contracts)
        if graphs[1] is not None and graphs[2] is not None:
            validate_s2ut_shared_io(contracts[1], graphs[1][1], contracts[2], graphs[2][0])
        wait_k = case.get("wait_k", 3)
        return None, s2ut_onset_latency_samples(*contracts, wait_k)
    return None, None


def test_case_files_are_well_formed():
    for kind in ("valid", "invalid"):
        for name, case in load_cases(kind):
            assert isinstance(case.get("description"), str) and case["description"], name
            assert len(case["models"]) in (1, 2, 4), name
            assert ("error" in case) == (kind == "invalid"), name


def test_valid_cases_pass():
    for name, case in load_cases("valid"):
        latency, s2ut_onset = run_case(case)
        if "algorithmic_latency_samples" in case:
            assert latency == case["algorithmic_latency_samples"], (name, latency)
        if "s2ut_onset_latency_samples" in case:
            assert s2ut_onset == case["s2ut_onset_latency_samples"], (name, s2ut_onset)
        for m in case["models"]:
            c = ModelContract.from_json(json.dumps(m["contract"]))
            round_tripped = ModelContract.from_json(c.to_json())
            assert round_tripped == c, name
            assert hash(round_tripped) == hash(c), name


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
        ({"latent_dim": 64}, "key 'latent_dim' requires contract_version 2"),
        ({"has_readiness_output": "yes"}, "key 'has_readiness_output' must be a boolean"),
        ({"langs": ["en"]}, "key 'langs' must be an object"),
    ):
        contract = ModelContract(**{**good.__dict__, **fields})
        for check in (contract.validate, contract.to_json):
            try:
                check()
            except ContractError as e:
                assert message in str(e), (fields, str(e))
            else:
                raise AssertionError(f"accepted {fields}")
    listed = ModelContract(**{**good.__dict__, "states": ["a", "b"]})
    assert listed.states == ("a", "b") and ModelContract.from_json(listed.to_json()) == listed
    assert hash(listed) == hash(ModelContract.from_json(listed.to_json()))


def test_v2_s2ut_contracts_and_onset_latency():
    langs = {"en": 0, "es": 1, "fr": 2, "de": 3}
    enc = ModelContract(contract_version=2, role="encoder", sample_rate=16000, frame_samples=320,
                        frames_per_step=4, lookahead_frames=1, content_kind="discrete",
                        vocab=1000, latent_dim=128, license="MIT")
    t_enc = ModelContract(contract_version=2, role="translator_enc", sample_rate=16000,
                          frame_samples=320, chunk_frames=4, latent_dim=128, max_src_steps=128,
                          shared_mem_states=["mem_k_0", "mem_v_0"], langs=langs,
                          has_readiness_output=True, states=["src_pos", "mem_k_0", "mem_v_0"],
                          license="MIT")
    t_dec = ModelContract(contract_version=2, role="translator_dec", sample_rate=16000,
                          frame_samples=320, chunk_frames=4, vocab=1000,
                          special_units={"bos": 1000, "eos": 1001, "pad": 1002},
                          max_src_steps=128, max_tgt_steps=128,
                          shared_mem_states=["mem_k_0", "mem_v_0"], langs=langs,
                          has_prosody_output=True, states=["tgt_pos"], license="MIT")
    voc = ModelContract(contract_version=2, role="vocoder", sample_rate=16000, frame_samples=320,
                        frames_per_step=4, lookahead_frames=0, content_kind="discrete",
                        vocab=1000, cond_frame_samples=320, cond_inputs=(), spk_emb_dim=0,
                        overlap_samples=0, license="MIT")
    assert isinstance(t_enc.langs, FrozenMap) and t_enc.langs["es"] == 1
    assert hash(t_enc) == hash(ModelContract.from_json(t_enc.to_json()))
    assert hash(t_dec) == hash(ModelContract.from_json(t_dec.to_json()))
    validate_s2ut_bundle(enc, t_enc, t_dec, voc)
    for k, want in ((1, 1600), (3, 4160), (5, 6720), (10, 13120)):
        assert s2ut_onset_latency_samples(enc, t_enc, t_dec, voc, k) == want
    for bad_k in (0, 11, True, 3.0):
        try:
            s2ut_onset_latency_samples(enc, t_enc, t_dec, voc, bad_k)
        except ContractError as e:
            assert "wait_k must be between 1 and 10" in str(e)
        else:
            raise AssertionError(f"accepted wait_k={bad_k!r}")


def test_malformed_input_raises_only_contract_error():
    for fn in (lambda: ModelContract.from_json("[" * 100000),
               lambda: TensorSpec.from_dict({"name": "a", "shape": [1]}),
               lambda: TensorSpec.from_dict({"name": "a", "dtype": "f32"}),
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
    by_file, graphs = {}, {}
    for m in manifest["models"]:
        assert (FIXTURES / m["file"]).is_file(), m["file"]
        ins, outs = specs(m["graph"]["inputs"]), specs(m["graph"]["outputs"])
        graphs[m["file"]] = (ins, outs)
        try:
            c = check_model(m["contract"], ins, outs)
        except ContractError as e:
            assert m["error"] and m["error"] == str(e), (m["file"], str(e))
            continue
        assert m["error"] is None, (m["file"], "accepted")
        by_file[m["file"]] = c
    for p in manifest["pairs"]:
        enc, voc = by_file[p["encoder"]], by_file[p["vocoder"]]
        validate_pair(enc, voc)
        latency = algorithmic_latency(enc, voc)
        assert latency == p["algorithmic_latency_samples"] == FIXTURE_LATENCY[p["name"]], p
    assert sorted(p["name"] for p in manifest["pairs"]) == sorted(FIXTURE_LATENCY)

    assert len(manifest.get("s2ut_quartets", [])) == 3
    for q in manifest["s2ut_quartets"]:
        enc = by_file[q["encoder"]]
        t_enc = by_file[q["translator_enc"]]
        t_dec = by_file[q["translator_dec"]]
        voc = by_file[q["vocoder"]]
        validate_s2ut_bundle(enc, t_enc, t_dec, voc)
        validate_s2ut_shared_io(t_enc, graphs[q["translator_enc"]][1],
                                t_dec, graphs[q["translator_dec"]][0])
        for k, want in FIXTURE_S2UT_ONSET.items():
            got = s2ut_onset_latency_samples(enc, t_enc, t_dec, voc, k)
            assert got == want == q["onset_latency_samples"][str(k)], (q["name"], k, got)

    for bundle in manifest.get("bundles", []):
        voices = json.loads((FIXTURES / bundle["voices"]).read_text(encoding="utf-8"))
        validate_voices(voices, by_file[bundle["vocoder"]])
        if "langs" in bundle:
            langs = json.loads((FIXTURES / bundle["langs"]).read_text(encoding="utf-8"))
            validate_langs(langs, by_file[bundle["translator"]])


def test_voices_and_langs_validation():
    voc = ModelContract(role="vocoder", sample_rate=16000, frame_samples=320, frames_per_step=1,
                        lookahead_frames=0, content_kind="discrete", vocab=100,
                        cond_frame_samples=320, cond_inputs=("logf0", "spk_emb"), spk_emb_dim=2,
                        license="x")
    voice = {"name": "lj", "display": "LJ", "embedding": [0.5, -0.5], "logf0_mean": 5.3,
             "logf0_std": 0.2, "energy_mean": -3.0, "energy_std": 1.1, "license": "PD",
             "source": "LJSpeech 1.1"}
    validate_voices({"schema": 1, "voices": [voice]}, voc)
    validate_voices({"schema": 1, "voices": [{**voice, "attribution": "LJSpeech, public domain",
                                              "lang": "en"}]}, voc)
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
        (one(lang="EN"), "invalid language code 'EN'"),
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

    # langs.json validation + parity with frontend/assets/langs.default.json
    default_langs = json.loads(DEFAULT_LANGS.read_text(encoding="utf-8"))
    validate_langs(default_langs)
    for bad, msg in (
        ({"schema": True, "langs": default_langs["langs"]}, "unsupported langs.json schema"),
        ({"schema": 1, "langs": []}, "'langs' must be a non-empty list"),
        ({"schema": 1, "langs": default_langs["langs"], "extra": 1}, "unknown key 'extra'"),
        ({"schema": 1, "langs": [{"code": "EN", "id": 0, "name": "English"}]},
         "'code' must be a lowercase language tag"),
        ({"schema": 1, "langs": [{"code": "en", "id": 0, "name": "English"},
                                 {"code": "es", "id": 0, "name": "Spanish"}]},
         "duplicate id for 'es'"),
        ({**default_langs, "default_wait_k": {"en->en": 3}}, "invalid language pair 'en->en'"),
        ({**default_langs, "default_wait_k": {"es->en": 0}},
         "default_wait_k['es->en'] must be an integer in [1, 10]"),
    ):
        try:
            validate_langs(bad)
        except ContractError as e:
            assert msg in str(e), (msg, str(e))
        else:
            raise AssertionError(f"accepted bad langs.json: {msg}")


def main():
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name} -> {type(e).__name__}: {e}")
    print(f"[contract_py] {'all checks passed' if not failed else f'{failed} test(s) FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
