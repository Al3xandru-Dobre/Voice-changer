# BUG-008 — Stage 01's `.gitignore` plan would hide `tests/data/`, and its other patterns are still missing

> [Bug index](00_index.md)

| | |
|---|---|
| **Tag** | `BUG-008` |
| **Name** | *Data-root trap* — the rule Stage 01 prescribes (`data/`) also ignores the committed test fixtures |
| **Reported** | 2026-10-18 |
| **Severity** | Medium (latent trap + partial gap; the imminent hazard was removed during the review, see below) |
| **Status** | ☑ fixed — verified 2026-10-18 15:30, see [Resolution](#resolution-verified-2026-10-18); follow-ups closed, regression test added 17:00 |
| **Component** | `.gitignore`; [Milestone 2 / Stage 01](../Milestone%202/01_python_toolchain_eval_harness.md) "Data root" task |
| **Blocks** | Stage 01 definition of done (data root); a clean-clone CI run of every contract test |

## Summary

Three facts, all verified on 2026-10-18:

1. **Resolved during the review.** When first checked (shortly before 14:35),
   `<repo>/data/cache/models/utmos22_strong_step7459_v1.pt` (**411 179 119
   bytes**) was untracked and *not* ignored: `git add -A --dry-run data` staged
   it. `data/.gitignore` (`*`, modified 14:35:39) appeared afterwards, and
   `git check-ignore -v` now reports `data/.gitignore:2:*` for the file. The
   weights are now safe.
2. **Still open: the trap.** The Stage 01 task says to add `data/` to the root
   `.gitignore`. Without a leading slash Git matches that name at any depth, so
   it would also ignore `tests/data/` — the contract cases and fixture models
   (`tests/data/contracts/**`, `tests/data/fixtures/**`) that the Python and C++
   contract tests read. With `data/.gitignore` in place the root rule is not
   needed at all, but the stage text still prescribes it.
3. **Still open: the other patterns.** Stage 01 also lists `checkpoints/`,
   `runs/`, `*.pt`, `*.ckpt`, `*.onnx.data`. None is in `.gitignore`. Scratch
   files named `checkpoints/x.ckpt`, `runs/exp1/events.out`, `model.pt`,
   `weights/best.ckpt` and `big.onnx.data` were all reported as **not ignored**.

## Steps to reproduce

The trap, in a scratch repo (never in the real one):

```bash
git init -q . && mkdir -p tests/data/fixtures data
touch tests/data/fixtures/manifest.json data/y.wav
printf 'data/\n' > .gitignore
git check-ignore -v tests/data/fixtures/manifest.json data/y.wav   # BOTH are ignored
git add -A --dry-run                                                # stages only .gitignore
printf '/data/\n' > .gitignore
git check-ignore -v tests/data/fixtures/manifest.json data/y.wav   # only data/y.wav is ignored
```

The missing patterns, in the real repo (create and delete a scratch file):

```bash
touch model.pt && git check-ignore -q model.pt; echo $?     # 1 = not ignored
rm model.pt
```

## Expected

`checkpoints/`, `runs/` and `data/` at the repo root, plus `*.pt`, `*.ckpt` and
`*.onnx.data` anywhere, are ignored; `tests/data/` stays tracked.

## Actual

| State on 2026-10-18 | Data-root files | `tests/data/**` | `*.pt` / `*.ckpt` / `checkpoints/` / `runs/` |
|---|---|---|---|
| Now (`data/.gitignore` = `*`, root unchanged) | ignored | tracked | **not ignored** |
| Stage 01 text applied literally (`data/`) | ignored | **silently ignored** | still not ignored |
| Stage 01 text with `/data/` | ignored | tracked | still not ignored |

The second row is the dangerous one. Nothing fails locally: the files stay on
disk and every test passes; the fixtures are simply absent from the commit, and
`test_contract`, `test_model_contract` and `test_fixtures` fail on the first
clean clone or CI run.

## Root cause

- gitignore semantics: a pattern with no slash, or only a trailing slash,
  matches a directory *name* at every depth; a leading `/` anchors it to the
  repo root.
- Stage 02 put committed test data in a directory also called `data`, after
  Stage 01's task text was written.
- Stage 01's `.gitignore` task has not been done; the nested `data/.gitignore`
  covers only the directory it sits in.

## Fix

Add to the root `.gitignore` (anchored directories, global file patterns):

```gitignore
# Milestone 2 training output and weights (data/ has its own .gitignore)
/checkpoints/
/runs/
*.pt
*.ckpt
*.onnx.data
```

and correct the Stage 01 task to say `/data/` (or "`data/.gitignore` containing
`*`"), never `data/`. Add a regression check so the trap cannot return: a
CTest/CI step that runs `git check-ignore -q tests/data/contracts/valid/pair_a0_s1.json`
and requires exit 1.

## Workaround

Keep large artefacts under `data/` (ignored), not in `checkpoints/` or `runs/`,
until the root rules land. When someone does edit the root `.gitignore`: write
`/data/`, never `data/`.

## Context

Found while reviewing Milestone 2 stages 1–2. The repo was being edited
concurrently (a `Devin` process was running; `src/pipeline/ModelContract.hpp`,
`prepare_data.py` and `data/.gitignore` appeared mid-review), so this report
does not touch `.gitignore` itself.

## Resolution (verified 2026-10-18)

The root `.gitignore` now has `/data/`, `/checkpoints/`, `/runs/`,
`/python_training/{checkpoints,runs}/`, `*.pt`, `*.ckpt`, `*.onnx.data` and
`models/*/`, with a comment explaining why the directories are anchored.
`.gitattributes` marks `*.onnx` and `*.npy` binary and forces LF on
`tests/data/**/*.json`, `python_training/eval_lists/*.tsv` and `uv.lock`.
The Stage 01 task now carries a note pointing at this bug.

Checked with `git check-ignore --no-index` on paths that do not exist (so no
scratch files were created this time):

| Path | Result |
|---|---|
| `data/cache/models/x.pt`, `data/manifests/a.jsonl` | ignored |
| `checkpoints/x.ckpt`, `runs/exp1/events.out`, `python_training/runs/e/x`, `python_training/checkpoints/c.pt` | ignored |
| `model.pt`, `weights/best.ckpt`, `big.onnx.data` | ignored |
| `models/some_bundle/encoder_s1.onnx` | ignored |
| `tests/data/contracts/valid/pair_a0_s1.json`, `tests/data/fixtures/manifest.json`, `tests/data/fixtures/identity/encoder_s1.onnx`, `python_training/eval_lists/test_clean_200.tsv` | **not ignored** (tracked normally) |

`git check-attr` confirms `binary` on the fixtures and `text eol=lf` on the JSON.

Follow-ups, as first noted (later status in brackets):

1. **`models/*/` ignores everything in a bundle, including `MODEL_CARD.md` and
   `voices.json`.** I raised it because Stage 06 expects a per-bundle
   `MODEL_CARD.md` with training data and licences. *[Closed as intended: the
   Stage 06 spec says bundles are generated and git-ignored, and
   `models/README.md` says only the README is tracked, so a bundle's card travels
   with the bundle. If Stage 06's owner later wants cards in git, ignoring the
   contents instead (`models/*/*`, then `!models/*/MODEL_CARD.md` and
   `!models/*/voices.json`) is the fix; Git cannot re-include a file under an
   ignored directory.]*
2. **The Stage 01 task text still said `data/`**, with a note beneath it.
   *[Done: the text now reads `/data/`, `/checkpoints/`, `/runs/` and explains
   why; I re-read it.]*

I also suggested a regression check (`git check-ignore -q tests/data/...` must
exit 1). *[Done, 2026-10-18 17:00: CTest `gitignore_rules` runs
`tests/check_gitignore.cmake`, which asserts the tracked and ignored paths in
the table above with `git check-ignore --no-index`. It is registered when Git
and a `.git` directory exist, so it runs on all four CI platforms. Checked
against a mutation: a scratch repo whose `.gitignore` has `data/` instead of
`/data/` fails it, naming the four `tests/data/` paths.]*

