"""Where things live: the repo, the eval lists and the FLUXVOICE_DATA data root.

Every path stored in a manifest or an eval list is relative to the data root
(forward slashes), so lists stay valid on any machine and OS.
"""
import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAINING_DIR = REPO_ROOT / "python_training"
EVAL_LISTS = TRAINING_DIR / "eval_lists"


def data_root():
    """$FLUXVOICE_DATA, default <repo>/data/ (git-ignored)."""
    env = os.environ.get("FLUXVOICE_DATA")
    return pathlib.Path(env).expanduser().resolve() if env else REPO_ROOT / "data"


def cache_dir(*parts):
    path = data_root().joinpath("cache", *parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve(path):
    """A manifest/list path (relative to the data root) or an absolute path."""
    p = pathlib.Path(path)
    return p if p.is_absolute() else data_root() / p


def relative(path):
    """Inverse of resolve() for files under the data root."""
    return pathlib.Path(path).resolve().relative_to(data_root().resolve()).as_posix()


def configure_model_caches():
    """Keep downloaded model weights next to the data unless the user chose a
    Hugging Face cache already (HF_HOME). Call before importing transformers."""
    os.environ.setdefault("HF_HOME", str(cache_dir("huggingface")))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # Windows without Developer Mode
