"""Dataset registry and preparation (Stage 01): download, verify, extract,
resample to 16 kHz mono, write JSONL manifests.

Data root ($FLUXVOICE_DATA, default <repo>/data/, git-ignored):

  archives/<file>              downloads (--delete-archives removes them after extraction)
  raw/<corpus>/...             extracted originals; 16 kHz corpora are used in place
  16k/<dataset>/...            resampled copies of non-16 kHz corpora (FLAC, 24-bit)
  manifests/<dataset>.jsonl    one row per utterance/clip, paths relative to the data root
  manifests/summary.json       hours per dataset, subset and speaker
  .state/                      verification/extraction stamps, so reruns skip finished work

Checksums: OpenSLR and Edinburgh DataShare publish MD5 only, and Hi-Fi TTS
publishes nothing. SHA256 pins therefore live in python_training/datasets.lock.json,
pre-seeded from torchaudio's dataset tables. An archive without a pin is checked
against its published MD5 (if any) and its SHA256 is recorded on first download
("trust on first use"); commit the updated lock so every later download is
pinned.

Resampled copies are stored as 24-bit FLAC rather than float WAV: about a third
of the disk, with quantisation noise near -144 dBFS. fvml.audio.load() returns
float32 either way.
"""
import csv
import hashlib
import json
import os
import pathlib
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field

from . import audio, paths

LOCK_FILE = paths.TRAINING_DIR / "datasets.lock.json"
OPENSLR_MIRRORS = ("https://www.openslr.org/resources", "https://openslr.elda.org/resources",
                   "https://openslr.trmal.net/resources", "https://openslr.magicdatatech.com/resources")
CHUNK = 1 << 20

# Hi-Fi TTS readers (arXiv 2104.01497, Table 2). Only 92, 6097 and 9017 have
# "clean" (SNR >= 40 dB) data; the rest are "other" (>= 32 dB).
HIFITTS_SPEAKERS = {
    "92": ("F", "Cori Samuel"), "6097": ("M", "Phil Benson"), "9017": ("M", "John Van Stan"),
    "6670": ("M", "Mike Pelton"), "6671": ("M", "Tony Oliva"), "8051": ("F", "Maria Kasper"),
    "9136": ("F", "Helen Taylor"), "11614": ("F", "Sylviamb"), "11697": ("F", "Celine Major"),
    "12787": ("F", "LikeManyWaters"),
}
# Two women, two men: the three clean readers plus the strongest "other" female.
HIFITTS_DEFAULT = ("92", "8051", "6097", "9017")


class PrepareError(RuntimeError):
    pass


def openslr(slr, name):
    return tuple(f"{m}/{slr}/{name}" for m in OPENSLR_MIRRORS)


@dataclass(frozen=True)
class Archive:
    file: str
    urls: tuple
    size: int
    md5: str | None = None       # published checksum, if the source has one


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    group: str                   # "eval" | "core" | "optional"
    archives: tuple
    builder: str                 # name of a build_* function below
    raw_dir: str                 # directory under raw/ the archives extract into
    licence: str
    arg: str = ""                # builder argument (LibriSpeech subset)


def _ls(subset, size, md5, group):
    return Dataset(f"librispeech-{subset}", f"LibriSpeech {subset}", group,
                   (Archive(f"{subset}.tar.gz", openslr(12, f"{subset}.tar.gz"), size, md5),),
                   "build_librispeech", "", "CC BY 4.0", subset)


DATASETS = {d.key: d for d in (
    _ls("test-clean", 346_663_984, "32fa31d27d2e1cad72775fee3f4849a9", "eval"),
    _ls("dev-clean", 337_926_286, "42e2234ba48799c1f50f24a7926300a1", "eval"),
    _ls("train-clean-100", 6_387_309_499, "2a93770f6d5c6c964bc36631d331a522", "core"),
    _ls("train-clean-360", 23_049_477_885, "c0e676e450a7ff2f54aeade5171606fa", "core"),
    Dataset("ljspeech", "LJSpeech 1.1", "core",
            (Archive("LJSpeech-1.1.tar.bz2",
                     ("https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",), 2_748_572_632),),
            "build_ljspeech", "", "Public domain"),
    Dataset("hifitts", "Hi-Fi TTS v0 (selected speakers)", "core",
            (Archive("hi_fi_tts_v0.tar.gz", openslr(109, "hi_fi_tts_v0.tar.gz"), 41_352_291_414),),
            "build_hifitts", "", "CC BY 4.0"),
    Dataset("musan", "MUSAN", "core",
            (Archive("musan.tar.gz", openslr(17, "musan.tar.gz"), 11_086_114_085,
                     "0c472d4fc0c5141eca47ad1ffeb2a7df"),),
            "build_musan", "", "CC BY 4.0 (corpus); per-file licences in each subset's LICENSE"),
    Dataset("rirs", "RIRS_NOISES (OpenSLR 28; includes the OpenSLR 26 simulated RIRs)", "core",
            (Archive("rirs_noises.zip", openslr(28, "rirs_noises.zip"), 1_311_166_223,
                     "e6f48e257286e05de56413b4779d8ffb"),),
            "build_rirs", "", "Apache 2.0"),
    Dataset("vctk", "VCTK 0.92 (mic1)", "optional",
            (Archive("VCTK-Corpus-0.92.zip",
                     ("https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip",),
                     11_747_302_977, "8a6ba2946b36fcbef0212cad601f4bfa"),),
            "build_vctk", "VCTK-Corpus-0.92", "CC BY 4.0"),
)}
GROUPS = {"eval": ("eval",), "core": ("eval", "core"), "all": ("eval", "core", "optional")}


def select(spec):
    """'eval' | 'core' | 'all' | comma-separated keys -> [Dataset]."""
    if spec in GROUPS:
        return [d for d in DATASETS.values() if d.group in GROUPS[spec]]
    keys = [k.strip() for k in spec.split(",") if k.strip()]
    unknown = [k for k in keys if k not in DATASETS]
    if unknown:
        raise PrepareError(f"unknown dataset(s): {', '.join(unknown)} (see --list)")
    return [DATASETS[k] for k in keys]


@dataclass
class Context:
    root: pathlib.Path
    jobs: int = max(1, (os.cpu_count() or 2) // 2)
    hifitts_speakers: tuple = HIFITTS_DEFAULT
    delete_archives: bool = False
    force_manifests: bool = False
    lock_file: pathlib.Path = LOCK_FILE
    log: object = print
    _lock: dict = field(default=None, repr=False)

    def dir(self, *parts):
        p = self.root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        ignore = self.root / ".gitignore"
        if not ignore.is_file():  # the default root sits inside the repo
            ignore.write_text("# Dataset root (FLUXVOICE_DATA): never commit anything here.\n*\n",
                              encoding="utf-8")
        return p

    @property
    def raw(self):
        return self.dir("raw")

    def rel(self, path):
        return pathlib.Path(path).resolve().relative_to(self.root.resolve()).as_posix()

    # -- SHA256 lock ---------------------------------------------------------
    def lock(self):
        if self._lock is None:
            self._lock = json.loads(self.lock_file.read_text(encoding="utf-8")) \
                if self.lock_file.is_file() else {"sha256": {}}
        return self._lock["sha256"]

    def pin(self, file, sha256):
        self.lock()[file] = sha256
        self._lock["sha256"] = dict(sorted(self._lock["sha256"].items()))
        self.lock_file.write_text(json.dumps(self._lock, indent=2) + "\n", encoding="utf-8")


# -- download & verify -----------------------------------------------------
def download(ctx, archive):
    dest = ctx.dir("archives") / archive.file
    if dest.is_file() and dest.stat().st_size == archive.size:
        return dest
    part = dest.with_name(dest.name + ".part")
    errors = []
    for url in rank_mirrors(ctx, archive.urls):
        try:
            _fetch(ctx, url, part, archive.size)
            part.replace(dest)
            return dest
        except (urllib.error.URLError, OSError, PrepareError) as e:
            errors.append(f"{url}: {e}")
            ctx.log(f"  {url}: {e}")
    raise PrepareError(f"{archive.file}: every source failed:\n  " + "\n  ".join(errors))


def rank_mirrors(ctx, urls, probe_bytes=2 << 20):
    """Fastest first, by timing a 2 MB range request to each (OpenSLR mirror
    speeds differ by 10x or more depending on where you are)."""
    if len(urls) < 2:
        return list(urls)
    speeds = {}
    for url in urls:
        req = urllib.request.Request(url, headers={"User-Agent": "fluxvoice-prepare-data/1",
                                                   "Range": f"bytes=0-{probe_bytes - 1}"})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                n = len(r.read(probe_bytes))
            speeds[url] = n / max(time.monotonic() - t0, 1e-3)
        except (urllib.error.URLError, OSError):
            speeds[url] = 0.0
    ranked = sorted(urls, key=lambda u: -speeds[u])
    ctx.log("  mirrors: " + ", ".join(f"{u.split('/')[2]} {speeds[u] / 1e6:.1f} MB/s" for u in ranked))
    return ranked


def _fetch(ctx, url, part, size):
    have = part.stat().st_size if part.is_file() else 0
    if have >= size:
        part.unlink()
        have = 0
    headers = {"User-Agent": "fluxvoice-prepare-data/1"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and getattr(r, "status", None) != 206:
            have = 0                                   # server ignored the range: restart
        ctx.log(f"  downloading {url}" + (f" (resuming at {have / 1e9:.2f} GB)" if have else ""))
        with open(part, "ab" if have else "wb") as f:
            done, t0, last = have, time.monotonic(), 0.0
            while chunk := r.read(CHUNK):
                f.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if now - last > 10:
                    rate = (done - have) / max(now - t0, 1e-3)
                    ctx.log(f"    {done / 1e9:7.2f} / {size / 1e9:.2f} GB  {rate / 1e6:5.1f} MB/s")
                    last = now
    got = part.stat().st_size
    if got != size:
        raise PrepareError(f"downloaded {got} bytes, expected {size}")


def file_hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 * CHUNK):
            md5.update(chunk)
            sha.update(chunk)
    return md5.hexdigest(), sha.hexdigest()


def verify(ctx, archive, path):
    """Published MD5 + pinned SHA256. Returns the SHA256."""
    st = path.stat()
    stamp = ctx.dir(".state") / f"{archive.file}.verified.json"
    pinned = ctx.lock().get(archive.file)
    if stamp.is_file():
        s = json.loads(stamp.read_text(encoding="utf-8"))
        if s["size"] == st.st_size and s["mtime_ns"] == st.st_mtime_ns and s["sha256"] == pinned:
            return pinned
    ctx.log(f"  verifying {archive.file} ({st.st_size / 1e9:.2f} GB)")
    md5, sha = file_hashes(path)
    problem = None
    if st.st_size != archive.size:
        problem = f"size {st.st_size}, expected {archive.size}"
    elif archive.md5 and md5 != archive.md5:
        problem = f"MD5 {md5}, published {archive.md5}"
    elif pinned and sha != pinned:
        problem = f"SHA256 {sha}, pinned {pinned} in {ctx.lock_file.name}"
    if problem:
        path.replace(path.with_name(path.name + ".bad"))   # a rerun downloads it again
        raise PrepareError(f"{archive.file}: checksum failed ({problem}); moved to {path.name}.bad")
    if not pinned:
        ctx.pin(archive.file, sha)
        ctx.log(f"  pinned SHA256 {sha} for {archive.file} (first download) — "
                f"commit {ctx.lock_file.name}")
    stamp.write_text(json.dumps({"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha}),
                     encoding="utf-8")
    return sha


# -- extraction --------------------------------------------------------------
def extract(path, dest, keep=None):
    """Extract an archive into dest. keep(name) -> bool filters members."""
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    if path.name.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if keep is None or keep(info.filename):
                    z.extract(info, dest)          # zipfile strips absolute and ".." parts
                    n += 1
    else:
        with tarfile.open(path, mode="r|*") as tf:  # one sequential pass, even for 41 GB
            for m in tf:
                if keep is None or keep(m.name):
                    tf.extract(m, dest, filter="data")
                    n += 1
    return n


def stream_tar_metadata(ctx, archive, stop_prefix, dest):
    """Extract the leading non-audio members of a remote tar without
    downloading the rest (Hi-Fi TTS keeps its manifests first)."""
    dest.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(archive.urls[0], headers={"User-Agent": "fluxvoice-prepare-data/1"})
    n = 0
    with urllib.request.urlopen(req, timeout=60) as r, tarfile.open(fileobj=r, mode="r|*") as tf:
        for m in tf:
            if m.name.startswith(stop_prefix):
                break
            tf.extract(m, dest, filter="data")
            n += 1
    ctx.log(f"  {archive.file}: extracted {n} metadata members (archive not downloaded)")
    return n


# -- manifest rows -----------------------------------------------------------
def row(ctx, utt_id, path, duration, speaker, dataset, subset, **extra):
    r = {"utt_id": utt_id, "path": ctx.rel(path), "duration": round(float(duration), 3),
         "speaker": speaker, "dataset": dataset, "subset": subset}
    r.update({k: v for k, v in extra.items() if v is not None})
    return r


def _resample_job(src, dst):
    """Worker: src -> 16 kHz mono 24-bit FLAC at dst (skipped if present)."""
    dst = pathlib.Path(dst)
    if not dst.is_file():
        x = audio.load(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.stem + ".tmp.flac")
        audio.save(tmp, x, subtype="PCM_24")
        tmp.replace(dst)
    return audio.info(dst)[0]


def resample_all(ctx, pairs):
    """[(src, dst)] -> [duration], in order, using ctx.jobs processes."""
    if not pairs:
        return []
    with ProcessPoolExecutor(ctx.jobs) as ex:
        out = list(ex.map(_resample_job, [str(s) for s, _ in pairs], [str(d) for _, d in pairs],
                          chunksize=32))
    return out


def durations(ctx, files):
    with ThreadPoolExecutor(max(4, ctx.jobs)) as ex:
        return list(ex.map(lambda p: audio.info(p)[0], files))


def parse_librispeech_speakers(path):
    genders = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(";") or "|" not in line:
            continue
        fields = [f.strip() for f in line.split("|")]
        genders[fields[0]] = fields[1]
    return genders


def build_librispeech(ctx, ds):
    root = ctx.raw / "LibriSpeech"
    genders = parse_librispeech_speakers(root / "SPEAKERS.TXT")
    items = []
    for trans in sorted((root / ds.arg).glob("*/*/*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt, _, text = line.strip().partition(" ")
            if utt:
                items.append((utt, trans.parent / f"{utt}.flac", text))
    durs = durations(ctx, [p for _, p, _ in items])
    return [row(ctx, utt, p, d, utt.split("-")[0], "librispeech", ds.arg,
                gender=genders.get(utt.split("-")[0]), text=text)
            for (utt, p, text), d in zip(items, durs)]


def build_ljspeech(ctx, ds):
    root = ctx.raw / "LJSpeech-1.1"
    with open(root / "metadata.csv", encoding="utf-8", newline="") as f:
        meta = [r for r in csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE) if r]
    out = ctx.dir("16k", "ljspeech")
    pairs = [(root / "wavs" / f"{r[0]}.wav", out / f"{r[0]}.flac") for r in meta]
    durs = resample_all(ctx, pairs)
    return [row(ctx, r[0], dst, d, "lj", "ljspeech", "all", gender="F", text=r[-1])
            for r, (_, dst), d in zip(meta, pairs, durs)]


def hifitts_keep(speakers):
    prefixes = tuple(f"hi_fi_tts_v0/audio/{s}_" for s in speakers)
    return lambda name: not name.startswith("hi_fi_tts_v0/audio/") or name.startswith(prefixes)


def hifitts_manifest_rows(root, speakers):
    """(speaker, quality, split, entry) for every manifest line of `speakers`."""
    for spk in speakers:
        for path in sorted(root.glob(f"{spk}_manifest_*_*.json")):
            _, _, quality, split = path.stem.split("_")
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    yield spk, quality, split, json.loads(line)


def hifitts_utt_id(spk, audio_filepath):
    p = pathlib.PurePosixPath(audio_filepath)
    return f"hifitts_{spk}_{p.parent.name}_{p.stem}"


def build_hifitts(ctx, ds):
    root = ctx.raw / "hi_fi_tts_v0"
    out = ctx.dir("16k", "hifitts")
    entries = list(hifitts_manifest_rows(root, ctx.hifitts_speakers))
    pairs = [(root / e["audio_filepath"], out / pathlib.PurePosixPath(e["audio_filepath"])
              .relative_to("audio").with_suffix(".flac")) for _, _, _, e in entries]
    durs = resample_all(ctx, pairs)
    return [row(ctx, hifitts_utt_id(spk, e["audio_filepath"]), dst, d, f"hifitts_{spk}", "hifitts",
                f"{quality}_{split}", gender=HIFITTS_SPEAKERS.get(spk, ("?",))[0],
                text=e.get("text_normalized", e.get("text")))
            for (spk, quality, split, e), (_, dst), d in zip(entries, pairs, durs)]


def build_musan(ctx, ds):
    root = ctx.raw / "musan"
    files = sorted(root.rglob("*.wav"))
    durs = durations(ctx, files)
    rows = []
    for p, d in zip(files, durs):
        rel = p.relative_to(root).parts
        licence = p.parent / "LICENSE"
        rows.append(row(ctx, p.stem, p, d, None, "musan", rel[0], source=rel[1] if len(rel) > 2 else None,
                        licence_file=ctx.rel(licence) if licence.is_file() else None))
    return rows


def build_rirs(ctx, ds):
    root = ctx.raw / "RIRS_NOISES"
    rows = []
    files = sorted(root.rglob("*.wav"))
    infos = [audio.info(p) for p in files]
    for p, (d, sr, ch) in zip(files, infos):
        top = p.relative_to(root).parts[0]
        if top == "simulated_rirs":
            subset = "simulated_rir"
        elif top == "pointsource_noises":
            subset = "pointsource_noise"
        else:
            subset = "isotropic_noise" if "noise" in p.stem else "real_rir"
        if sr != audio.SAMPLE_RATE:
            raise PrepareError(f"{p}: {sr} Hz, expected 16 kHz")
        # Simulated RIR file names repeat across rooms: the relative path is the id.
        rows.append(row(ctx, p.relative_to(root).with_suffix("").as_posix(), p, d, None, "rirs",
                        subset, channels=ch))
    return rows


def build_vctk(ctx, ds):
    root = ctx.raw / ds.raw_dir
    genders, accents = {}, {}
    for line in (root / "speaker-info.txt").read_text(encoding="utf-8").splitlines()[1:]:
        f = line.split()
        if len(f) >= 4:
            genders[f[0]], accents[f[0]] = f[2], f[3]
    out = ctx.dir("16k", "vctk")
    items = []
    for txt in sorted((root / "txt").glob("*/*.txt")):
        spk, utt = txt.parent.name, txt.stem
        src = root / "wav48_silence_trimmed" / spk / f"{utt}_mic1.flac"
        if src.is_file():
            items.append((spk, utt, src, out / spk / f"{utt}.flac",
                          txt.read_text(encoding="utf-8").strip()))
    durs = resample_all(ctx, [(s, d) for _, _, s, d, _ in items])
    return [row(ctx, utt, dst, d, spk, "vctk", "mic1", gender=genders.get(spk),
                accent=accents.get(spk), text=text)
            for (spk, utt, _, dst, text), d in zip(items, durs)]


EXTRACT_FILTERS = {
    "hifitts": lambda ctx: hifitts_keep(ctx.hifitts_speakers),
    "vctk": lambda ctx: (lambda name: not name.endswith("_mic2.flac")),
}


# -- driver ------------------------------------------------------------------
def prepare(ctx, ds):
    """Download, verify, extract and build the manifest for one dataset."""
    ctx.log(f"[{ds.key}] {ds.title}")
    state = ctx.dir(".state")
    stamp = state / f"{ds.key}.extracted.json"
    pins = {a.file: ctx.lock().get(a.file) for a in ds.archives}
    extracted = stamp.is_file() and json.loads(stamp.read_text(encoding="utf-8")) == {
        "archives": pins, "filter": _filter_tag(ctx, ds)}
    if not extracted:
        dest = ctx.raw / ds.raw_dir if ds.raw_dir else ctx.raw
        keep = EXTRACT_FILTERS[ds.key](ctx) if ds.key in EXTRACT_FILTERS else None
        for a in ds.archives:
            path = download(ctx, a)
            verify(ctx, a, path)
            ctx.log(f"  extracting {a.file}")
            n = extract(path, dest, keep)
            ctx.log(f"  {n} members")
        pins = {a.file: ctx.lock().get(a.file) for a in ds.archives}
        stamp.write_text(json.dumps({"archives": pins, "filter": _filter_tag(ctx, ds)}),
                         encoding="utf-8")
        if ctx.delete_archives:
            for a in ds.archives:
                (ctx.root / "archives" / a.file).unlink(missing_ok=True)
    else:
        ctx.log("  already extracted and verified")

    manifest = ctx.dir("manifests") / f"{ds.key}.jsonl"
    mstamp = state / f"{ds.key}.manifest.json"
    if manifest.is_file() and mstamp.is_file() and not ctx.force_manifests and \
            json.loads(mstamp.read_text(encoding="utf-8")).get("filter") == _filter_tag(ctx, ds):
        ctx.log(f"  manifest up to date: {ctx.rel(manifest)}")
        return manifest
    rows = globals()[ds.builder](ctx, ds)
    if not rows:
        raise PrepareError(f"{ds.key}: no audio found under {ctx.raw}")
    write_jsonl(manifest, sorted(rows, key=lambda r: (r["subset"], r["utt_id"])))
    mstamp.write_text(json.dumps({"rows": len(rows), "filter": _filter_tag(ctx, ds)}),
                      encoding="utf-8")
    ctx.log(f"  {len(rows)} rows, {sum(r['duration'] for r in rows) / 3600:.2f} h -> "
            f"{ctx.rel(manifest)}")
    return manifest


def _filter_tag(ctx, ds):
    return ",".join(ctx.hifitts_speakers) if ds.key == "hifitts" else ""


def write_jsonl(path, rows):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(ctx):
    """Hours per dataset/subset and per speaker, over every manifest present."""
    out = {}
    for path in sorted((ctx.root / "manifests").glob("*.jsonl")):
        rows = read_jsonl(path)
        subsets, speakers = defaultdict(float), defaultdict(float)
        for r in rows:
            subsets[r["subset"]] += r["duration"]
            if r.get("speaker"):
                speakers[r["speaker"]] += r["duration"]
        out[path.stem] = {
            "rows": len(rows), "hours": round(sum(subsets.values()) / 3600, 3),
            "subsets": {k: round(v / 3600, 3) for k, v in sorted(subsets.items())},
            "speakers": len(speakers),
            "speaker_hours": {k: round(v / 3600, 3) for k, v in sorted(speakers.items())},
        }
    (ctx.dir("manifests") / "summary.json").write_text(json.dumps(out, indent=2) + "\n",
                                                        encoding="utf-8")
    return out


def disk_free_gb(path):
    return shutil.disk_usage(path).free / 1e9
