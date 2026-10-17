# FluxVoice-S2UT

**Real-time textless voice changer & simultaneous speech translator — written in C++.**

FluxVoice-S2UT is an ultra-low-latency, continuous-streaming voice conversion (VC) and simultaneous speech-to-speech translation (Simul-S2ST) engine. Instead of a traditional high-latency cascaded pipeline (`Speech → Text → Translated Text → Speech`), FluxVoice operates entirely in the **acoustic-semantic latent space** using a **Speech-to-Unit Translation (S2UT)** architecture: discrete phonetic/semantic audio units — not written words — are the primary language of the neural network.

| Mode | Path | Target latency |
|---|---|---|
| **A — Voice Changer only** | Encoder → Vocoder | ~60–100 ms |
| **B — Simultaneous translation + VC** | Encoder → Wait-*k* S2UT → Vocoder | ~300–500 ms |

---

## Contents

- [1. Core architectural pillars](#1-core-architectural-pillars)
- [2. System data flow](#2-system-data-flow)
- [3. Machine learning pipeline](#3-machine-learning-pipeline)
- [4. C++ engine & DSP](#4-c-engine--dsp)
- [5. Repository structure](#5-repository-structure)
- [6. Building](#6-building)
- [7. Development milestones](#7-development-milestones)
- [8. Research & references](#8-research--references)

---

## 1. Core architectural pillars

1. **Textless encode–decode (discrete speech units).**
   Incoming audio is continuously encoded into *semantic tokens* (what is being said) plus *prosody/pitch features* (how it is being said), stripping away the source speaker's timbre.
2. **Dual-mode routing (VC-only vs. S2UT).**
   - **Mode A (Voice Changer):** semantic units bypass the translator and feed the target-voice unit vocoder directly.
   - **Mode B (Translation + VC):** semantic units pass through a causal sequence-to-sequence latent translator before vocoding.
3. **Continuous stream & Wait-*k* policy.**
   No VAD sentence-waiting. Audio is processed in fixed **80 ms causal chunks** with **KV-caching** (key–value state persistence). During translation, a Wait-*k* policy buffers just enough acoustic tokens (*k* chunks) to resolve cross-lingual word order before emitting a continuous stream of target-language units.
4. **Overlap-Add (OLA) waveform synthesis.**
   To prevent phase discontinuities (clicks/pops) at chunk boundaries, the unit vocoder synthesizes overlapping windows blended in C++ with a raised-cosine (Hann) crossfade.

---

## 2. System data flow

```text
[Selectable Input Device (miniaudio)]
   │  raw PCM float32 @ 16/24 kHz, 20 ms callback frames
   ▼
[Lock-Free Input Ring Buffer (SPSC)]
   │  slices 80 ms hop chunks + 20 ms historical context
   ▼
[Stage 1: Causal Semantic Encoder (ONNX)] ◄──► [Encoder KV-cache / CNN state]
   │  emits discrete semantic units + F0 pitch contour every 80 ms
   ├───► (translation OFF) ──────────────────────────┐
   ▼ (translation ON)                                │
[Stage 2: Wait-k Latent S2UT Translator (ONNX)] ◄──► [Decoder KV-cache]
   │  buffers k semantic units → emits target units  │
   ▼                                                 ▼
[Stage 3: Causal Unit-Vocoder (ONNX)] ◄─────────────┘
   │  conditioned on target speaker embedding → 100 ms PCM
   ▼
[DSP Overlap-Add (OLA) Crossfader]
   │  20 ms Hann crossfade with previous chunk tail → 80 ms clean PCM
   ▼
[Lock-Free Output Ring Buffer (SPSC)]
   ▼
[Selectable Output Device / Virtual Audio Cable (miniaudio)]
```

---

## 3. Machine learning pipeline

All models are trained/fine-tuned in **PyTorch** and exported to **stateful ONNX graphs** (explicitly exposing `past_key_values` / recurrent states as graph inputs and outputs).

### Stage 1 — Causal speech tokenizer (encoder)

**Goal:** convert 80 ms of raw waveform into speaker-independent discrete semantic tokens.

Architecture options:

- **Kyutai Mimi codec (recommended).** A streaming neural audio codec running at 12.5 Hz (one frame per 80 ms) that natively splits audio into *semantic* and *acoustic* tokens using causal convolutions.
- **Causal HuBERT / ContentVec + k-means.** Continuous features from layer 6/9 of a causal self-supervised speech model, quantized into `V ∈ {100, 500, 1000}` discrete cluster IDs via a pre-trained k-means codebook, paired with RMVPE or SwiftF0 for pitch extraction.

### Stage 2 — Wait-*k* latent translator (S2UT)

**Goal:** translate source semantic units `U_src` into target-language semantic units `U_tgt` on a continuous stream.

- **Architecture:** causal Conformer/Transformer encoder–decoder with Monotonic Multihead Attention, or a strict Wait-*k* state machine.
- **Wait-*k* execution logic** — let `t_src` be the number of source unit chunks received since speech onset and `t_tgt` the number of target unit chunks emitted:
  - **READ** — if `t_src − t_tgt < k`: append the incoming unit to the encoder KV-cache and emit a `<SILENCE/WAIT>` token to the vocoder.
  - **WRITE** — once `t_src − t_tgt ≥ k`: autoregressively decode target semantic units while advancing both encoder and decoder KV-caches.
  - **Tail flush** — when energy drops below the noise floor for `> 400 ms`: flush remaining cached tokens and reset the *k* counter.

### Stage 3 — Custom-voice unit vocoder (decoder)

**Goal:** synthesize high-fidelity raw waveform from discrete semantic units in a custom target voice.

- **Architecture:** causal Unit-HiFi-GAN (or BigVGAN-streaming / Mimi decoder).
- **Training your own voice:**
  1. Record 1–3 h of clean, dry audio of the target speaker.
  2. Pass the dataset through Stage 1 to extract ground-truth discrete units and F0 contours.
  3. Train the causal Unit-HiFi-GAN generator to reconstruct the target waveform directly from those units. Because the encoder stripped the original speaker identity, the vocoder is forced to bake the target timbre into its weights (optionally conditioned on a static speaker embedding `e_spk`).

---

## 4. C++ engine & DSP

### Threading model

Three isolated threads guarantee zero audio dropouts:

| Thread | Priority | Duties | Constraints |
|---|---|---|---|
| **Audio I/O** | Real-time | `memcpy` between OS audio buffers and the lock-free SPSC ring buffers (via miniaudio) | No allocations, no mutexes, no logging |
| **Inference worker** | Normal | Polls input ring buffer; when ≥ 80 ms (1280 samples @ 16 kHz) is available, runs the ONNX sessions, applies OLA, pushes to output ring buffer | Budgeted per-chunk |
| **UI** | Normal | Dear ImGui / Qt6: device dropdowns, translation toggle, Wait-*k* slider (`k ∈ [1..10]`), `.onnx` model loading, latency telemetry | — |

### Overlap-Add (OLA) crossfader

Neural vocoders produce slight phase/amplitude mismatches between independent chunks, so the vocoder emits `CHUNK + OVERLAP` samples (e.g. 80 ms + 20 ms = 100 ms) and the leading 20 ms is crossfaded with the previous chunk's tail:

```cpp
#include <vector>
#include <cmath>

class OverlapAddProcessor {
public:
    OverlapAddProcessor(size_t chunk_size, size_t overlap_size)
        : chunk_size_(chunk_size), overlap_size_(overlap_size),
          prev_tail_(overlap_size, 0.0f),
          fade_in_(overlap_size), fade_out_(overlap_size)
    {
        // Raised-cosine (Hann) curves: fade_in[i] + fade_out[i] == 1.0f
        for (size_t i = 0; i < overlap_size_; ++i) {
            float phase = static_cast<float>(i) / static_cast<float>(overlap_size_);
            fade_in_[i]  = 0.5f * (1.0f - std::cos(phase * 3.1415926535f));
            fade_out_[i] = 1.0f - fade_in_[i];
        }
    }

    // in:  vocoder output of size (chunk_size_ + overlap_size_)
    // out: clean chunk of size (chunk_size_) for the output ring buffer
    void process(const float* vocoder_out, float* clean_out) {
        for (size_t i = 0; i < overlap_size_; ++i)
            clean_out[i] = vocoder_out[i] * fade_in_[i] + prev_tail_[i] * fade_out_[i];
        for (size_t i = overlap_size_; i < chunk_size_; ++i)
            clean_out[i] = vocoder_out[i];
        for (size_t i = 0; i < overlap_size_; ++i)
            prev_tail_[i] = vocoder_out[chunk_size_ + i];
    }

private:
    size_t chunk_size_, overlap_size_;
    std::vector<float> prev_tail_, fade_in_, fade_out_;
};
```

Canonical implementation: [`src/audio/OverlapAdd.hpp`](src/audio/OverlapAdd.hpp).

---

## 5. Repository structure

```text
FluxVoice-S2UT/
├── CMakeLists.txt
├── README.md
├── RESEARCH_GUIDE.md          # Annotated bibliography of papers, repos, tools
├── Documents/                 # Milestone plans (staged implementation docs)
│   ├── Milestone 1/, 1.5/, 2/…4/
│   └── Research_documents/    # Dated research reports feeding the plans
├── models/                    # Exported .onnx graphs (git-ignored)
│   ├── encoder_mimi.onnx
│   ├── translator_s2ut_waitk.onnx
│   └── vocoder_custom_voice.onnx
├── python_training/           # PyTorch training & ONNX export scripts
│   ├── extract_units.py
│   ├── train_unit_vocoder.py
│   └── export_stateful_onnx.py
├── third_party/               # Vendored / fetched dependencies
│   ├── miniaudio/             # Single-header audio I/O
│   ├── readerwriterqueue/     # moodycamel lock-free SPSC queue
│   ├── onnxruntime/           # Prebuilt ONNX Runtime C++ SDK
│   └── imgui/                 # Lightweight GUI
├── tests/                     # CTest unit tests (no audio device needed)
├── tools/                     # fv_probe (record / tone / latency), loopback + soak analyzers, OLA sine WAVs
└── src/
    ├── main.cpp
    ├── audio/
    │   ├── AudioEngine.hpp    # Device enumeration & miniaudio duplex stream
    │   ├── DuplexRouter.hpp   # Real-time callback body + SPSC queues + counters
    │   ├── ChunkAssembler.hpp # Arbitrary callback sizes → fixed 80 ms model frames
    │   └── OverlapAdd.hpp     # OLA crossfader
    ├── pipeline/
    │   ├── PipelineTiming.hpp # Chunk / context / overlap geometry @ 16 kHz
    │   ├── KVStateCache.hpp   # Recurrent ONNX tensor state across chunks
    │   ├── WaitKPolicy.hpp    # Simultaneous-translation state machine
    │   └── InferencePipeline.cpp
    └── ui/
        └── AppWindow.cpp      # Device dropdowns & latency telemetry
```

---

## 6. Building

### Prerequisites

- CMake ≥ 3.22
- C++20 compiler — MSVC 2022 / GCC 12+ / Clang 15+
- ONNX Runtime C++ SDK (CUDA, DirectML or CoreML execution provider) — *required from Milestone 2*
- Virtual audio cable — VB-Cable (Windows) / BlackHole (macOS) / PipeWire virtual sink (Linux)

### Quick start

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release   # fetches miniaudio + readerwriterqueue
cmake --build build -j
ctest --test-dir build                           # unit + CLI tests (no audio device needed)
./build/fluxvoice --list-devices                 # enumerate audio devices
./build/fluxvoice --passthrough                  # M1: click-free loopback test (Enter / Ctrl+C stops)
tools/pw_loopback_check.sh --mode queue --load   # Linux: tone loopback via null sinks + glitch analysis
./build/fv_probe --list-devices                  # M1 stages 05–06: record / measure a cable route
tools/pw_latency_check.sh                        # Linux: path latencies via null sinks + fv_probe
powershell -ExecutionPolicy Bypass -File tools/setup_test_rig.ps1   # Windows: check/install VB-CABLE + Voicemeeter, build, verify the rig
```

> Header-only deps (miniaudio, readerwriterqueue) are fetched automatically via `FetchContent`.
> Point `-DONNXRUNTIME_ROOT=/path/to/onnxruntime` at a prebuilt SDK when wiring the neural stages.
> CI builds should pass `-DFLUXVOICE_WARNINGS_AS_ERRORS=ON` to keep our sources warning-free.

---

## 7. Development milestones

Detailed staged plans live in [`Documents/`](Documents/). Each milestone has a `00_master.md` index plus numbered stage files.

| # | Milestone | Deliverable | Plan |
|---|---|---|---|
| 1 | **DSP loopback & audio engine** | miniaudio device selector + SPSC ring buffer + `OverlapAddProcessor`; click-free pass-through into a virtual audio cable | [Documents/Milestone 1](Documents/Milestone%201/00_master.md) |
| 1.5 | **Cross-platform audio devices** | Stable device selection, hot-plug refresh/recovery, typed errors; CI + hardware validation on Windows/macOS/Linux (runs parallel to M2) | [Documents/Milestone 1.5](Documents/Milestone%201.5/00_master.md) |
| 2 | **Textless VC mode** | Causal encoder + custom Unit-HiFi-GAN exported to ONNX; Mode A runs in C++ with stateful KV/CNN caches | [Documents/Milestone 2](Documents/Milestone%202/00_master.md) |
| 3 | **Wait-*k* S2UT mode** | Causal S2UT transformer on discrete units; `WaitKPolicy` bridges encoder → vocoder in C++ | [Documents/Milestone 3](Documents/Milestone%203/00_master.md) |
| 4 | **UI & optimization** | ImGui dashboard, FP16/INT8 quantization, GPU execution-provider toggles | [Documents/Milestone 4](Documents/Milestone%204/00_master.md) |

---

## 8. Research & references

See [`RESEARCH_GUIDE.md`](RESEARCH_GUIDE.md) for the curated, annotated bibliography — covering Simul-S2UT (Hibiki, StreamSpeech, SeamlessStreaming, fairseq S2UT), causal speech tokenizers (Mimi, HuBERT), unit vocoders (Unit-HiFi-GAN, speech-resynthesis), C++ streaming inference (sherpa-onnx), and datasets (CVSS, VoxPopuli).
