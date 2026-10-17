# FluxVoice-S2UT: Real-Time Textless Voice Changer & Simultaneous Translator

FluxVoice-S2UT is an ultra-low-latency, continuous-streaming voice conversion and simultaneous speech-to-speech translation (Simul-S2ST) engine written in C++. 

Instead of relying on a traditional, high-latency cascaded pipeline (`Speech -> Text -> Translated Text -> Speech`), FluxVoice operates entirely in **Acoustic-Semantic Latent Space** using a **Speech-to-Unit Translation (S2UT)** architecture. Written words are merely a secondary syntactical representation of meaning; FluxVoice treats discrete phonetic/semantic audio units as the primary language of the neural network.

---

## 1. Core Architectural Pillars

1. **Textless Encode-Decode (Discrete Speech Units):** 
   Incoming audio is continuously encoded into semantic tokens (representing *what* is being said) and prosody/pitch features (representing *how* it is being said), stripping away the source speaker's timbre.
2. **Dual-Mode Routing (VC-Only vs. S2UT):**
   * **Mode A (Voice Changer Only):** Semantic units bypass the translator and feed directly into the target voice Unit-Vocoder (~60–100ms latency).
   * **Mode B (Simultaneous Translation + Voice Change):** Semantic units pass through a causal Sequence-to-Sequence latent translator before vocoding (~300–500ms latency).
3. **Continuous Stream & Wait-$k$ Policy:**
   No Voice Activity Detection (VAD) sentence-waiting. Audio is processed in fixed 80ms causal chunks using **KV-Caching** (Key-Value state persistence). During translation, a **Wait-$k$ policy** buffers just enough acoustic tokens ($k$ chunks) to resolve cross-lingual word order before emitting a continuous stream of target language units.
4. **Overlap-Add (OLA) Waveform Synthesis:**
   To prevent phase discontinuities (clicks and pops) at chunk boundaries, the Unit-Vocoder synthesizes overlapping audio windows that are blended in C++ using a raised-cosine (Hann) crossfade.

---

## 2. System Data Flow

```text
[Selectable Input Device (Miniaudio)] 
   │ (Raw PCM Float32 @ 16kHz/24kHz, 20ms callback frames)
   ▼
[Lock-Free Input Ring Buffer (SPSC)]
   │ (Slices 80ms hop chunks + 20ms historical context)
   ▼
[Stage 1: Causal Semantic Encoder (ONNX)] ◄──► [Encoder KV-Cache / CNN State]
   │ (Emits Discrete Semantic Units + F0 Pitch Contour every 80ms)
   ├───► (If Translation == OFF) ──────────────────────────────┐
   ▼ (If Translation == ON)                                    │
[Stage 2: Wait-k Latent S2UT Translator (ONNX)] ◄──► [Decoder KV-Cache]
   │ (Buffers k semantic units -> Emits Target Language Units) │
   ▼                                                           ▼
[Stage 3: Causal Unit-Vocoder (ONNX)] ◄────────────────────────┘
   │ (Conditioned on Target Speaker Embedding + Target Units -> 100ms PCM)
   ▼
[DSP Overlap-Add (OLA) Crossfader]
   │ (20ms Hann Window crossfade with previous chunk tail -> 80ms clean PCM)
   ▼
[Lock-Free Output Ring Buffer (SPSC)]
   │
   ▼
[Selectable Output Device / Virtual Audio Cable (Miniaudio)]

## 3. The Machine Learning Pipeline (Python Training -> ONNX Export)

All models are trained/fine-tuned in PyTorch and exported to stateful ONNX graphs (explicitly exposing past_key_values / recurrent states as graph inputs and outputs).
### Stage 1: Causal Speech Tokenizer (The Encoder)

    Goal: Convert 80ms of raw waveform into speaker-independent discrete semantic tokens.

    Architecture Options:

        Kyutai Mimi Codec (Recommended): A streaming neural audio codec operating at 12.5Hz (one frame per 80ms) that natively splits audio into semantic tokens and acoustic tokens using causal convolutions.

        Causal HuBERT / ContentVec + k-Means: Extracts continuous features from layer 6/9 of a causal self-supervised speech model, quantized into

                
        V∈{100,500,1000}
        V∈{100,500,1000}

              

        discrete cluster IDs via a pre-trained k-Means codebook, paired with RMVPE or SwiftF0 for pitch extraction.

### Stage 2: Wait-

        
k
k

      

Latent Translator (S2UT)

    Goal: Translate source semantic units

            
    Usrc
    Usrc​

          

    into target language semantic units

            
    Utgt
    Utgt​

          

    on a continuous stream.

    Architecture: Causal Conformer/Transformer Encoder-Decoder with Monotonic Multihead Attention or a strict Wait-

            
    k
    k

          

    state machine.

    Wait-

            
    k
    k

          

    Execution Logic:

        Let

                
        tsrc
        tsrc​

              

        be the number of encoded source unit chunks received since speech onset, and

                
        ttgt
        ttgt​

              

        be the number of target unit chunks emitted.

        READ Action: If

                
        tsrc−ttgt<k
        tsrc​−ttgt​<k

              

        , append incoming unit to encoder KV-cache and output a <SILENCE/WAIT> token to the vocoder.

        WRITE Action: Once

                
        tsrc−ttgt≥k
        tsrc​−ttgt​≥k

              

        , autoregressively decode target semantic units while advancing both encoder and decoder KV-caches simultaneously.

        Tail Flush: When energy drops below a noise floor for

                
        >400ms
        >400ms

              

        , flush remaining cached tokens and reset the

                
        k
        k

              

        counter.

###Stage 3: Custom Voice Unit-Vocoder (The Decoder)

    Goal: Synthesize high-fidelity raw waveform from discrete semantic units in your custom target voice.

    Architecture: Causal Unit-HiFi-GAN (or BigVGAN-streaming / Mimi Decoder).

    Training Your Own Voice:

        Record 1–3 hours of clean, dry audio of the target speaker.

        Pass the dataset through Stage 1 (Encoder) to extract ground-truth discrete semantic units and F0 pitch contours.

        Train the causal Unit-HiFi-GAN generator to reconstruct the target speaker's waveform directly from those discrete units. Because the encoder stripped the original speaker's identity, the vocoder is forced to bake the target speaker's timbre into its weights (or via a static Speaker Embedding vector

                
        espk
        espk​

              

        ).


## 4. C++ Engine & DSP Implementation Details

This complete, production-ready README.md is designed to be dropped directly into the root of your repository to serve as your architectural blueprint, followed by a curated, annotated bibliography of papers, repositories, and libraries.
README.md
code Markdown

# FluxVoice-S2UT: Real-Time Textless Voice Changer & Simultaneous Translator

FluxVoice-S2UT is an ultra-low-latency, continuous-streaming voice conversion and simultaneous speech-to-speech translation (Simul-S2ST) engine written in C++. 

Instead of relying on a traditional, high-latency cascaded pipeline (`Speech -> Text -> Translated Text -> Speech`), FluxVoice operates entirely in **Acoustic-Semantic Latent Space** using a **Speech-to-Unit Translation (S2UT)** architecture. Written words are merely a secondary syntactical representation of meaning; FluxVoice treats discrete phonetic/semantic audio units as the primary language of the neural network.

---

## 1. Core Architectural Pillars

1. **Textless Encode-Decode (Discrete Speech Units):** 
   Incoming audio is continuously encoded into semantic tokens (representing *what* is being said) and prosody/pitch features (representing *how* it is being said), stripping away the source speaker's timbre.
2. **Dual-Mode Routing (VC-Only vs. S2UT):**
   * **Mode A (Voice Changer Only):** Semantic units bypass the translator and feed directly into the target voice Unit-Vocoder (~60–100ms latency).
   * **Mode B (Simultaneous Translation + Voice Change):** Semantic units pass through a causal Sequence-to-Sequence latent translator before vocoding (~300–500ms latency).
3. **Continuous Stream & Wait-$k$ Policy:**
   No Voice Activity Detection (VAD) sentence-waiting. Audio is processed in fixed 80ms causal chunks using **KV-Caching** (Key-Value state persistence). During translation, a **Wait-$k$ policy** buffers just enough acoustic tokens ($k$ chunks) to resolve cross-lingual word order before emitting a continuous stream of target language units.
4. **Overlap-Add (OLA) Waveform Synthesis:**
   To prevent phase discontinuities (clicks and pops) at chunk boundaries, the Unit-Vocoder synthesizes overlapping audio windows that are blended in C++ using a raised-cosine (Hann) crossfade.

---

## 2. System Data Flow

```text
[Selectable Input Device (Miniaudio)] 
   │ (Raw PCM Float32 @ 16kHz/24kHz, 20ms callback frames)
   ▼
[Lock-Free Input Ring Buffer (SPSC)]
   │ (Slices 80ms hop chunks + 20ms historical context)
   ▼
[Stage 1: Causal Semantic Encoder (ONNX)] ◄──► [Encoder KV-Cache / CNN State]
   │ (Emits Discrete Semantic Units + F0 Pitch Contour every 80ms)
   ├───► (If Translation == OFF) ──────────────────────────────┐
   ▼ (If Translation == ON)                                    │
[Stage 2: Wait-k Latent S2UT Translator (ONNX)] ◄──► [Decoder KV-Cache]
   │ (Buffers k semantic units -> Emits Target Language Units) │
   ▼                                                           ▼
[Stage 3: Causal Unit-Vocoder (ONNX)] ◄────────────────────────┘
   │ (Conditioned on Target Speaker Embedding + Target Units -> 100ms PCM)
   ▼
[DSP Overlap-Add (OLA) Crossfader]
   │ (20ms Hann Window crossfade with previous chunk tail -> 80ms clean PCM)
   ▼
[Lock-Free Output Ring Buffer (SPSC)]
   │
   ▼
[Selectable Output Device / Virtual Audio Cable (Miniaudio)]

3. The Machine Learning Pipeline (Python Training -> ONNX Export)

All models are trained/fine-tuned in PyTorch and exported to stateful ONNX graphs (explicitly exposing past_key_values / recurrent states as graph inputs and outputs).
Stage 1: Causal Speech Tokenizer (The Encoder)

    Goal: Convert 80ms of raw waveform into speaker-independent discrete semantic tokens.

    Architecture Options:

        Kyutai Mimi Codec (Recommended): A streaming neural audio codec operating at 12.5Hz (one frame per 80ms) that natively splits audio into semantic tokens and acoustic tokens using causal convolutions.

        Causal HuBERT / ContentVec + k-Means: Extracts continuous features from layer 6/9 of a causal self-supervised speech model, quantized into

                
        V∈{100,500,1000}
        V∈{100,500,1000}

              

        discrete cluster IDs via a pre-trained k-Means codebook, paired with RMVPE or SwiftF0 for pitch extraction.

Stage 2: Wait-

        
k
k

      

Latent Translator (S2UT)

    Goal: Translate source semantic units

            
    Usrc
    Usrc​

          

    into target language semantic units

            
    Utgt
    Utgt​

          

    on a continuous stream.

    Architecture: Causal Conformer/Transformer Encoder-Decoder with Monotonic Multihead Attention or a strict Wait-

            
    k
    k

          

    state machine.

    Wait-

            
    k
    k

          

    Execution Logic:

        Let

                
        tsrc
        tsrc​

              

        be the number of encoded source unit chunks received since speech onset, and

                
        ttgt
        ttgt​

              

        be the number of target unit chunks emitted.

        READ Action: If

                
        tsrc−ttgt<k
        tsrc​−ttgt​<k

              

        , append incoming unit to encoder KV-cache and output a <SILENCE/WAIT> token to the vocoder.

        WRITE Action: Once

                
        tsrc−ttgt≥k
        tsrc​−ttgt​≥k

              

        , autoregressively decode target semantic units while advancing both encoder and decoder KV-caches simultaneously.

        Tail Flush: When energy drops below a noise floor for

                
        >400ms
        >400ms

              

        , flush remaining cached tokens and reset the

                
        k
        k

              

        counter.

Stage 3: Custom Voice Unit-Vocoder (The Decoder)

    Goal: Synthesize high-fidelity raw waveform from discrete semantic units in your custom target voice.

    Architecture: Causal Unit-HiFi-GAN (or BigVGAN-streaming / Mimi Decoder).

    Training Your Own Voice:

        Record 1–3 hours of clean, dry audio of the target speaker.

        Pass the dataset through Stage 1 (Encoder) to extract ground-truth discrete semantic units and F0 pitch contours.

        Train the causal Unit-HiFi-GAN generator to reconstruct the target speaker's waveform directly from those discrete units. Because the encoder stripped the original speaker's identity, the vocoder is forced to bake the target speaker's timbre into its weights (or via a static Speaker Embedding vector

                
        espk
        espk​

              

        ).

## 4. C++ Engine & DSP Implementation Details
### Threading Model

To guarantee zero audio dropouts (glitches/underruns), the C++ application runs three isolated threads:

    Audio I/O Thread (Real-Time Priority): Managed by miniaudio. Strictly performs memcpy between OS audio buffers and Lock-Free SPSC (Single-Producer Single-Consumer) Ring Buffers. Zero memory allocations or mutex locks allowed.

    Inference Worker Thread: Polls the Input Ring Buffer. When   
    ≥80ms
    ≥80ms
    (e.g., 1280 samples @ 16kHz) is available, it runs the ONNX Runtime session, applies Overlap-Add (OLA), and pushes the result to the Output Ring Buffer.

    UI Thread: Runs Dear ImGui or Qt6 for selecting Input/Output devices, toggling Translation ON/OFF, adjusting the Wait-
    k
    k
    slider (
    k∈[1..10]
    k∈[1..10]
    ), and loading .onnx voice models.

### Overlap-Add (OLA) Crossfader Reference Implementation

Because neural vocoders generate slight phase and amplitude mismatches between independent chunks, the vocoder outputs CHUNK_SAMPLES + OVERLAP_SAMPLES (e.g., 80ms + 20ms = 100ms), and we crossfade the 20ms boundary:

#include <vector>
#include <cmath>

class OverlapAddProcessor {
private:
    size_t chunk_size_;   // e.g., 1280 samples (80ms @ 16kHz)
    size_t overlap_size_; // e.g., 320 samples  (20ms @ 16kHz)
    std::vector<float> prev_tail_;
    std::vector<float> fade_in_;
    std::vector<float> fade_out_;

public:
    OverlapAddProcessor(size_t chunk_size, size_t overlap_size)
        : chunk_size_(chunk_size), overlap_size_(overlap_size),
          prev_tail_(overlap_size, 0.0f),
          fade_in_(overlap_size), fade_out_(overlap_size) 
    {
        // Precompute Raised-Cosine (Hann) Crossfade Curves
        // Guarantees fade_in[i] + fade_out[i] == 1.0f (Constant Amplitude)
        for (size_t i = 0; i < overlap_size_; ++i) {
            float phase = static_cast<float>(i) / static_cast<float>(overlap_size_);
            fade_in_[i]  = 0.5f * (1.0f - std::cos(phase * 3.1415926535f));
            fade_out_[i] = 1.0f - fade_in_[i];
        }
    }

    // Input: vocoder_out of size (chunk_size_ + overlap_size_)
    // Output: clean_out of size (chunk_size_) ready for Output Ring Buffer
    void process(const float* vocoder_out, float* clean_out) {
        // 1. Crossfade the leading overlap region with previous chunk's tail
        for (size_t i = 0; i < overlap_size_; ++i) {
            clean_out[i] = (vocoder_out[i] * fade_in_[i]) + (prev_tail_[i] * fade_out_[i]);
        }
        // 2. Copy the non-overlapping middle section directly
        for (size_t i = overlap_size_; i < chunk_size_; ++i) {
            clean_out[i] = vocoder_out[i];
        }
        // 3. Save the trailing overlap region for the next iteration
        for (size_t i = 0; i < overlap_size_; ++i) {
            prev_tail_[i] = vocoder_out[chunk_size_ + i];
        }
    }
};

## 5. Repository Structure

FluxVoice-S2UT/
├── CMakeLists.txt
├── third_party/
│   ├── miniaudio/           # Single-header audio I/O
│   ├── readerwriterqueue/   # Moodycamel lock-free SPSC ring buffer
│   ├── onnxruntime/         # Prebuilt ONNX Runtime C++ headers & libs
│   └── imgui/               # Lightweight GUI
├── models/                  # Exported .onnx graphs
│   ├── encoder_mimi.onnx
│   ├── translator_s2ut_waitk.onnx
│   └── vocoder_custom_voice.onnx
├── python_training/         # PyTorch training & ONNX export scripts
│   ├── extract_units.py
│   ├── train_unit_vocoder.py
│   └── export_stateful_onnx.py
└── src/
    ├── main.cpp
    ├── audio/
    │   ├── AudioEngine.hpp  # Device enumeration & miniaudio duplex callback
    │   └── OverlapAdd.hpp   # OLA Crossfader
    ├── pipeline/
    │   ├── KVStateCache.hpp # Manages recurrent ONNX tensors across chunks
    │   ├── WaitKPolicy.hpp  # Simultaneous translation state machine
    │   └── InferencePipeline.cpp
    └── ui/
        └── AppWindow.cpp    # Device dropdowns & latency telemetry

## 6. Build Instructions

### Prerequisites
- CMake 3.22+
- C++20 Compatible Compiler (MSVC 2022, GCC 12+, or Clang 15+)
- ONNX Runtime C++ SDK (with CUDA, DirectML, or CoreML execution provider)
- Virtual Audio Cable (VB-Cable on Windows / BlackHole on macOS / PipeWire virtual sink on Linux)

## 7. Development Milestones

- [] Milestone 1 (DSP Loopback & Audio Engine) : Implement miniaudio device selector + SPSC ring buffer + OverlapAddProcessor in C++. Verify click-free audio pass-trhough into Virtual Audio Cable

- [] Milestone 2(Textless VC Mode): Export casual Encoder + custom Unit-HiFi-GAN to ONNX, Run Mode A (Voice changer without translation) in C++ with stateful KV/CNN padding caches

- [] Milestone 3 (Wait-k S2UT Mode) : Train/fine-tune casual S2UT transformer on discrete units, Impelement WaitKPolicy in C++ and connect between Econder and Vocoder

- [] Milestione 4 (UI & Optimisation) : Add ImGUI dashboard, FP16/INT8 quantization and GPU Execution Provider toggles