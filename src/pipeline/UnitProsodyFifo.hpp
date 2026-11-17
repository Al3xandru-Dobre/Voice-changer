// UnitProsodyFifo — frame ring between translator_dec and the vocoder (Mode B,
// Milestone 3 / Stage 04).
//
// translator_dec emits chunk_frames (4) target frames per step — or 8 when an
// adaptive step writes twice — while the vocoder consumes 1, 2 or 4 frames per
// hop (--frames-per-step). The FIFO decouples the two: push whole translator
// chunks, pop exact vocoder hops.
//
// A frame is the target unit plus the contract-v2 `prosody` row
// (z_f0, voicing, aperiodicity, z_energy). The backing store is allocated in
// the constructor; push_chunk() and pop_hop() never allocate, lock or block.
// Single-threaded: both ends are driven by the inference worker.
// python_training/fvml/waitk_policy.py carries the Python twin.
#pragma once

#include "dsp/PitchTracker.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

class UnitProsodyFifo {
public:
    static constexpr std::size_t kDefaultCapacity = 64;   // 1.28 s at 50 Hz
    static constexpr std::size_t kProsodyDims = 4;

    struct Frame {
        std::int64_t unit = 0;
        float logf0_z = 0.0f;
        float voicing = 0.0f;
        float aperiodicity = 1.0f;
        float energy_z = 0.0f;
    };

    struct PushResult {
        std::size_t pushed = 0;    // frames queued (0 on overflow)
        bool eos = false;          // the chunk contained the <EOS> unit
        bool overflow = false;     // not enough room: nothing was queued
    };

    /// `vocab` is the number of regular vocoder units; `eos_unit` is the
    /// contract's special_units.eos.
    UnitProsodyFifo(std::size_t capacity, std::int64_t vocab, std::int64_t eos_unit)
        : buf_(std::max<std::size_t>(capacity, 1)), vocab_(vocab), eos_unit_(eos_unit) {}

    explicit UnitProsodyFifo(std::size_t capacity = kDefaultCapacity, std::int64_t vocab = 1000)
        : UnitProsodyFifo(capacity, vocab, vocab + 1) {}

    std::size_t capacity() const { return buf_.size(); }
    std::size_t size() const { return size_; }
    std::size_t free_frames() const { return buf_.size() - size_; }
    bool empty() const { return size_ == 0; }
    std::uint64_t overflows() const { return overflows_; }

    void clear() {
        head_ = 0;
        size_ = 0;
    }

    /// Queue one translator chunk. `prosody` is row-major [n, 4]. Frames from
    /// the first special unit (< 0 or >= vocab: <PAD>/<EOS>/<BOS>) onwards are
    /// dropped. All valid frames are queued, or none (overflow).
    PushResult push_chunk(std::span<const std::int64_t> units, std::span<const float> prosody) {
        const std::size_t n = std::min(units.size(), prosody.size() / kProsodyDims);
        PushResult r;
        std::size_t valid = n;
        for (std::size_t i = 0; i < n; ++i) {
            const std::int64_t u = units[i];
            if (u == eos_unit_) r.eos = true;
            if (valid == n && (u < 0 || u >= vocab_)) valid = i;
        }
        if (valid > free_frames()) {
            ++overflows_;
            r.overflow = true;
            return r;
        }
        std::size_t tail = (head_ + size_) % buf_.size();
        for (std::size_t i = 0; i < valid; ++i) {
            const float* p = prosody.data() + i * kProsodyDims;
            buf_[tail] = Frame{units[i], p[0], p[1], p[2], p[3]};
            tail = (tail + 1) % buf_.size();
        }
        size_ += valid;
        r.pushed = valid;
        return r;
    }

    /// Pop exactly one vocoder hop. Returns false (and writes nothing) when
    /// fewer than n_frames are queued: the caller emits silence.
    bool pop_hop(std::size_t n_frames, Frame* frames_out) {
        if (n_frames > size_) return false;
        for (std::size_t i = 0; i < n_frames; ++i) {
            frames_out[i] = buf_[head_];
            head_ = (head_ + 1) % buf_.size();
        }
        size_ -= n_frames;
        return true;
    }

    /// Same, split into the vocoder's unit IDs and whitened prosody: z_logf0,
    /// voicing, aperiodicity and z_energy are set, the raw-domain PitchFrame
    /// fields keep their defaults.
    bool pop_hop(std::size_t n_frames, std::int64_t* units_out, PitchFrame* prosody_out) {
        if (n_frames > size_) return false;
        for (std::size_t i = 0; i < n_frames; ++i) {
            const Frame& f = buf_[head_];
            units_out[i] = f.unit;
            PitchFrame p;
            p.z_logf0 = f.logf0_z;
            p.voicing = f.voicing;
            p.aperiodicity = f.aperiodicity;
            p.z_energy = f.energy_z;
            prosody_out[i] = p;
            head_ = (head_ + 1) % buf_.size();
        }
        size_ -= n_frames;
        return true;
    }

private:
    std::vector<Frame> buf_;
    std::size_t head_ = 0;
    std::size_t size_ = 0;
    std::int64_t vocab_;
    std::int64_t eos_unit_;
    std::uint64_t overflows_ = 0;
};
