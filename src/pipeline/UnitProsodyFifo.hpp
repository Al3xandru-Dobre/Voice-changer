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
// the constructor; push_chunk(), pad_to_multiple() and pop_hop() never
// allocate, lock or block. Single-threaded: both ends are driven by the
// inference worker. python_training/fvml/waitk_policy.py carries the Python
// twin.
//
// BUG-028 (stranded tail frames): push_chunk() drops everything from the
// first special unit, so an <EOS> chunk queues 1..3 frames and an s2/s4
// vocoder could never pop them (and their phase would shift every later
// hop). After each push the Mode B pipeline calls pad_to_multiple(hop
// frames): the remainder is completed with `pad` frames (last unit, voicing 0,
// aperiodicity 1) whose vocoded samples the pipeline replaces by silence, so
// the queue stays hop-aligned and no frame survives into the next utterance.
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
        bool pad = false;          // alignment filler (pad_to_multiple): render as silence
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
            buf_[tail] = Frame{units[i], p[0], p[1], p[2], p[3], false};
            tail = (tail + 1) % buf_.size();
        }
        size_ += valid;
        r.pushed = valid;
        return r;
    }

    /// Complete the queue to a multiple of `n_frames` (the vocoder hop) with
    /// pad frames: the newest frame's unit and pitch, voicing 0, aperiodicity
    /// 1, `pad = true`. Returns the number of pad frames appended (0 when
    /// already aligned, when empty, or when n_frames <= 1). Needs room for
    /// them: keep capacity a multiple of n_frames (then the room always
    /// exists); otherwise it pads nothing and counts an overflow.
    std::size_t pad_to_multiple(std::size_t n_frames) {
        if (n_frames <= 1 || size_ == 0) return 0;
        const std::size_t rem = size_ % n_frames;
        if (rem == 0) return 0;
        const std::size_t pads = n_frames - rem;
        if (pads > free_frames()) {
            ++overflows_;
            return 0;
        }
        const Frame& last = buf_[(head_ + size_ - 1) % buf_.size()];
        const Frame filler{last.unit, last.logf0_z, 0.0f, 1.0f, last.energy_z, true};
        std::size_t tail = (head_ + size_) % buf_.size();
        for (std::size_t i = 0; i < pads; ++i) {
            buf_[tail] = filler;
            tail = (tail + 1) % buf_.size();
        }
        size_ += pads;
        return pads;
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
