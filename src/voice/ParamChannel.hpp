// ParamChannel — control-thread -> worker hand-off for full VoiceParams
// snapshots (Milestone 2 / Stage 09). Strictly SPSC: exactly one producer
// (the control/UI thread) and exactly one consumer (the inference worker),
// like the audio queues. The snapshot is what the worker applies at the next
// hop; the newest enqueued snapshot always wins.
//
// push() never blocks: it parks the newest snapshot in a pending slot and
// tries a try_enqueue. If the queue is momentarily full the snapshot stays
// pending and a later push() or flush() delivers it — flush() is called by
// the control loop each tick, so the latest wins eventually.
#pragma once

#include "voice/VoiceParams.hpp"

#include <readerwriterqueue.h>

#include <atomic>
#include <cstdint>

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4324)  // cache-line aligned SPSC queue padding
#endif

namespace fluxvoice {

class ParamChannel {
public:
    struct Message {
        VoiceParams params;
        std::uint64_t generation = 0;
    };

    static_assert(std::atomic<std::uint64_t>::is_always_lock_free,
                  "generations must be lock-free");

    ParamChannel() : queue_(64) {}

    ParamChannel(const ParamChannel&) = delete;
    ParamChannel& operator=(const ParamChannel&) = delete;

    // -- control thread -----------------------------------------------------
    /// Queue a full snapshot; returns its generation number.
    std::uint64_t push(const VoiceParams& p) {
        pending_.params = p;
        pending_.generation = ++next_generation_;
        has_pending_ = true;
        flush();
        return pending_.generation;
    }

    /// Retry the pending snapshot (control loop each tick). True when nothing
    /// is left pending.
    bool flush() {
        if (!has_pending_) return true;
        if (!queue_.try_enqueue(pending_)) return false;
        has_pending_ = false;
        pushed_generation_.store(pending_.generation, std::memory_order_relaxed);
        return true;
    }

    std::uint64_t pushed_generation() const {
        return pushed_generation_.load(std::memory_order_relaxed);
    }

    // -- worker -------------------------------------------------------------
    /// Drain everything, keep the newest. True when `out` was updated.
    bool drain(VoiceParams& out) noexcept {
        Message m{};
        bool got = false;
        std::uint64_t last = 0;
        while (queue_.try_dequeue(m)) {
            out = m.params;
            last = m.generation;
            got = true;
        }
        if (got) applied_generation_.store(last, std::memory_order_relaxed);
        return got;
    }

    /// Generation the worker last applied; read from the control thread.
    std::uint64_t applied_generation() const noexcept {
        return applied_generation_.load(std::memory_order_relaxed);
    }

private:
    moodycamel::ReaderWriterQueue<Message> queue_;
    Message pending_{};          // producer-side only
    bool has_pending_ = false;   // producer-side only
    std::uint64_t next_generation_ = 0;  // producer-side only
    std::atomic<std::uint64_t> pushed_generation_{0};
    std::atomic<std::uint64_t> applied_generation_{0};
};

} // namespace fluxvoice

#if defined(_MSC_VER)
#pragma warning(pop)
#endif
