// LoaderThread — one short-lived worker thread and a one-slot result queue,
// the pattern AppModel uses for bundle loading (M4 Stage 05) next to
// VoiceStudioModel::poll_analysis and the native dialogs: the work runs off
// the UI thread, the result waits under a mutex, and the control thread
// drains it from tick(). It owns no application state.
//
// Contract:
//   * start() / poll() / busy() / join() are control-thread calls;
//   * only the job runs on the loader thread, and it must not touch anything
//     the control thread owns (capture copies);
//   * at most one job at a time: start() refuses while busy(), which spans
//     "running" and "finished, result not yet drained";
//   * the destructor joins, so a result nobody drained is destroyed on the
//     control thread, after the job has ended;
//   * an exception escaping the job is stored and rethrown by poll().
#pragma once

#include <exception>
#include <functional>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace fluxvoice::frontend {

template <typename Result>
class LoaderThread {
public:
    LoaderThread() = default;
    ~LoaderThread() { join(); }

    LoaderThread(const LoaderThread&)            = delete;
    LoaderThread& operator=(const LoaderThread&) = delete;

    /// Run `job` on a new thread. False (nothing started) while busy().
    bool start(std::function<Result()> job) {
        if (busy_) return false;
        if (thread_.joinable()) thread_.join();  // a finished thread whose result was drained
        // busy_ is set only once the thread exists: a failed std::thread
        // construction (std::system_error) leaves the loader free.
        std::thread t([this, job = std::move(job)]() mutable {
            std::optional<Result> out;
            std::exception_ptr error;
            try {
                out.emplace(job());
            } catch (...) {
                error = std::current_exception();
            }
            const std::lock_guard<std::mutex> lock(mutex_);
            result_ = std::move(out);
            error_  = error;
            ready_  = true;
        });
        thread_ = std::move(t);
        busy_   = true;
        return true;
    }

    /// True from start() until poll() has returned the result.
    bool busy() const { return busy_; }

    /// The finished job's result, once; nullopt while it is still running.
    /// Rethrows an exception that escaped the job.
    std::optional<Result> poll() {
        if (!busy_) return std::nullopt;
        std::optional<Result> out;
        std::exception_ptr error;
        {
            const std::lock_guard<std::mutex> lock(mutex_);
            if (!ready_) return std::nullopt;
            out    = std::move(result_);
            error  = error_;
            result_.reset();
            error_ = nullptr;
            ready_ = false;
        }
        if (thread_.joinable()) thread_.join();  // already past its last statement
        busy_ = false;
        if (error) std::rethrow_exception(error);
        return out;
    }

    /// Wait for the running job (if any) and discard nothing: the result stays
    /// queued for poll(). Used by the destructor.
    void join() {
        if (thread_.joinable()) thread_.join();
    }

private:
    std::mutex             mutex_;
    std::optional<Result>  result_;
    std::exception_ptr     error_;
    bool                   ready_ = false;
    bool                   busy_  = false;  // control-thread only
    std::thread            thread_;
};

} // namespace fluxvoice::frontend
