#pragma once

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <avrt.h>
#elif defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#endif

#include <cerrno>

namespace fluxvoice {

// Best effort for THIS worker thread only; never requests sudo/capabilities or
// changes another process. A denied request is reported, not a startup failure.
class WorkerPriority {
public:
    explicit WorkerPriority(bool attempt = true) noexcept {
        if (!attempt) return;
#if defined(_WIN32)
        DWORD task = 0;
        method_ = "MMCSS Pro Audio";
        handle_ = AvSetMmThreadCharacteristicsW(L"Pro Audio", &task);
        elevated_ = handle_ != nullptr;
        result_ = elevated_ ? 0 : static_cast<int>(GetLastError());
#elif defined(__APPLE__)
        method_ = "QoS user-interactive";
        old_qos_ = pthread_get_qos_class_np(pthread_self(), &old_relative_);
        result_ = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
        elevated_ = result_ == 0;
#elif defined(__linux__)
        method_ = "SCHED_FIFO";
        result_ = pthread_getschedparam(pthread_self(), &old_policy_, &old_param_);
        if (result_ != 0) return;
        if ((old_policy_ == SCHED_FIFO || old_policy_ == SCHED_RR) && old_param_.sched_priority >= 1) {
            method_ = "already real-time";
            elevated_ = true;
            return;
        }
        sched_param requested{};
        requested.sched_priority = 1;
        result_ = pthread_setschedparam(pthread_self(), SCHED_FIFO, &requested);
        if (result_ == 0) { elevated_ = true; fifo_ = true; return; }
        method_ = "nice -10";
        errno = 0;
        old_nice_ = getpriority(PRIO_PROCESS, 0);
        if (errno != 0) { result_ = errno; return; }
        if (old_nice_ <= -10) { elevated_ = true; result_ = 0; return; }
        result_ = setpriority(PRIO_PROCESS, 0, -10);
        if (result_ == 0) { elevated_ = true; nice_ = true; }
        else result_ = errno;
#else
        method_ = "unsupported platform";
#endif
    }

    ~WorkerPriority() {
        if (!elevated_) return;
#if defined(_WIN32)
        AvRevertMmThreadCharacteristics(handle_);
#elif defined(__APPLE__)
        if (old_qos_ != QOS_CLASS_UNSPECIFIED)
            pthread_set_qos_class_self_np(old_qos_, old_relative_);
#elif defined(__linux__)
        if (fifo_) pthread_setschedparam(pthread_self(), old_policy_, &old_param_);
        else if (nice_) setpriority(PRIO_PROCESS, 0, old_nice_);
#endif
    }

    WorkerPriority(const WorkerPriority&) = delete;
    WorkerPriority& operator=(const WorkerPriority&) = delete;
    bool elevated() const noexcept { return elevated_; }
    int result() const noexcept { return result_; }
    const char* method() const noexcept { return method_; }

private:
    const char* method_ = "not requested";
    bool elevated_ = false;
    int result_ = 0;
#if defined(_WIN32)
    HANDLE handle_ = nullptr;
#elif defined(__APPLE__)
    qos_class_t old_qos_ = QOS_CLASS_UNSPECIFIED;
    int old_relative_ = 0;
#elif defined(__linux__)
    bool fifo_ = false, nice_ = false;
    int old_policy_ = 0, old_nice_ = 0;
    sched_param old_param_{};
#endif
};

} // namespace fluxvoice
