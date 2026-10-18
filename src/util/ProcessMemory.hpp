// ProcessMemory — resident set size of the current process, for soak tests
// (Milestone 1 / Stage 06: "RSS stable, no leaks in the RT path").
//
// Not real-time safe (system calls, file I/O): call from the control thread.
#pragma once

#include <cstddef>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <cstdio>
#include <unistd.h>
#endif

namespace fluxvoice {

/// Resident set size in bytes (Windows: working set); 0 if unavailable.
inline std::size_t resident_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) return pmc.WorkingSetSize;
    return 0;
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS)
        return static_cast<std::size_t>(info.resident_size);
    return 0;
#elif defined(__linux__)
    // /proc/self/statm: total and resident size, in pages.
    long total = 0, resident = 0;
    std::FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    const bool ok = std::fscanf(f, "%ld %ld", &total, &resident) == 2;
    std::fclose(f);
    return ok ? static_cast<std::size_t>(resident) * static_cast<std::size_t>(sysconf(_SC_PAGESIZE)) : 0;
#else
    return 0;
#endif
}

} // namespace fluxvoice
