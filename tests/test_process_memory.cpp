// test_process_memory -- native-header/link smoke test for ProcessMemory.hpp.
//
// util/ProcessMemory.hpp is included FIRST on purpose: on Windows it pulls in
// <windows.h> and <psapi.h>, and its NOMINMAX guard must keep the min/max
// macros from breaking std::min/std::max below. Building and linking this TU
// also proves the psapi link needed by GetProcessMemoryInfo. No audio
// hardware involved -- the documented "0 means unavailable" result is fine.
#include "util/ProcessMemory.hpp"
#include "TestSupport.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <type_traits>

static_assert(std::is_same_v<decltype(fluxvoice::resident_bytes()), std::size_t>,
              "resident_bytes() must return std::size_t");

int main() {
    CHECK(std::min(2, 3) == 2);
    CHECK(std::max(2, 3) == 3);

    const std::size_t rss = fluxvoice::resident_bytes();
    std::printf("resident_bytes = %llu\n", static_cast<unsigned long long>(rss));
    // Any value is valid: a positive RSS is expected locally, while 0 is the
    // documented "unavailable" result on an unsupported platform.
    return test::finish("process_memory");
}
