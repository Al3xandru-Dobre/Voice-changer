// Minimal dependency-free test helpers for CTest executables.
// Each test binary returns non-zero if any CHECK failed.
#pragma once

#include <cstdio>

namespace test {

inline int& failures() {
    static int n = 0;
    return n;
}

inline int finish(const char* suite) {
    if (failures() == 0) std::printf("[%s] all checks passed\n", suite);
    else                 std::printf("[%s] %d check(s) FAILED\n", suite, failures());
    return failures() == 0 ? 0 : 1;
}

} // namespace test

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::printf("%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
            ++test::failures();                                                 \
        }                                                                       \
    } while (0)
