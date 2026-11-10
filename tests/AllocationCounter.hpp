#pragma once

#if defined(__SANITIZE_THREAD__)
#define FLUXVOICE_TEST_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define FLUXVOICE_TEST_TSAN 1
#endif
#endif

namespace test {
#ifdef FLUXVOICE_TEST_TSAN
inline constexpr bool kAllocationCounting = false;
#else
inline constexpr bool kAllocationCounting = true;
#endif
long allocation_count() noexcept;
} // namespace test
