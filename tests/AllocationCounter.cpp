#include "AllocationCounter.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace {
std::atomic<long> g_allocs{0};
}

long test::allocation_count() noexcept { return g_allocs.load(std::memory_order_relaxed); }

// Keep replacements in their own TU: GCC otherwise inlines free() into STL
// callers and misdiagnoses the deliberately paired malloc/new replacement.
// TSan provides its own replacements.
#ifndef FLUXVOICE_TEST_TSAN
void* operator new(std::size_t n) {
    g_allocs.fetch_add(1, std::memory_order_relaxed);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t n) { return operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
#endif
