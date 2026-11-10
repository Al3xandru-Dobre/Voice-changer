// KVStateCache — unit tests for typed state backing buffers
// (Milestone 2 / Stage 07, Documents/Milestone 2/07_ort_cpp_runtime.md).
//
// ORT-free: runs in every CI build. Covers registration, lookup, shape/size
// calculations (including scalar [] and zero/negative dimensions), dtypes
// (f32, i64, f16 raw bytes), 8-byte alignment, re-registration semantics, and
// zero-allocation reset().
#include "pipeline/KVStateCache.hpp"
#include "TestSupport.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#if defined(__SANITIZE_THREAD__)
#define FLUXVOICE_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define FLUXVOICE_TSAN 1
#endif
#endif

#ifndef FLUXVOICE_TSAN
static std::atomic<long> g_allocs{0};

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif

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
#endif // FLUXVOICE_TSAN

namespace {

using fluxvoice::KVStateCache;
using fluxvoice::TensorDtype;
using fluxvoice::dtype_element_size;
using fluxvoice::dtype_from_string;
using fluxvoice::to_string;

void test_dtype_helpers() {
    CHECK(dtype_from_string("f32") == TensorDtype::kF32);
    CHECK(dtype_from_string("i64") == TensorDtype::kI64);
    CHECK(dtype_from_string("f16") == TensorDtype::kF16);
    CHECK(dtype_from_string("i32") == TensorDtype::kOther);

    CHECK(std::strcmp(to_string(TensorDtype::kF32), "f32") == 0);
    CHECK(std::strcmp(to_string(TensorDtype::kI64), "i64") == 0);
    CHECK(std::strcmp(to_string(TensorDtype::kF16), "f16") == 0);
    CHECK(std::strcmp(to_string(TensorDtype::kOther), "other") == 0);

    CHECK(dtype_element_size(TensorDtype::kF32) == 4);
    CHECK(dtype_element_size(TensorDtype::kI64) == 8);
    CHECK(dtype_element_size(TensorDtype::kF16) == 2);
    CHECK(dtype_element_size(TensorDtype::kOther) == 0);
}

void test_registration_and_dtypes() {
    KVStateCache cache;
    CHECK(cache.size() == 0);
    CHECK(cache.total_bytes() == 0);
    CHECK(cache.find("missing") == nullptr);

    // 1. f32 buffer (default dtype)
    auto& f32_buf = cache.register_buffer("conv.0", {1, 16, 4});
    CHECK(cache.size() == 1);
    CHECK(f32_buf.dtype == TensorDtype::kF32);
    CHECK(f32_buf.num_elements() == 64);
    CHECK(f32_buf.element_size() == 4);
    CHECK(f32_buf.num_bytes() == 256);
    CHECK(f32_buf.as_f32() != nullptr);
    CHECK(f32_buf.as_i64() == nullptr);
    CHECK(f32_buf.as_f16() == nullptr);
    CHECK(f32_buf.span_f32().size() == 64);
    CHECK(f32_buf.span_i64().empty());
    CHECK(f32_buf.span_f16().empty());
    CHECK(f32_buf.span_bytes().size() == 256);
    CHECK(reinterpret_cast<std::uintptr_t>(f32_buf.raw_data()) % alignof(std::int64_t) == 0);

    // 2. i64 buffer
    auto& i64_buf = cache.register_buffer("step_pos", {1, 3}, TensorDtype::kI64);
    CHECK(cache.size() == 2);
    CHECK(i64_buf.dtype == TensorDtype::kI64);
    CHECK(i64_buf.num_elements() == 3);
    CHECK(i64_buf.element_size() == 8);
    CHECK(i64_buf.num_bytes() == 24);
    CHECK(i64_buf.as_i64() != nullptr);
    CHECK(i64_buf.as_f32() == nullptr);
    CHECK(i64_buf.as_f16() == nullptr);
    CHECK(i64_buf.span_i64().size() == 3);
    CHECK(i64_buf.span_bytes().size() == 24);
    CHECK(reinterpret_cast<std::uintptr_t>(i64_buf.raw_data()) % alignof(std::int64_t) == 0);

    // 3. f16 buffer (odd element count -> tests 8-byte word rounding)
    auto& f16_buf = cache.register_buffer("kv_cache", {1, 5}, TensorDtype::kF16);
    CHECK(cache.size() == 3);
    CHECK(f16_buf.dtype == TensorDtype::kF16);
    CHECK(f16_buf.num_elements() == 5);
    CHECK(f16_buf.element_size() == 2);
    CHECK(f16_buf.num_bytes() == 10);
    CHECK(f16_buf.as_f16() != nullptr);
    CHECK(f16_buf.as_f32() == nullptr);
    CHECK(f16_buf.as_i64() == nullptr);
    CHECK(f16_buf.span_f16().size() == 5);
    CHECK(f16_buf.span_bytes().size() == 10);

    CHECK(cache.total_bytes() == 256 + 24 + 10);

    // Const lookup
    const KVStateCache& ccache = cache;
    CHECK(ccache.find("conv.0") == &f32_buf);
    CHECK(ccache.find("step_pos") == &i64_buf);
    CHECK(ccache.find("kv_cache") == &f16_buf);
    CHECK(ccache.find("unknown") == nullptr);
    CHECK(ccache.find("conv.0")->span_f32().size() == 64);
    CHECK(ccache.find("step_pos")->span_i64().size() == 3);
    CHECK(ccache.find("kv_cache")->span_f16().size() == 5);
    CHECK(ccache.find("kv_cache")->span_bytes().size() == 10);
}

void test_scalar_and_degenerate_shapes() {
    KVStateCache cache;
    // Scalar shape [] has 1 element.
    auto& scalar = cache.register_buffer("scalar_pos", {}, TensorDtype::kI64);
    CHECK(scalar.shape.empty());
    CHECK(scalar.num_elements() == 1);
    CHECK(scalar.num_bytes() == 8);
    scalar.span_i64()[0] = 42;
    CHECK(scalar.as_i64()[0] == 42);

    // Zero or negative dimension yields 0 elements and nullptr raw_data.
    auto& empty_dim = cache.register_buffer("empty", {1, 0, 8}, TensorDtype::kF32);
    CHECK(empty_dim.num_elements() == 0);
    CHECK(empty_dim.num_bytes() == 0);
    CHECK(empty_dim.raw_data() == nullptr);
    CHECK(empty_dim.span_f32().empty());

    auto& dyn_dim = cache.register_buffer("dyn", {1, -1}, TensorDtype::kF32);
    CHECK(dyn_dim.num_elements() == 0);
    CHECK(dyn_dim.num_bytes() == 0);
    CHECK(dyn_dim.raw_data() == nullptr);
}

void test_reregistration_and_reset_no_alloc() {
    KVStateCache cache;
    auto& a = cache.register_buffer("buf_f32", {2, 4}, TensorDtype::kF32);
    auto& b = cache.register_buffer("buf_i64", {3}, TensorDtype::kI64);
    auto& c = cache.register_buffer("buf_f16", {7}, TensorDtype::kF16);

    for (std::size_t i = 0; i < a.span_f32().size(); ++i) a.span_f32()[i] = static_cast<float>(i + 1);
    for (std::size_t i = 0; i < b.span_i64().size(); ++i) b.span_i64()[i] = static_cast<std::int64_t>(-100 - i);
    for (std::size_t i = 0; i < c.span_f16().size(); ++i) c.span_f16()[i] = static_cast<std::uint16_t>(0x3C00u + i);

    // Re-registering with the same shape & dtype preserves contents.
    auto& a_same = cache.register_buffer("buf_f32", {2, 4}, TensorDtype::kF32);
    CHECK(&a_same == &a);
    CHECK(a_same.span_f32()[3] == 4.0f);

#ifndef FLUXVOICE_TSAN
    const long before = g_allocs.load(std::memory_order_relaxed);
#endif
    cache.reset();
#ifndef FLUXVOICE_TSAN
    CHECK(g_allocs.load(std::memory_order_relaxed) == before);
#endif

    for (float v : a.span_f32()) CHECK(v == 0.0f);
    for (std::int64_t v : b.span_i64()) CHECK(v == 0);
    for (std::uint16_t v : c.span_f16()) CHECK(v == 0);

    // Re-registering with a different shape or dtype resizes and zeroes.
    a.span_f32()[0] = 99.0f;
    auto& a_resized = cache.register_buffer("buf_f32", {4, 4}, TensorDtype::kI64);
    CHECK(a_resized.dtype == TensorDtype::kI64);
    CHECK(a_resized.num_elements() == 16);
    CHECK(a_resized.span_i64()[0] == 0);

    cache.clear();
    CHECK(cache.size() == 0);
    CHECK(cache.total_bytes() == 0);
}

} // namespace

int main() {
    test_dtype_helpers();
    test_registration_and_dtypes();
    test_scalar_and_degenerate_shapes();
    test_reregistration_and_reset_no_alloc();
    return test::finish("kv_state_cache");
}
