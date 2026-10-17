cmake_minimum_required(VERSION 3.22)

project(FluxVoice-S2UT VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(FLUXVOICE_FETCH_DEPS      "Fetch header-only dependencies via FetchContent" ON)
option(FLUXVOICE_STRICT_WARNINGS "Enable strict compiler warnings"                 ON)
option(FLUXVOICE_WARNINGS_AS_ERRORS "Treat warnings from our sources as errors (CI)" OFF)
option(FLUXVOICE_BUILD_TESTS     "Register CTest smoke/unit tests"                  ON)

# ---------------------------------------------------------------------------
# Language settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Dependencies
#
# Header-only deps (miniaudio, readerwriterqueue) are fetched automatically.
# ONNX Runtime is a prebuilt SDK: pass -DONNXRUNTIME_ROOT=/path/to/sdk once the
# neural stages are wired (Milestone 2). See third_party/README.md.
# ---------------------------------------------------------------------------
if(FLUXVOICE_FETCH_DEPS)
    include(FetchContent)

    # SOURCE_SUBDIR trick: download the sources without add_subdirectory()-ing
    # the dependency's own CMakeLists.txt — we only need the headers.
    FetchContent_Declare(miniaudio
        GIT_REPOSITORY https://github.com/mackron/miniaudio.git
        GIT_TAG        0.11.25
        GIT_SHALLOW    TRUE
        SOURCE_SUBDIR  "__no_cmakelists__")

    FetchContent_Declare(readerwriterqueue
        GIT_REPOSITORY https://github.com/cameron314/readerwriterqueue.git
        GIT_TAG        v1.0.7
        GIT_SHALLOW    TRUE
        SOURCE_SUBDIR  "__no_cmakelists__")

    FetchContent_MakeAvailable(miniaudio readerwriterqueue)

    add_library(miniaudio_headers INTERFACE)
    target_include_directories(miniaudio_headers SYSTEM INTERFACE ${miniaudio_SOURCE_DIR})

    add_library(rwqueue_headers INTERFACE)
    target_include_directories(rwqueue_headers SYSTEM INTERFACE ${readerwriterqueue_SOURCE_DIR})
else()
    # Fallback: expect vendored copies under third_party/
    add_library(miniaudio_headers INTERFACE)
    target_include_directories(miniaudio_headers SYSTEM INTERFACE ${CMAKE_SOURCE_DIR}/third_party/miniaudio)

    add_library(rwqueue_headers INTERFACE)
    target_include_directories(rwqueue_headers SYSTEM INTERFACE ${CMAKE_SOURCE_DIR}/third_party/readerwriterqueue)
endif()

# miniaudio's implementation pulls in <windows.h>, whose min/max macros break
# std::min / std::max in every header included after it.
target_compile_definitions(miniaudio_headers INTERFACE $<$<PLATFORM_ID:Windows>:NOMINMAX>)

# Optional: prebuilt ONNX Runtime SDK (needed from Milestone 2).
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to a prebuilt ONNX Runtime SDK")
if(ONNXRUNTIME_ROOT)
    add_library(onnxruntime_sdk INTERFACE)
    target_include_directories(onnxruntime_sdk SYSTEM INTERFACE ${ONNXRUNTIME_ROOT}/include)
    target_link_directories(onnxruntime_sdk INTERFACE ${ONNXRUNTIME_ROOT}/lib)
    target_link_libraries(onnxruntime_sdk INTERFACE onnxruntime)
    message(STATUS "ONNX Runtime SDK: ${ONNXRUNTIME_ROOT}")
endif()

# ---------------------------------------------------------------------------
# Executable
# ---------------------------------------------------------------------------
add_executable(fluxvoice
    src/main.cpp
    src/pipeline/InferencePipeline.cpp
    src/ui/AppWindow.cpp
)

target_include_directories(fluxvoice PRIVATE src)

find_package(Threads REQUIRED)

# miniaudio + the platform audio backends it loads; for every target that
# defines MINIAUDIO_IMPLEMENTATION.
function(fluxvoice_link_audio target)
    target_link_libraries(${target} PRIVATE miniaudio_headers rwqueue_headers Threads::Threads)
    if(APPLE)
        target_link_libraries(${target} PRIVATE
            "-framework CoreAudio" "-framework AudioToolbox" "-framework AVFoundation"
            "-framework CoreFoundation")
    elseif(UNIX)
        target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS} m)
    elseif(WIN32)
        target_link_libraries(${target} PRIVATE ole32 winmm)
    endif()
endfunction()

fluxvoice_link_audio(fluxvoice)

if(ONNXRUNTIME_ROOT)
    target_link_libraries(fluxvoice PRIVATE onnxruntime_sdk)
    target_compile_definitions(fluxvoice PRIVATE FLUXVOICE_WITH_ORT=1)
endif()

# Route-measurement companion (M1 stages 05-06): records, plays a test tone or
# measures latency at the far end of a virtual-cable route. See tools/fv_probe.cpp.
add_executable(fv_probe tools/fv_probe.cpp)
target_include_directories(fv_probe PRIVATE tools)
fluxvoice_link_audio(fv_probe)

# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------
# Applied to every target built from our sources (app + tests).
function(fluxvoice_set_warnings target)
    if(FLUXVOICE_STRICT_WARNINGS)
        if(MSVC)
            target_compile_options(${target} PRIVATE /W4)
        else()
            target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    endif()

    # Dependencies are SYSTEM includes, so this only gates warnings from our code.
    if(FLUXVOICE_WARNINGS_AS_ERRORS)
        target_compile_options(${target} PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/WX,-Werror>)
    endif()
endfunction()

fluxvoice_set_warnings(fluxvoice)
fluxvoice_set_warnings(fv_probe)

# ---------------------------------------------------------------------------
# Tests  (run with: ctest --test-dir build)
#
# None of these need audio hardware: CLI argument errors are rejected before
# any device is opened, and the unit tests exercise the real-time path
# (DuplexRouter), chunking, the OLA crossfader, the feedback-route guard and
# the latency probe's detector without miniaudio.
# ---------------------------------------------------------------------------
if(FLUXVOICE_BUILD_TESTS)
    enable_testing()

    foreach(_t duplex_router chunk_assembler overlap_add route_guard latency_probe)
        add_executable(test_${_t} tests/test_${_t}.cpp)
        target_include_directories(test_${_t} PRIVATE src tests tools)
        target_link_libraries(test_${_t} PRIVATE rwqueue_headers Threads::Threads)
        fluxvoice_set_warnings(test_${_t})
        add_test(NAME ${_t} COMMAND test_${_t})
    endforeach()

    # OLA listening aid: writes reference / butt-joined / crossfaded WAVs into
    # the build directory. Run as a test so it keeps building and working.
    add_executable(ola_sine_test tools/ola_sine_test.cpp)
    target_include_directories(ola_sine_test PRIVATE src)
    fluxvoice_set_warnings(ola_sine_test)
    add_test(NAME ola_sine_wavs COMMAND ola_sine_test ${CMAKE_CURRENT_BINARY_DIR})

    # Soak-log analyzer (standard-library Python) checks its own verdicts.
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(NAME analyze_soak_selftest
                 COMMAND Python3::Interpreter -I ${CMAKE_SOURCE_DIR}/tools/analyze_soak.py --self-test)
    endif()

    add_test(NAME cli_help       COMMAND fluxvoice --help)
    add_test(NAME cli_help_short COMMAND fluxvoice -h)
    set_tests_properties(cli_help PROPERTIES PASS_REGULAR_EXPRESSION "Usage:")

    # Each of these must exit non-zero without touching an audio device.
    set(_bad_args_unknown        --bogus)
    set(_bad_args_missing_value  --input)
    set(_bad_args_non_numeric    --input abc)
    set(_bad_args_negative       --output -1)
    set(_bad_args_trailing_junk  --sample-rate 16000hz)
    set(_bad_args_rate_range     --sample-rate 0)
    set(_bad_args_duration_zero  --duration 0)
    set(_bad_args_stats_zero     --stats 0)
    foreach(_case unknown missing_value non_numeric negative trailing_junk rate_range
                  duration_zero stats_zero)
        add_test(NAME cli_rejects_${_case} COMMAND fluxvoice ${_bad_args_${_case}})
        set_tests_properties(cli_rejects_${_case} PROPERTIES WILL_FAIL TRUE)
    endforeach()

    add_test(NAME probe_help COMMAND fv_probe --help)
    set_tests_properties(probe_help PROPERTIES PASS_REGULAR_EXPRESSION "Usage:")

    # Rejected before an audio context is even created.
    set(_probe_args_no_command      --input 0)
    set(_probe_args_no_input        record out.wav)
    set(_probe_args_no_output       latency --input 0)
    set(_probe_args_no_wav          tone --input 0 --output 0)
    set(_probe_args_latency_wav     latency --input 0 --output 0 out.wav)
    set(_probe_args_short_interval  latency --input 0 --output 0 --interval-ms 50)
    set(_probe_args_two_commands    record latency --input 0 out.wav)
    foreach(_case no_command no_input no_output no_wav latency_wav short_interval two_commands)
        add_test(NAME probe_rejects_${_case} COMMAND fv_probe ${_probe_args_${_case}})
        set_tests_properties(probe_rejects_${_case} PROPERTIES WILL_FAIL TRUE)
    endforeach()
endif()
