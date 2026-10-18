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

    # nlohmann/json (Milestone 2: model contracts in Stage 02, presets in Stage 09).
    # The release tarball, pinned by hash; only its headers are used.
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)   # extracted files get the extraction time
    endif()
    FetchContent_Declare(nlohmann_json
        URL           https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
        URL_HASH      SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
        SOURCE_SUBDIR "__no_cmakelists__")

    FetchContent_MakeAvailable(miniaudio readerwriterqueue nlohmann_json)

    add_library(miniaudio_headers INTERFACE)
    target_include_directories(miniaudio_headers SYSTEM INTERFACE ${miniaudio_SOURCE_DIR})

    add_library(rwqueue_headers INTERFACE)
    target_include_directories(rwqueue_headers SYSTEM INTERFACE ${readerwriterqueue_SOURCE_DIR})

    add_library(nlohmann_json_headers INTERFACE)
    target_include_directories(nlohmann_json_headers SYSTEM INTERFACE ${nlohmann_json_SOURCE_DIR}/include)
else()
    # Fallback: expect vendored copies under third_party/
    add_library(miniaudio_headers INTERFACE)
    target_include_directories(miniaudio_headers SYSTEM INTERFACE ${CMAKE_SOURCE_DIR}/third_party/miniaudio)

    add_library(rwqueue_headers INTERFACE)
    target_include_directories(rwqueue_headers SYSTEM INTERFACE ${CMAKE_SOURCE_DIR}/third_party/readerwriterqueue)

    add_library(nlohmann_json_headers INTERFACE)
    target_include_directories(nlohmann_json_headers SYSTEM INTERFACE ${CMAKE_SOURCE_DIR}/third_party/nlohmann_json/include)
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
    src/audio/AudioController.cpp
    src/audio/AudioDeviceRegistry.cpp
    src/audio/MiniaudioBackend.cpp
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

    foreach(_t duplex_router chunk_assembler overlap_add route_guard latency_probe audio_types
               device_registry audio_controller)
        add_executable(test_${_t} tests/test_${_t}.cpp)
        target_include_directories(test_${_t} PRIVATE src tests tools)
        target_link_libraries(test_${_t} PRIVATE rwqueue_headers Threads::Threads)
        fluxvoice_set_warnings(test_${_t})
        add_test(NAME ${_t} COMMAND test_${_t})
    endforeach()

    # The registry and controller tests link their implementations over a fake
    # backend (no miniaudio, no hardware).
    target_sources(test_device_registry PRIVATE src/audio/AudioDeviceRegistry.cpp)
    target_sources(test_audio_controller PRIVATE
        src/audio/AudioController.cpp
        src/audio/AudioDeviceRegistry.cpp
        src/pipeline/InferencePipeline.cpp)

    # M2 stage 02: model contract v1, ORT-free. Runs the cases in
    # tests/data/contracts/ (shared with the Python twin) and the fixture manifest.
    add_executable(test_model_contract tests/test_model_contract.cpp)
    target_include_directories(test_model_contract PRIVATE src tests)
    target_link_libraries(test_model_contract PRIVATE nlohmann_json_headers)
    fluxvoice_set_warnings(test_model_contract)
    add_test(NAME model_contract COMMAND test_model_contract ${CMAKE_SOURCE_DIR}/tests/data)

    # BUG-008: the root .gitignore keeps tests/data/ tracked and ignores the M2
    # data root, weights and run output. Needs a git checkout (not a tarball).
    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS ${CMAKE_SOURCE_DIR}/.git)
        add_test(NAME gitignore_rules
                 COMMAND ${CMAKE_COMMAND} -DGIT=${GIT_EXECUTABLE} -DSRC=${CMAKE_SOURCE_DIR}
                         -P ${CMAKE_SOURCE_DIR}/tests/check_gitignore.cmake)
    endif()

    # M2 stage 03: causal YIN tracker vs the Python golden vectors (ORT-free).
    # Picked up at configure time once tests/test_pitch_tracker.cpp exists.
    if(EXISTS ${CMAKE_SOURCE_DIR}/tests/test_pitch_tracker.cpp)
        add_executable(test_pitch_tracker tests/test_pitch_tracker.cpp)
        target_include_directories(test_pitch_tracker PRIVATE src tests)
        fluxvoice_set_warnings(test_pitch_tracker)
        add_test(NAME pitch_tracker COMMAND test_pitch_tracker ${CMAKE_SOURCE_DIR}/tests/data/pitch)
    endif()

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

        # Voicemeeter rig configurator: keeps the CLI well-formed in CI (the
        # script itself needs Voicemeeter and only runs on the rig machine).
        add_test(NAME vm_route_help
                 COMMAND Python3::Interpreter -I ${CMAKE_SOURCE_DIR}/tools/vm_route.py --help)
        set_tests_properties(vm_route_help PROPERTIES PASS_REGULAR_EXPRESSION "usage")
        add_test(NAME vm_route_selftest
                 COMMAND Python3::Interpreter -I ${CMAKE_SOURCE_DIR}/tools/vm_route.py --self-test)

        # M2 stage 02: the Python twin of the contract validator, and the committed
        # fixtures against their manifest. Both standard library only.
        add_test(NAME model_contract_py
                 COMMAND Python3::Interpreter -I ${CMAKE_SOURCE_DIR}/python_training/tests/test_contract.py)
        add_test(NAME fixture_models_selftest
                 COMMAND Python3::Interpreter -I ${CMAKE_SOURCE_DIR}/tools/make_fixture_models.py --self-test)
    endif()

    # M2 stages 01-02 with the pinned training environment (Python 3.12, onnx,
    # onnxruntime, ...; `uv sync --frozen` in python_training/). Registered only
    # when that environment exists, so plain CI builds skip it.
    if(WIN32)
        set(_training_python ${CMAKE_SOURCE_DIR}/python_training/.venv/Scripts/python.exe)
    else()
        set(_training_python ${CMAKE_SOURCE_DIR}/python_training/.venv/bin/python)
    endif()
    if(EXISTS ${_training_python})
        add_test(NAME python_training_stage01_02
                 COMMAND ${_training_python} -m pytest -q -p no:cacheprovider
                         tests/test_contract.py tests/test_fixtures.py tests/test_prepare_data.py
                         tests/test_eval_tools.py
                 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python_training)
        add_test(NAME python_training_stage03_04
                 COMMAND ${_training_python} -m pytest -q -p no:cacheprovider tests/test_features.py
                         tests/test_streaming.py tests/test_content_encoder.py
                 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python_training)
        set_tests_properties(python_training_stage03_04 PROPERTIES TIMEOUT 300)
        add_test(NAME python_training_stage05
                 COMMAND ${_training_python} -m pytest -q -p no:cacheprovider tests/test_vocoder.py
                 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python_training)
        set_tests_properties(python_training_stage05 PROPERTIES TIMEOUT 300
                             ENVIRONMENT "PYTHONIOENCODING=utf-8")
        add_test(NAME python_training_stage06
                 COMMAND ${_training_python} -m pytest -q -p no:cacheprovider tests/test_export.py
                 WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/python_training)
        # The dynamo exporter prints emoji; keep a cp1252 console pipe from crashing it.
        set_tests_properties(python_training_stage06 PROPERTIES TIMEOUT 600
                             ENVIRONMENT "PYTHONIOENCODING=utf-8")
    endif()

    add_test(NAME cli_help       COMMAND fluxvoice --help)
    add_test(NAME cli_help_short COMMAND fluxvoice -h)
    set_tests_properties(cli_help PROPERTIES PASS_REGULAR_EXPRESSION "Usage:")

    # M1.5 stage 04: the help documents index|default device selection.
    add_test(NAME cli_help_selection COMMAND fluxvoice --help)
    set_tests_properties(cli_help_selection PROPERTIES PASS_REGULAR_EXPRESSION "input.*default")

    # Each of these must exit non-zero without touching an audio device.
    set(_bad_args_unknown        --bogus)
    set(_bad_args_missing_value  --input)
    set(_bad_args_non_numeric    --input abc)
    set(_bad_args_negative       --output -1)
    set(_bad_args_trailing_junk  --sample-rate 16000hz)
    set(_bad_args_rate_range     --sample-rate 0)
    set(_bad_args_duration_zero  --duration 0)
    set(_bad_args_stats_zero     --stats 0)
    set(_bad_args_input_word         --input mic1)
    set(_bad_args_input_float        --input 1.5)
    set(_bad_args_input_upper        --input DEFAULT)
    set(_bad_args_input_and_loopback --input 0 --loopback-input 0)
    foreach(_case unknown missing_value non_numeric negative trailing_junk rate_range
                  duration_zero stats_zero input_word input_float input_upper input_and_loopback)
        add_test(NAME cli_rejects_${_case} COMMAND fluxvoice ${_bad_args_${_case}})
        set_tests_properties(cli_rejects_${_case} PROPERTIES WILL_FAIL TRUE)
    endforeach()

    add_test(NAME probe_help COMMAND fv_probe --help)
    set_tests_properties(probe_help PROPERTIES PASS_REGULAR_EXPRESSION "Usage:")

    # Rejected before an audio context is even created.
    set(_probe_args_no_command         --input 0)
    set(_probe_args_no_input           record out.wav)
    set(_probe_args_no_output          latency --input 0)
    set(_probe_args_no_wav             tone --input 0 --output 0)
    set(_probe_args_latency_wav        latency --input 0 --output 0 out.wav)
    set(_probe_args_short_interval     latency --input 0 --output 0 --interval-ms 50)
    set(_probe_args_two_commands       record latency --input 0 out.wav)
    set(_probe_args_name_and_index     tone --input 0 --input-name CABLE --output 0 out.wav)
    set(_probe_args_loopback_with_tone tone --loopback --input 0 --output 0 out.wav)
    foreach(_case no_command no_input no_output no_wav latency_wav short_interval two_commands
                  name_and_index loopback_with_tone)
        add_test(NAME probe_rejects_${_case} COMMAND fv_probe ${_probe_args_${_case}})
        set_tests_properties(probe_rejects_${_case} PROPERTIES WILL_FAIL TRUE)
    endforeach()
endif()
