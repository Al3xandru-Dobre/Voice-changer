# Frontend (Milestone 4) — see frontend/README.md §8.
#
#   fluxvoice_frontend_core  toolkit-free model layer (always built, tested headless)
#   fluxvoice_gui            SDL3 + Dear ImGui desktop application (FLUXVOICE_BUILD_GUI)
#   test_frontend_model, test_frontend_shell,
#   test_frontend_studio                     CTest, no display, no audio hardware

add_library(fluxvoice_frontend_core STATIC
    core/AppModel.cpp
    core/DeviceSelectorModel.cpp
    core/EngineHost.cpp
    core/Paths.cpp
    core/Settings.cpp
    core/Telemetry.cpp
    core/TranslationModel.cpp
    core/TrayModel.cpp
    core/VoiceCatalog.cpp
    core/VoiceStudio.cpp
    ${CMAKE_SOURCE_DIR}/src/voice/Presets.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/AudioController.cpp
    ${CMAKE_SOURCE_DIR}/src/audio/AudioDeviceRegistry.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/InferencePipeline.cpp
    # M4 stage 05: the Mode A host shared with the CLI, and the bundle loader.
    # OrtModel.cpp compiles to stubs that report kBuiltWithoutOrtMessage when
    # ONNX Runtime is absent, so the ORT-less GUI still links and runs.
    ${CMAKE_SOURCE_DIR}/src/pipeline/ModeAHost.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/ModelBundle.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/OrtBundleLoader.cpp
    ${CMAKE_SOURCE_DIR}/src/pipeline/OrtModel.cpp)
target_include_directories(fluxvoice_frontend_core PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fluxvoice_frontend_core PUBLIC nlohmann_json_headers rwqueue_headers Threads::Threads)
# Paths.cpp falls back to the source tree for presets/ and frontend/assets/ in
# development builds (installed builds find them next to the executable).
target_compile_definitions(fluxvoice_frontend_core PRIVATE FLUXVOICE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(WIN32)
    # util/ProcessMemory.hpp (psapi); util/WorkerPriority.hpp, MMCSS (avrt).
    target_link_libraries(fluxvoice_frontend_core PUBLIC psapi avrt)
endif()
if(FLUXVOICE_HAS_ORT)
    # Every executable below that links the core also calls
    # fluxvoice_link_ort (RPATH + runtime staging).
    fluxvoice_compile_ort(fluxvoice_frontend_core)
endif()
fluxvoice_set_warnings(fluxvoice_frontend_core)

if(FLUXVOICE_BUILD_GUI)
    # WIN32: no console window on Windows (SDL_main.h supplies the entry point);
    # ignored elsewhere.
    add_executable(fluxvoice_gui WIN32
        app/main.cpp
        platform/SdlTray.cpp
        imgui/GuiApp.cpp
        imgui/Theme.cpp
        imgui/Widgets.cpp
        imgui/ViewLive.cpp
        imgui/ViewDevices.cpp
        imgui/ViewVoices.cpp
        imgui/ViewStudio.cpp
        imgui/ViewTranslate.cpp
        imgui/ViewCatalogue.cpp
        imgui/ViewTelemetry.cpp
        imgui/ViewSettings.cpp
        imgui/ViewMiniPanel.cpp
        ${CMAKE_SOURCE_DIR}/src/audio/MiniaudioBackend.cpp)  # the one MINIAUDIO_IMPLEMENTATION TU of this binary
    target_include_directories(fluxvoice_gui PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(fluxvoice_gui PRIVATE fluxvoice_frontend_core fluxvoice_imgui SDL3::SDL3-static)
    target_compile_definitions(fluxvoice_gui PRIVATE
        FLUXVOICE_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        FLUXVOICE_VERSION="${PROJECT_VERSION}")
    fluxvoice_link_audio(fluxvoice_gui)
    if(FLUXVOICE_HAS_ORT)
        fluxvoice_link_ort(fluxvoice_gui)
    endif()
    fluxvoice_set_warnings(fluxvoice_gui)
endif()

if(FLUXVOICE_BUILD_TESTS)
    # test_voice_params moved to the root CMakeLists (it now links
    # InferencePipeline for the pipeline-level control tests).
    add_executable(test_frontend_model ${CMAKE_SOURCE_DIR}/tests/test_frontend_model.cpp)
    target_include_directories(test_frontend_model PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_frontend_model PRIVATE fluxvoice_frontend_core)
    if(FLUXVOICE_HAS_ORT)
        fluxvoice_link_ort(test_frontend_model)
    endif()
    fluxvoice_set_warnings(test_frontend_model)
    # The second argument receives GUI-produced [stats] lines (Bypass and Mode
    # A over the fake models); gui_stats_parse then checks them with the same
    # parser that reads CLI soak logs (M4 Stage 06).
    set(_fv_gui_stats ${CMAKE_CURRENT_BINARY_DIR}/gui_stats_sample.log)
    add_test(NAME frontend_model COMMAND test_frontend_model ${CMAKE_SOURCE_DIR} ${_fv_gui_stats})
    set_tests_properties(frontend_model PROPERTIES FIXTURES_SETUP gui_stats_sample)
    if(TARGET Python3::Interpreter)
        add_test(NAME gui_stats_parse
                 COMMAND Python3::Interpreter -I ${CMAKE_SOURCE_DIR}/tools/analyze_soak.py
                         --check-format ${_fv_gui_stats})
        set_tests_properties(gui_stats_parse PROPERTIES FIXTURES_REQUIRED gui_stats_sample)
    endif()

    # Stage 01 shell residuals (paths, settings robustness, URL policy) and
    # Stage 03 Voice Studio residuals (WAV hardening, async analysis, job
    # commands) — separate executables so each failure is attributable.
    foreach(_fv_suite frontend_shell frontend_studio)
        add_executable(test_${_fv_suite} ${CMAKE_SOURCE_DIR}/tests/test_${_fv_suite}.cpp)
        target_include_directories(test_${_fv_suite} PRIVATE ${CMAKE_SOURCE_DIR}/tests)
        target_link_libraries(test_${_fv_suite} PRIVATE fluxvoice_frontend_core)
        if(FLUXVOICE_HAS_ORT)
            fluxvoice_link_ort(test_${_fv_suite})
        endif()
        fluxvoice_set_warnings(test_${_fv_suite})
        add_test(NAME ${_fv_suite} COMMAND test_${_fv_suite} ${CMAKE_SOURCE_DIR})
    endforeach()

    if(TARGET fluxvoice_gui)
        # Argument handling runs before SDL is initialised, so these need no
        # display. Exit codes only: a WIN32-subsystem process may have no
        # console to capture.
        add_test(NAME gui_help COMMAND fluxvoice_gui --help)
        set(_gui_bad_unknown      --bogus)
        set(_gui_bad_missing      --models-dir)
        set(_gui_bad_frames       --smoke-frames abc)
        set(_gui_bad_screenshot   --screenshot)
        set(_gui_bad_section      --section nowhere)
        foreach(_case unknown missing frames screenshot section)
            add_test(NAME gui_rejects_${_case} COMMAND fluxvoice_gui ${_gui_bad_${_case}})
            set_tests_properties(gui_rejects_${_case} PROPERTIES WILL_FAIL TRUE)
        endforeach()

        # Headless GUI smoke test (frontend/README.md §8): xvfb-run + Mesa
        # llvmpipe render 40 frames of one section (or the Mini panel) under
        # the software renderer; any "[imgui-error]" line fails the test.
        # Every writable location is redirected under a per-test directory in
        # the build tree (--config-dir also covers the settings file and the
        # default user-preset dir, --studio-dir the Voice Studio projects),
        # so the GUI never touches ~/.config or the user's data and the tests
        # can run in parallel. An empty --models-dir makes the catalogue fall
        # back to the read-only fixture bundle. Registered only where
        # xvfb-run exists: the Linux CI legs install it; macOS and display-less
        # developer machines use the manual README §8 command instead.
        if(UNIX AND NOT APPLE)
            find_program(FV_XVFB_RUN xvfb-run)
            if(FV_XVFB_RUN)
                foreach(_fv_smoke live devices voices studio translate
                                  catalogue telemetry settings)
                    set(_fv_smoke_dir ${CMAKE_CURRENT_BINARY_DIR}/gui_smoke/${_fv_smoke})
                    file(MAKE_DIRECTORY ${_fv_smoke_dir}/config ${_fv_smoke_dir}/studio ${_fv_smoke_dir}/models)
                    add_test(NAME gui_smoke_${_fv_smoke}
                        COMMAND ${FV_XVFB_RUN} -a -s "-screen 0 1280x800x24"
                                $<TARGET_FILE:fluxvoice_gui> --smoke-frames 40
                                --section ${_fv_smoke}
                                --config-dir ${_fv_smoke_dir}/config
                                --studio-dir ${_fv_smoke_dir}/studio
                                --models-dir ${_fv_smoke_dir}/models)
                    set_tests_properties(gui_smoke_${_fv_smoke} PROPERTIES
                        ENVIRONMENT "LIBGL_ALWAYS_SOFTWARE=1"
                        FAIL_REGULAR_EXPRESSION "\\[imgui-error\\]"
                        TIMEOUT 120
                        LABELS "gui_smoke")
                endforeach()
                set(_fv_smoke_dir ${CMAKE_CURRENT_BINARY_DIR}/gui_smoke/mini)
                file(MAKE_DIRECTORY ${_fv_smoke_dir}/config ${_fv_smoke_dir}/studio ${_fv_smoke_dir}/models)
                add_test(NAME gui_smoke_mini
                    COMMAND ${FV_XVFB_RUN} -a -s "-screen 0 1280x800x24"
                            $<TARGET_FILE:fluxvoice_gui> --smoke-frames 40 --mini
                            --config-dir ${_fv_smoke_dir}/config
                            --studio-dir ${_fv_smoke_dir}/studio
                            --models-dir ${_fv_smoke_dir}/models)
                set_tests_properties(gui_smoke_mini PROPERTIES
                    ENVIRONMENT "LIBGL_ALWAYS_SOFTWARE=1"
                    FAIL_REGULAR_EXPRESSION "\\[imgui-error\\]"
                    TIMEOUT 120
                    LABELS "gui_smoke")

                # M4 stage 05 (ORT builds): load and warm up the identity
                # fixture bundle from the command line (no stream is opened).
                # Passes only on the "[bundle] loaded" line, so a silent
                # Bypass fallback fails the test.
                if(FLUXVOICE_HAS_ORT)
                    set(_fv_smoke_dir ${CMAKE_CURRENT_BINARY_DIR}/gui_smoke/bundle)
                    file(MAKE_DIRECTORY ${_fv_smoke_dir}/config ${_fv_smoke_dir}/studio)
                    add_test(NAME gui_smoke_bundle
                        COMMAND ${FV_XVFB_RUN} -a -s "-screen 0 1280x800x24"
                                $<TARGET_FILE:fluxvoice_gui> --smoke-frames 40 --section live
                                --config-dir ${_fv_smoke_dir}/config
                                --studio-dir ${_fv_smoke_dir}/studio
                                --models-dir ${CMAKE_SOURCE_DIR}/tests/data/fixtures
                                --bundle ${CMAKE_SOURCE_DIR}/tests/data/fixtures/identity)
                    set_tests_properties(gui_smoke_bundle PROPERTIES
                        ENVIRONMENT "LIBGL_ALWAYS_SOFTWARE=1"
                        PASS_REGULAR_EXPRESSION "\\[bundle\\] loaded"
                        FAIL_REGULAR_EXPRESSION "\\[imgui-error\\]|\\[bundle\\] error"
                        TIMEOUT 120
                        LABELS "gui_smoke")
                endif()
            endif()
        endif()
    endif()
endif()
