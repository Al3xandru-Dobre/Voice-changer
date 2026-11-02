# Frontend (Milestone 4) — see frontend/README.md §8.
#
#   fluxvoice_frontend_core  toolkit-free model layer (always built, tested headless)
#   fluxvoice_gui            SDL3 + Dear ImGui desktop application (FLUXVOICE_BUILD_GUI)
#   test_voice_params, test_frontend_model   CTest, no display, no audio hardware

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
    ${CMAKE_SOURCE_DIR}/src/pipeline/InferencePipeline.cpp)
target_include_directories(fluxvoice_frontend_core PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fluxvoice_frontend_core PUBLIC nlohmann_json_headers rwqueue_headers Threads::Threads)
# Paths.cpp falls back to the source tree for presets/ and frontend/assets/ in
# development builds (installed builds find them next to the executable).
target_compile_definitions(fluxvoice_frontend_core PRIVATE FLUXVOICE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(WIN32)
    target_link_libraries(fluxvoice_frontend_core PUBLIC psapi)  # util/ProcessMemory.hpp
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
    fluxvoice_set_warnings(fluxvoice_gui)
endif()

if(FLUXVOICE_BUILD_TESTS)
    add_executable(test_voice_params
        ${CMAKE_SOURCE_DIR}/tests/test_voice_params.cpp
        ${CMAKE_SOURCE_DIR}/src/voice/Presets.cpp)
    target_include_directories(test_voice_params PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_voice_params PRIVATE nlohmann_json_headers)
    fluxvoice_set_warnings(test_voice_params)
    add_test(NAME voice_params COMMAND test_voice_params ${CMAKE_SOURCE_DIR}/presets)

    add_executable(test_frontend_model ${CMAKE_SOURCE_DIR}/tests/test_frontend_model.cpp)
    target_include_directories(test_frontend_model PRIVATE ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(test_frontend_model PRIVATE fluxvoice_frontend_core)
    fluxvoice_set_warnings(test_frontend_model)
    add_test(NAME frontend_model COMMAND test_frontend_model ${CMAKE_SOURCE_DIR})

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
    endif()
endif()
