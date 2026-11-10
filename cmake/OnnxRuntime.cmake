# OnnxRuntime.cmake — ONNX Runtime C++ SDK import & runtime library staging
# (Milestone 2 / Stage 07, Documents/Milestone 2/07_ort_cpp_runtime.md).
#
# Usage:
#   - Default (ORT-less CI build): neither ONNXRUNTIME_ROOT nor
#     FLUXVOICE_FETCH_ORT is set -> FLUXVOICE_HAS_ORT is FALSE and no target
#     defines FLUXVOICE_WITH_ORT.
#   - Prebuilt SDK on disk: -DONNXRUNTIME_ROOT=/path/to/onnxruntime-<os>-<arch>-1.31.0
#   - Auto-fetched SDK:     -DFLUXVOICE_FETCH_ORT=ON (downloads the pinned
#     v1.31.0 archive for the host OS/arch and verifies its SHA256).
#
# Creates the IMPORTED SHARED target `onnxruntime::onnxruntime` with SYSTEM
# headers and provides `fluxvoice_link_ort(<target>)`, which links ORT,
# defines FLUXVOICE_WITH_ORT=1, configures $ORIGIN / @loader_path RPATH on
# Unix/macOS, and copies the shared library next to the executable so Windows
# never loads an older C:\Windows\System32\onnxruntime.dll.

set(FLUXVOICE_ORT_VERSION "1.31.0")

option(FLUXVOICE_FETCH_ORT "Fetch the pinned ONNX Runtime C++ SDK via FetchContent" OFF)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to a prebuilt ONNX Runtime SDK")

set(FLUXVOICE_HAS_ORT FALSE)
set(FLUXVOICE_ORT_ROOT "")

if(ONNXRUNTIME_ROOT)
    if(EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
        set(FLUXVOICE_ORT_ROOT "${ONNXRUNTIME_ROOT}")
    else()
        file(GLOB _ort_candidates "${ONNXRUNTIME_ROOT}/onnxruntime-*/include/onnxruntime_cxx_api.h")
        list(LENGTH _ort_candidates _ort_n)
        if(_ort_n EQUAL 1)
            get_filename_component(_ort_inc "${_ort_candidates}" DIRECTORY)
            get_filename_component(FLUXVOICE_ORT_ROOT "${_ort_inc}" DIRECTORY)
        else()
            message(FATAL_ERROR
                "ONNXRUNTIME_ROOT '${ONNXRUNTIME_ROOT}' does not contain "
                "include/onnxruntime_cxx_api.h")
        endif()
    endif()
elseif(FLUXVOICE_FETCH_ORT)
    include(FetchContent)
    if(POLICY CMP0135)
        cmake_policy(SET CMP0135 NEW)
    endif()

    set(_ort_base_url "https://github.com/microsoft/onnxruntime/releases/download/v${FLUXVOICE_ORT_VERSION}")
    set(_ort_arch "${CMAKE_SYSTEM_PROCESSOR}")
    if(MSVC AND CMAKE_VS_PLATFORM_NAME)
        set(_ort_arch "${CMAKE_VS_PLATFORM_NAME}")
    endif()

    if(WIN32)
        if(_ort_arch MATCHES "^(ARM64|arm64|aarch64)$")
            set(_ort_archive "onnxruntime-win-arm64-${FLUXVOICE_ORT_VERSION}.zip")
            set(_ort_sha256  "ef6206fa8d6975878d93a97035d280240ce5764c126a7430e5427e18923316d5")
        else()
            set(_ort_archive "onnxruntime-win-x64-${FLUXVOICE_ORT_VERSION}.zip")
            set(_ort_sha256  "3958d8a44984160692c1b4ffd0de3cb48f8c39f9d34bb31e433db06e30b82195")
        endif()
    elseif(APPLE)
        set(_ort_archive "onnxruntime-osx-arm64-${FLUXVOICE_ORT_VERSION}.tgz")
        set(_ort_sha256  "1031b36dd3bdaa8d976a6130d9cf2a05e9be320b93ae804f8427b91dc3432afc")
    elseif(UNIX)
        if(_ort_arch MATCHES "^(aarch64|arm64|ARM64)$")
            set(_ort_archive "onnxruntime-linux-aarch64-${FLUXVOICE_ORT_VERSION}.tgz")
            set(_ort_sha256  "c5b8b3cca31f3d643a3b313b8f42e2d0f16a4fcefd02cda343d97bd5306afaab")
        else()
            set(_ort_archive "onnxruntime-linux-x64-${FLUXVOICE_ORT_VERSION}.tgz")
            set(_ort_sha256  "cc5c72baf5ae5c8238a6841f0897227be2d02826b9cf98eaf02fdefaeeb03a57")
        endif()
    else()
        message(FATAL_ERROR "FLUXVOICE_FETCH_ORT is not supported on ${CMAKE_SYSTEM_NAME}")
    endif()

    FetchContent_Declare(onnxruntime_prebuilt
        URL           "${_ort_base_url}/${_ort_archive}"
        URL_HASH      "SHA256=${_ort_sha256}"
        SOURCE_SUBDIR "__no_cmakelists__")
    FetchContent_MakeAvailable(onnxruntime_prebuilt)
    set(FLUXVOICE_ORT_ROOT "${onnxruntime_prebuilt_SOURCE_DIR}")
endif()

if(FLUXVOICE_ORT_ROOT)
    file(TO_CMAKE_PATH "${FLUXVOICE_ORT_ROOT}" FLUXVOICE_ORT_ROOT)
    if(NOT EXISTS "${FLUXVOICE_ORT_ROOT}/include/onnxruntime_cxx_api.h")
        message(FATAL_ERROR
            "ONNX Runtime header not found at '${FLUXVOICE_ORT_ROOT}/include/onnxruntime_cxx_api.h'")
    endif()

    set(_ort_ver "${FLUXVOICE_ORT_VERSION}")
    if(EXISTS "${FLUXVOICE_ORT_ROOT}/VERSION_NUMBER")
        file(STRINGS "${FLUXVOICE_ORT_ROOT}/VERSION_NUMBER" _ort_ver_lines LIMIT_COUNT 1)
        if(_ort_ver_lines)
            string(STRIP "${_ort_ver_lines}" _ort_ver)
        endif()
    endif()

    add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
    target_include_directories(onnxruntime::onnxruntime SYSTEM INTERFACE
        "${FLUXVOICE_ORT_ROOT}/include"
    )

    if(WIN32)
        set(_ort_implib "${FLUXVOICE_ORT_ROOT}/lib/onnxruntime.lib")
        set(_ort_shlib  "${FLUXVOICE_ORT_ROOT}/lib/onnxruntime.dll")
        if(NOT EXISTS "${_ort_implib}")
            message(FATAL_ERROR "ONNX Runtime import library not found: ${_ort_implib}")
        endif()
        if(NOT EXISTS "${_ort_shlib}")
            message(FATAL_ERROR "ONNX Runtime DLL not found: ${_ort_shlib}")
        endif()
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_IMPLIB   "${_ort_implib}"
            IMPORTED_LOCATION "${_ort_shlib}"
        )
    elseif(APPLE)
        set(_ort_shlib "${FLUXVOICE_ORT_ROOT}/lib/libonnxruntime.${_ort_ver}.dylib")
        if(NOT EXISTS "${_ort_shlib}")
            set(_ort_shlib "${FLUXVOICE_ORT_ROOT}/lib/libonnxruntime.dylib")
        endif()
        if(NOT EXISTS "${_ort_shlib}")
            message(FATAL_ERROR "ONNX Runtime dylib not found under '${FLUXVOICE_ORT_ROOT}/lib'")
        endif()
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${_ort_shlib}"
        )
    else()
        set(_ort_shlib "${FLUXVOICE_ORT_ROOT}/lib/libonnxruntime.so.${_ort_ver}")
        if(NOT EXISTS "${_ort_shlib}")
            set(_ort_shlib "${FLUXVOICE_ORT_ROOT}/lib/libonnxruntime.so")
        endif()
        if(NOT EXISTS "${_ort_shlib}")
            message(FATAL_ERROR "ONNX Runtime shared library not found under '${FLUXVOICE_ORT_ROOT}/lib'")
        endif()
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${_ort_shlib}"
            IMPORTED_SONAME   "libonnxruntime.so.1"
        )
    endif()

    set(FLUXVOICE_ORT_SHARED_LIB "${_ort_shlib}")
    set(FLUXVOICE_ORT_LIB_VERSION "${_ort_ver}")
    set(FLUXVOICE_HAS_ORT TRUE)
    message(STATUS "ONNX Runtime SDK (${_ort_ver}): ${FLUXVOICE_ORT_ROOT}")
endif()

# Link a target against onnxruntime::onnxruntime, define FLUXVOICE_WITH_ORT=1,
# configure RPATH, and stage the shared library next to the executable.
function(fluxvoice_link_ort target)
    if(NOT FLUXVOICE_HAS_ORT)
        message(FATAL_ERROR "fluxvoice_link_ort(${target}) called when FLUXVOICE_HAS_ORT is FALSE")
    endif()

    target_link_libraries(${target} PRIVATE onnxruntime::onnxruntime)
    target_compile_definitions(${target} PRIVATE
        FLUXVOICE_WITH_ORT=1
        FLUXVOICE_ORT_VERSION="${FLUXVOICE_ORT_VERSION}"
    )

    if(APPLE)
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH   "@loader_path"
            INSTALL_RPATH "@loader_path"
        )
    elseif(UNIX)
        set_target_properties(${target} PROPERTIES
            BUILD_RPATH   "$ORIGIN"
            INSTALL_RPATH "$ORIGIN"
        )
    endif()

    if(WIN32)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:${target}>
                $<TARGET_FILE_DIR:${target}>
            COMMAND_EXPAND_LISTS
            COMMENT "Copying ONNX Runtime DLL next to ${target}"
        )
        if(EXISTS "${FLUXVOICE_ORT_ROOT}/lib/onnxruntime_providers_shared.dll")
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${FLUXVOICE_ORT_ROOT}/lib/onnxruntime_providers_shared.dll"
                    $<TARGET_FILE_DIR:${target}>
            )
        endif()
    elseif(APPLE)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FLUXVOICE_ORT_SHARED_LIB}"
                "$<TARGET_FILE_DIR:${target}>/libonnxruntime.${FLUXVOICE_ORT_LIB_VERSION}.dylib"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FLUXVOICE_ORT_SHARED_LIB}"
                "$<TARGET_FILE_DIR:${target}>/libonnxruntime.1.dylib"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FLUXVOICE_ORT_SHARED_LIB}"
                "$<TARGET_FILE_DIR:${target}>/libonnxruntime.dylib"
            COMMENT "Copying ONNX Runtime dylib next to ${target}"
        )
        if(EXISTS "${FLUXVOICE_ORT_ROOT}/lib/libonnxruntime_providers_shared.dylib")
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${FLUXVOICE_ORT_ROOT}/lib/libonnxruntime_providers_shared.dylib"
                    $<TARGET_FILE_DIR:${target}>
            )
        endif()
    else()
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FLUXVOICE_ORT_SHARED_LIB}"
                "$<TARGET_FILE_DIR:${target}>/libonnxruntime.so.${FLUXVOICE_ORT_LIB_VERSION}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FLUXVOICE_ORT_SHARED_LIB}"
                "$<TARGET_FILE_DIR:${target}>/libonnxruntime.so.1"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${FLUXVOICE_ORT_SHARED_LIB}"
                "$<TARGET_FILE_DIR:${target}>/libonnxruntime.so"
            COMMENT "Copying ONNX Runtime shared library next to ${target}"
        )
        if(EXISTS "${FLUXVOICE_ORT_ROOT}/lib/libonnxruntime_providers_shared.so")
            add_custom_command(TARGET ${target} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${FLUXVOICE_ORT_ROOT}/lib/libonnxruntime_providers_shared.so"
                    $<TARGET_FILE_DIR:${target}>
            )
        endif()
    endif()
endfunction()
