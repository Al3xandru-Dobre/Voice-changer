# fluxvoice_header_checks -- checks every first-party header for independent
# inclusion.
#
# Each generated TU includes one header twice and nothing else, so a header
# that silently depended on being preceded by other includes fails to compile
# here. Transitive includes *inside* a header can still mask a missing owning
# include -- this target guards independent inclusion; pair it with a manual
# include-what-you-use review. One TU per header keeps failures attributable.
# UNITY_BUILD is forced OFF and precompiled headers disabled so global
# settings cannot supply declarations the header forgot.

file(GLOB_RECURSE _fv_check_headers CONFIGURE_DEPENDS
     "${CMAKE_SOURCE_DIR}/src/*.hpp")
list(APPEND _fv_check_headers
     "${CMAKE_SOURCE_DIR}/tools/LatencyProbe.hpp"
     "${CMAKE_SOURCE_DIR}/tests/TestSupport.hpp")

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/header_checks")

set(_fv_check_srcs)
foreach(_hdr IN LISTS _fv_check_headers)
    # The include text is the path relative to the include root that owns it
    # (src/, tools/ or tests/); the target adds those roots to its include
    # path. TU names are derived deterministically from the project-relative
    # path; MAKE_C_IDENTIFIER is not injective (foo-bar vs foo_bar), so a
    # collision fails the configure loudly instead of silently dropping a TU.
    file(RELATIVE_PATH _rel "${CMAKE_SOURCE_DIR}" "${_hdr}")
    string(REGEX REPLACE "^(src|tools|tests)/" "" _inc "${_rel}")
    string(MAKE_C_IDENTIFIER "${_rel}" _stem)
    set(_tu "${CMAKE_CURRENT_BINARY_DIR}/header_checks/${_stem}.cpp")
    if(_tu IN_LIST _fv_check_srcs)
        message(FATAL_ERROR
                "header-check TU name collision: ${_rel} -> ${_stem}.cpp")
    endif()
    list(APPEND _fv_check_srcs "${_tu}")
    # file(GENERATE) writes only when content differs, so a no-op reconfigure
    # leaves every TU's timestamp (and its object file) untouched.
    file(GENERATE OUTPUT "${_tu}"
         CONTENT "#include \"${_inc}\"\n#include \"${_inc}\"\n")
endforeach()

add_library(fluxvoice_header_checks OBJECT ${_fv_check_srcs})
target_include_directories(fluxvoice_header_checks PRIVATE src tools tests)
target_link_libraries(fluxvoice_header_checks PRIVATE
    miniaudio_headers rwqueue_headers nlohmann_json_headers Threads::Threads)
fluxvoice_set_warnings(fluxvoice_header_checks)
set_target_properties(fluxvoice_header_checks PROPERTIES
    UNITY_BUILD OFF DISABLE_PRECOMPILE_HEADERS ON)
