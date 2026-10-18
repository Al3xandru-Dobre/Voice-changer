# fluxvoice_header_checks -- compiles every first-party header in isolation.
#
# Each generated TU includes one header twice (the second include proves the
# guard/pragma-once idempotency) and nothing else, so a header that relies on
# a transitive include fails to compile here. One TU per header keeps failures
# attributable. UNITY_BUILD is forced OFF and precompiled headers disabled so
# global settings can never mask a missing declaration.

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
    # path. The TU file name embeds the full project-relative path, so it is
    # deterministic and collision-free.
    file(RELATIVE_PATH _rel "${CMAKE_SOURCE_DIR}" "${_hdr}")
    string(REGEX REPLACE "^(src|tools|tests)/" "" _inc "${_rel}")
    string(MAKE_C_IDENTIFIER "${_rel}" _stem)
    set(_tu "${CMAKE_CURRENT_BINARY_DIR}/header_checks/${_stem}.cpp")
    file(WRITE "${_tu}" "#include \"${_inc}\"\n#include \"${_inc}\"\n")
    list(APPEND _fv_check_srcs "${_tu}")
endforeach()

add_library(fluxvoice_header_checks OBJECT ${_fv_check_srcs})
target_include_directories(fluxvoice_header_checks PRIVATE src tools tests)
target_link_libraries(fluxvoice_header_checks PRIVATE
    miniaudio_headers rwqueue_headers nlohmann_json_headers Threads::Threads)
fluxvoice_set_warnings(fluxvoice_header_checks)
set_target_properties(fluxvoice_header_checks PROPERTIES
    UNITY_BUILD OFF DISABLE_PRECOMPILE_HEADERS ON)
