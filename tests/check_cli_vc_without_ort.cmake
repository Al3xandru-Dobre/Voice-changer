# Milestone 2 / Stage 07: in an ORT-less build, `fluxvoice --vc ...` must exit
# non-zero AND print the exact diagnostic telling the user how to enable ORT.
#
# Run as: cmake -DEXE=$<TARGET_FILE:fluxvoice> -P tests/check_cli_vc_without_ort.cmake

set(_want "built without ONNX Runtime (configure with -DONNXRUNTIME_ROOT=... or -DFLUXVOICE_FETCH_ORT=ON)")

foreach(_extra "" "--model-dir;models/bundle;--voice;me")
    execute_process(
        COMMAND "${EXE}" --vc ${_extra}
        RESULT_VARIABLE _rc
        OUTPUT_VARIABLE _out
        ERROR_VARIABLE  _err
    )
    set(_combined "${_out}${_err}")
    if(_rc EQUAL 0)
        message(FATAL_ERROR "fluxvoice --vc ${_extra} exited 0 in an ORT-less build; output:\n${_combined}")
    endif()
    string(FIND "${_combined}" "${_want}" _pos)
    if(_pos EQUAL -1)
        message(FATAL_ERROR
            "fluxvoice --vc ${_extra} did not print expected message:\n"
            "  want: ${_want}\n"
            "  got:  ${_combined}")
    endif()
endforeach()

message(STATUS "PASS: fluxvoice --vc rejected with ORT-less diagnostic")
