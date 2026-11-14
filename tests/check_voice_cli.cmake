# Stage 09 CLI: --list-params / --list-presets are standalone, and --preset /
# --param rejections must fire for the intended reason, before ORT/audio.
function(expect_error needle)
    execute_process(COMMAND "${EXE}" ${ARGN}
        RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(rc EQUAL 0)
        message(FATAL_ERROR "Unexpected success: ${ARGN}\n${out}${err}")
    endif()
    string(FIND "${out}${err}" "${needle}" match)
    if(match EQUAL -1)
        message(FATAL_ERROR "${ARGN}\nExpected '${needle}', got:\n${out}${err}")
    endif()
endfunction()

function(expect_ok needle)
    execute_process(COMMAND "${EXE}" ${ARGN}
        RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err)
    if(NOT rc EQUAL 0)
        message(FATAL_ERROR "Expected exit 0: ${ARGN} (rc=${rc})\n${out}${err}")
    endif()
    string(FIND "${out}${err}" "${needle}" match)
    if(match EQUAL -1)
        message(FATAL_ERROR "${ARGN}\nExpected '${needle}', got:\n${out}${err}")
    endif()
endfunction()

# Standalone information commands: no --vc, no ORT, no audio.
expect_ok("pitch_shift" --list-params)
expect_ok("\"schema\": 1" --list-params --json)
foreach(preset Neutral Deep Bright Robot Whisper)
    expect_ok("${preset}" --list-presets)
endforeach()

# Rejections.
expect_error("requires --list-params" --json)
expect_error("require --vc" --preset deep)
expect_error("require --vc" --param pitch_shift=3)
expect_error("require --vc" --voice-b identity)
# These are validated before the ORT-availability check, so they must reject
# even in ORT-less builds.
expect_error("unknown parameter" --vc --param bogus=1)
expect_error("expected name=value" --vc --param pitch_shift)
expect_error("unknown preset" --vc --preset nosuch)
if(HAS_ORT)
    expect_error("unknown voice" --vc --model-dir "${FIXTURES}" --voice-b missing)
endif()
