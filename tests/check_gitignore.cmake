# BUG-008 regression check: the root .gitignore must keep the committed test
# data tracked while ignoring Milestone 2 data, weights and run output. An
# unanchored "data/" pattern would silently drop tests/data/ from every commit.
#
# Run as: cmake -DGIT=<git> -DSRC=<source dir> -P tests/check_gitignore.cmake
# --no-index checks the patterns only, so the paths need not exist.

set(_tracked
    tests/data/contracts/valid/pair_a0_s1.json
    tests/data/fixtures/manifest.json
    tests/data/fixtures/identity/encoder_s1.onnx
    tests/data/pitch/README.md
    python_training/eval_lists/test_clean_200.tsv
    models/README.md)
set(_ignored
    data/cache/models/x.pt
    data/manifests/a.jsonl
    checkpoints/x.ckpt
    runs/exp1/events.out
    python_training/checkpoints/c.pt
    python_training/runs/e/x
    model.pt
    weights/best.ckpt
    big.onnx.data
    models/some_bundle/encoder_s1.onnx)

set(_failures "")
foreach(_mode tracked ignored)
    if(_mode STREQUAL "tracked")
        set(_want 1)
    else()
        set(_want 0)
    endif()
    foreach(_p IN LISTS _${_mode})
        execute_process(COMMAND ${GIT} check-ignore --no-index -q ${_p}
                        WORKING_DIRECTORY ${SRC} RESULT_VARIABLE _rc)
        if(NOT _rc EQUAL _want)
            string(APPEND _failures "\n  ${_p}: expected ${_mode}, git check-ignore exit ${_rc}")
        endif()
    endforeach()
endforeach()

if(_failures)
    message(FATAL_ERROR "gitignore rules are wrong:${_failures}")
endif()
message(STATUS "PASS: test data tracked, data/weights/runs ignored")
