#!/usr/bin/env bash
# Latency of fluxvoice's paths on Linux (PipeWire via pipewire-pulse, or
# PulseAudio) — Milestone 1 / Stage 06, Milestone 3 / Stage 11.
#
#   baseline:  fv_probe ──► [FluxVoice-Lat-A] ──monitor──► fv_probe
#   per mode:  fv_probe ──► [FluxVoice-Lat-A] ──monitor──► fluxvoice ──► [FluxVoice-Lat-B] ──monitor──► fv_probe
#
# fv_probe plays a marker and times each one's return. A path's "added" figure
# is its median minus the baseline's: fluxvoice's own buffering, resampling and
# queues plus one null sink.
#
# Markers (--marker): 'chirp' every 500 ms (default; passthrough, queue,
# chunked), 'speech' (Mode A: --modes vc) and 'phrase' every 3500 ms (Mode B:
# --modes s2ut). The phrase marker reports two delays per path, the onset delay
# and the flush tail lag; the baseline is measured with the same marker.
#
# Usage: tools/pw_latency_check.sh [--count N] [--modes "passthrough queue chunked vc s2ut"]
#            [--marker chirp|speech|phrase] [--interval-ms M] [--period-ms P]
#            [--model-dir DIR] [--src-lang L] [--tgt-lang L] [--wait-k "1 2 3 4 5 8"]
#            [--keep-logs DIR]
#
#   --model-dir DIR   bundle for the vc / s2ut modes
#   --src-lang/--tgt-lang   language pair for s2ut (default es / en)
#   --wait-k LIST     one s2ut run per k (default: the pair's default)
#   --period-ms P     fluxvoice device period (default: fluxvoice's own)
#   --keep-logs DIR   keep fv_probe CSVs and fluxvoice logs ([stats] every 5 s)
#
# Mode B example (the fixture bundle's "en" voice is silent: translate to de):
#   tools/pw_latency_check.sh --modes s2ut --marker phrase --count 60 \
#       --model-dir tests/data/fixtures/s2ut --src-lang en --tgt-lang de \
#       --wait-k "1 2 3 4 5 8" --period-ms 10
#
# Requires: pactl, a built fluxvoice and fv_probe. The null sinks are unloaded
# and the previous default sink/source restored on exit.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${FLUXVOICE_BIN:-$ROOT/build/fluxvoice}"
PROBE="${FV_PROBE_BIN:-$ROOT/build/fv_probe}"
COUNT=60
MODES="passthrough queue chunked"
MARKER=chirp
INTERVAL_MS=""
PERIOD_MS=""
MODEL_DIR=""
SRC_LANG=es
TGT_LANG=en
WAIT_KS=""
KEEP_LOGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --count) COUNT="$2"; shift 2 ;;
        --modes) MODES="$2"; shift 2 ;;
        --marker) MARKER="$2"; shift 2 ;;
        --interval-ms) INTERVAL_MS="$2"; shift 2 ;;
        --period-ms) PERIOD_MS="$2"; shift 2 ;;
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        --src-lang) SRC_LANG="$2"; shift 2 ;;
        --tgt-lang) TGT_LANG="$2"; shift 2 ;;
        --wait-k) WAIT_KS="$2"; shift 2 ;;
        --keep-logs) KEEP_LOGS="$2"; shift 2 ;;
        -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
case "$MARKER" in
    chirp|speech) : "${INTERVAL_MS:=500}" ;;
    phrase)       : "${INTERVAL_MS:=3500}" ;;
    *) echo "bad marker: $MARKER (chirp, speech or phrase)" >&2; exit 2 ;;
esac
for mode in $MODES; do
    case "$mode" in
        passthrough|queue|chunked) ;;
        vc|s2ut) [[ -n "$MODEL_DIR" ]] || { echo "mode $mode needs --model-dir" >&2; exit 2; } ;;
        *) echo "bad mode: $mode" >&2; exit 2 ;;
    esac
done
for b in "$BIN" "$PROBE"; do
    [[ -x "$b" ]] || { echo "binary not found: $b (build first)" >&2; exit 2; }
done
command -v pactl >/dev/null || { echo "missing tool: pactl" >&2; exit 2; }
[[ -z "$KEEP_LOGS" ]] || mkdir -p "$KEEP_LOGS"

WORK="$(mktemp -d)"
SINK_A=fv_lat_a
SINK_B=fv_lat_b
PREV_SINK="$(pactl get-default-sink)"
PREV_SOURCE="$(pactl get-default-source)"
MODULES=()
FV_PID=""

cleanup() {
    [[ -n "$FV_PID" ]] && kill "$FV_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    for m in "${MODULES[@]:-}"; do [[ -n "$m" ]] && pactl unload-module "$m" 2>/dev/null || true; done
    pactl set-default-sink "$PREV_SINK" 2>/dev/null || true
    pactl set-default-source "$PREV_SOURCE" 2>/dev/null || true
    rm -rf "$WORK"
}
trap cleanup EXIT INT TERM

MODULES+=("$(pactl load-module module-null-sink sink_name=$SINK_A \
    sink_properties=device.description=FluxVoice-Lat-A)")
MODULES+=("$(pactl load-module module-null-sink sink_name=$SINK_B \
    sink_properties=device.description=FluxVoice-Lat-B)")
# Loading sinks must not hijack the user's audio.
pactl set-default-sink "$PREV_SINK"
pactl set-default-source "$PREV_SOURCE"
sleep 0.5

# fluxvoice and fv_probe enumerate through the same backend: same indices.
LIST="$("$BIN" --list-devices)"
index_of() {  # $1 = section header, $2 = exact device name
    awk -v sec="$1" -v name="$2" '
        /^[A-Za-z].*:$/ { insec = ($0 == sec) ; next }
        insec && match($0, /^  \[([0-9]+)\] /) {
            idx = substr($0, 4, RLENGTH - 5); dev = substr($0, RLENGTH + 1)
            sub(/  \(default\)$/, "", dev)
            if (dev == name) { print idx; exit }
        }' <<<"$LIST"
}
A_OUT="$(index_of 'Playback devices:' 'FluxVoice-Lat-A')"
A_MON="$(index_of 'Capture devices:' 'Monitor of FluxVoice-Lat-A')"
B_OUT="$(index_of 'Playback devices:' 'FluxVoice-Lat-B')"
B_MON="$(index_of 'Capture devices:' 'Monitor of FluxVoice-Lat-B')"
if [[ -z "$A_OUT" || -z "$A_MON" || -z "$B_OUT" || -z "$B_MON" ]]; then
    echo "could not find test devices in --list-devices output:" >&2
    echo "$LIST" >&2
    exit 1
fi

# $1 = label, then fv_probe latency arguments. Prints fv_probe's report and
# records one line in $WORK/results: "label median p95" for the chirp and
# speech markers, "label onset_median onset_p95 onset_max tail_median tail_p95"
# for the phrase marker.
measure() {
    local label="$1"; shift
    echo "=== $label"
    local status=0
    local csv=()
    [[ -z "$KEEP_LOGS" ]] || csv=(--csv "$KEEP_LOGS/probe_$label.csv")
    "$PROBE" latency --marker "$MARKER" --interval-ms "$INTERVAL_MS" --count "$COUNT" \
        "${csv[@]}" "$@" | tee "$WORK/probe.txt" || status=$?
    if [[ "$MARKER" == phrase ]]; then
        # "onset_delay_ms : median 12.34 ms, p95 13.45 ms, min 1.00 ms, max 2.00 ms"
        local onset tail
        onset="$(grep '^onset_delay_ms' "$WORK/probe.txt" || true)"
        tail="$(grep '^flush_tail_ms' "$WORK/probe.txt" || true)"
        if [[ $status -ne 0 || -z "$onset" || -z "$tail" ]]; then
            echo "$label FAIL FAIL FAIL FAIL FAIL" >>"$WORK/results"
            return 1
        fi
        echo "$label $(awk '{ print $4, $7, $13 }' <<<"$onset") $(awk '{ print $4, $7 }' <<<"$tail")" \
            >>"$WORK/results"
        return 0
    fi
    local line
    line="$(grep '^Latency :' "$WORK/probe.txt" || true)"
    if [[ $status -ne 0 || -z "$line" ]]; then
        echo "$label FAIL FAIL" >>"$WORK/results"
        return 1
    fi
    # "Latency : median 12.34 ms, p95 13.45 ms, min ..."
    awk -v l="$label" '{ print l, $4, $7 }' <<<"$line" >>"$WORK/results"
}

FAILED=0
measure baseline --output "$A_OUT" --input "$A_MON" || FAILED=1

# fv_probe: 1 s settle + COUNT markers; fluxvoice starts ~2 s earlier.
RUN_S=$(( (COUNT * INTERVAL_MS + 999) / 1000 + 8 ))
PERIOD_ARGS=()
[[ -z "$PERIOD_MS" ]] || PERIOD_ARGS=(--period-ms "$PERIOD_MS")

# $1 = label, then the fluxvoice mode arguments.
run_mode() {
    local label="$1"; shift
    "$BIN" "$@" "${PERIOD_ARGS[@]}" --input "$A_MON" --output "$B_OUT" --duration "$RUN_S" \
        --stats 5 </dev/null >"$WORK/fv_$label.txt" 2>&1 &
    FV_PID=$!
    sleep 2
    if ! kill -0 "$FV_PID" 2>/dev/null; then
        echo "=== $label"; echo "fluxvoice exited early:"; cat "$WORK/fv_$label.txt"
        echo "$label FAIL FAIL FAIL FAIL FAIL" >>"$WORK/results"
        FV_PID=""; FAILED=1
        return 0
    fi
    measure "$label" --output "$A_OUT" --input "$B_MON" || FAILED=1
    wait "$FV_PID" || FAILED=1
    FV_PID=""
    grep -E '^(Stats|Queued audio):' "$WORK/fv_$label.txt" || true
    # The last [stats] line carries the in-process estimates (lat_ms, lag_ms).
    grep -E '^\[stats\]' "$WORK/fv_$label.txt" | tail -n 1 || true
    [[ -z "$KEEP_LOGS" ]] || cp "$WORK/fv_$label.txt" "$KEEP_LOGS/fluxvoice_$label.log"
}

for mode in $MODES; do
    case "$mode" in
        passthrough) run_mode passthrough --passthrough ;;
        queue)       run_mode queue --queue-loopback ;;
        chunked)     run_mode chunked --chunked ;;
        vc)          run_mode vc --vc --model-dir "$MODEL_DIR" ;;
        s2ut)
            if [[ -z "$WAIT_KS" ]]; then
                run_mode s2ut --s2ut --model-dir "$MODEL_DIR" --src-lang "$SRC_LANG" --tgt-lang "$TGT_LANG"
            else
                for k in $WAIT_KS; do
                    run_mode "s2ut_k$k" --s2ut --model-dir "$MODEL_DIR" --src-lang "$SRC_LANG" \
                        --tgt-lang "$TGT_LANG" --wait-k "$k"
                done
            fi ;;
    esac
done

echo "=== summary (ms)"
if [[ "$MARKER" == phrase ]]; then
    awk 'NR == 1 { base_on = $2; base_tail = $5 }
         { printf "%-12s onset median %8s  p95 %8s  max %8s | tail median %8s  p95 %8s", $1, $2, $3, $4, $5, $6
           if (NR > 1 && $2 != "FAIL" && base_on != "FAIL")
               printf " | added onset %7.2f  p95 %7.2f  max %7.2f  tail %7.2f", $2 - base_on, $3 - base_on, $4 - base_on, $5 - base_tail
           printf "\n" }' "$WORK/results"
else
    awk 'NR == 1 { base = $2 }
         { printf "%-12s median %8s  p95 %8s", $1, $2, $3
           if (NR > 1 && $2 != "FAIL" && base != "FAIL") printf "  added %7.2f", $2 - base
           printf "\n" }' "$WORK/results"
fi

if [[ $FAILED -ne 0 ]]; then echo "RESULT: FAIL"; exit 1; fi
echo "RESULT: PASS"
