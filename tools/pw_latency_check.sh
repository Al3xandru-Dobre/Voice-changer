#!/usr/bin/env bash
# Latency of fluxvoice's paths on Linux (PipeWire via pipewire-pulse, or
# PulseAudio) — Milestone 1 / Stage 06.
#
#   baseline:  fv_probe ──► [FluxVoice-Lat-A] ──monitor──► fv_probe
#   per mode:  fv_probe ──► [FluxVoice-Lat-A] ──monitor──► fluxvoice ──► [FluxVoice-Lat-B] ──monitor──► fv_probe
#
# fv_probe plays a chirp every 500 ms and times each one's return. A path's
# "added" figure is its median minus the baseline's: fluxvoice's own buffering,
# resampling and queues plus one null sink.
#
# Usage: tools/pw_latency_check.sh [--count N] [--modes "passthrough queue chunked"]
#
# Requires: pactl, a built fluxvoice and fv_probe. The null sinks are unloaded
# and the previous default sink/source restored on exit.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${FLUXVOICE_BIN:-$ROOT/build/fluxvoice}"
PROBE="${FV_PROBE_BIN:-$ROOT/build/fv_probe}"
COUNT=60
MODES="passthrough queue chunked"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --count) COUNT="$2"; shift 2 ;;
        --modes) MODES="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
for b in "$BIN" "$PROBE"; do
    [[ -x "$b" ]] || { echo "binary not found: $b (build first)" >&2; exit 2; }
done
command -v pactl >/dev/null || { echo "missing tool: pactl" >&2; exit 2; }

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
# records "label median p95" in $WORK/results.
measure() {
    local label="$1"; shift
    echo "=== $label"
    local status=0
    "$PROBE" latency --count "$COUNT" "$@" | tee "$WORK/probe.txt" || status=$?
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

RUN_S=$(( COUNT / 2 + 5 ))  # fv_probe: 1 s settle + COUNT x 500 ms
for mode in $MODES; do
    case "$mode" in
        passthrough) flag=--passthrough ;;
        queue)       flag=--queue-loopback ;;
        chunked)     flag=--chunked ;;
        *) echo "bad mode: $mode" >&2; exit 2 ;;
    esac
    "$BIN" "$flag" --input "$A_MON" --output "$B_OUT" --duration "$RUN_S" \
        </dev/null >"$WORK/fv_$mode.txt" 2>&1 &
    FV_PID=$!
    sleep 1
    measure "$mode" --output "$A_OUT" --input "$B_MON" || FAILED=1
    wait "$FV_PID" || FAILED=1
    FV_PID=""
    grep -E '^(Stats|Queued audio):' "$WORK/fv_$mode.txt" || true
done

echo "=== summary (ms)"
awk 'NR == 1 { base = $2 }
     { printf "%-12s median %8s  p95 %8s", $1, $2, $3
       if (NR > 1 && $2 != "FAIL" && base != "FAIL") printf "  added %7.2f", $2 - base
       printf "\n" }' "$WORK/results"

if [[ $FAILED -ne 0 ]]; then echo "RESULT: FAIL"; exit 1; fi
echo "RESULT: PASS"
