#!/usr/bin/env bash
# Objective loopback check on Linux (PipeWire via pipewire-pulse, or PulseAudio).
#
#   tone ──► [FluxVoice-Test-Src] ──monitor──► fluxvoice ──► [FluxVoice-Test-Out] ──monitor──► recorder
#
# Two temporary null sinks keep the test off your speakers and microphone. The
# recording is analysed for dropouts, clicks and pitch errors
# (tools/analyze_loopback.py), and fluxvoice's own counters are printed.
#
# Usage: tools/pw_loopback_check.sh [--mode passthrough|queue|chunked]
#                                   [--seconds N] [--load] [--keep-wav DIR]
#   --load   saturate every CPU core with busy loops while testing
#
# Requires: pactl, paplay, parecord, python3 + numpy. The null sinks are
# unloaded and the previous default sink/source restored on exit.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${FLUXVOICE_BIN:-$ROOT/build/fluxvoice}"
MODE=queue
SECONDS_RUN=20
LOAD=0
KEEP_DIR=""
FREQ=440

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)     MODE="$2"; shift 2 ;;
        --seconds)  SECONDS_RUN="$2"; shift 2 ;;
        --load)     LOAD=1; shift ;;
        --keep-wav) KEEP_DIR="$2"; shift 2 ;;
        -h|--help)  sed -n '2,17p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
case "$MODE" in
    passthrough|queue) MODE_FLAG="--$MODE"; [[ $MODE == queue ]] && MODE_FLAG=--queue-loopback ;;
    chunked)           MODE_FLAG=--chunked ;;
    *) echo "bad --mode: $MODE" >&2; exit 2 ;;
esac
[[ -x "$BIN" ]] || { echo "fluxvoice binary not found: $BIN (build first)" >&2; exit 2; }
for t in pactl paplay parecord python3; do
    command -v "$t" >/dev/null || { echo "missing tool: $t" >&2; exit 2; }
done

WORK="$(mktemp -d)"
SRC_SINK=fv_test_src
OUT_SINK=fv_test_out
PREV_SINK="$(pactl get-default-sink)"
PREV_SOURCE="$(pactl get-default-source)"
MODULES=()
PIDS=()

cleanup() {
    for p in "${PIDS[@]:-}"; do [[ -n "$p" ]] && kill "$p" 2>/dev/null || true; done
    wait 2>/dev/null || true
    for m in "${MODULES[@]:-}"; do [[ -n "$m" ]] && pactl unload-module "$m" 2>/dev/null || true; done
    pactl set-default-sink "$PREV_SINK" 2>/dev/null || true
    pactl set-default-source "$PREV_SOURCE" 2>/dev/null || true
    rm -rf "$WORK"
}
trap cleanup EXIT INT TERM

MODULES+=("$(pactl load-module module-null-sink sink_name=$SRC_SINK \
    sink_properties=device.description=FluxVoice-Test-Src)")
MODULES+=("$(pactl load-module module-null-sink sink_name=$OUT_SINK \
    sink_properties=device.description=FluxVoice-Test-Out)")
# Loading sinks must not hijack the user's audio.
pactl set-default-sink "$PREV_SINK"
pactl set-default-source "$PREV_SOURCE"
sleep 0.5

# Resolve fluxvoice device indices by description.
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
IN_IDX="$(index_of 'Capture devices:' 'Monitor of FluxVoice-Test-Src')"
OUT_IDX="$(index_of 'Playback devices:' 'FluxVoice-Test-Out')"
if [[ -z "$IN_IDX" || -z "$OUT_IDX" ]]; then
    echo "could not find test devices in --list-devices output:" >&2
    echo "$LIST" >&2
    exit 1
fi

# 48 kHz test tone, longer than the run so fluxvoice never sees its end.
python3 - "$WORK/tone.wav" "$FREQ" "$((SECONDS_RUN + 5))" <<'EOF'
import sys, wave, numpy as np
path, f, secs = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
fs = 48000
t = np.arange(fs * secs) / fs
x = (0.25 * np.sin(2 * np.pi * f * t) * 32767).astype("<i2")
with wave.open(path, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(fs); w.writeframes(x.tobytes())
EOF

parecord -d "$OUT_SINK.monitor" --rate=16000 --channels=1 --format=s16le \
    --file-format=wav "$WORK/out.wav" &
PIDS+=($!)
paplay -d "$SRC_SINK" "$WORK/tone.wav" &
PIDS+=($!)

if [[ $LOAD -eq 1 ]]; then
    for _ in $(seq "$(nproc)"); do ( while :; do :; done ) & PIDS+=($!); done
    echo "CPU load: $(nproc) busy loops"
fi

sleep 1
echo "Running: fluxvoice $MODE_FLAG --input $IN_IDX --output $OUT_IDX --duration $SECONDS_RUN"
"$BIN" "$MODE_FLAG" --input "$IN_IDX" --output "$OUT_IDX" --duration "$SECONDS_RUN" \
    </dev/null 2>"$WORK/stderr.txt" | tee "$WORK/stdout.txt"
FV_STATUS=${PIPESTATUS[0]}
sleep 0.5
for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done
wait 2>/dev/null || true
PIDS=()

if [[ -s "$WORK/stderr.txt" ]]; then
    echo "--- fluxvoice stderr (expected empty) ---"; cat "$WORK/stderr.txt"
fi
[[ -n "$KEEP_DIR" ]] && mkdir -p "$KEEP_DIR" && cp "$WORK/out.wav" "$KEEP_DIR/loopback_${MODE}.wav"

echo "--- analysis of recorded output ---"
ANALYSIS_STATUS=0
python3 "$ROOT/tools/analyze_loopback.py" "$WORK/out.wav" --freq "$FREQ" || ANALYSIS_STATUS=$?

if [[ $FV_STATUS -ne 0 || $ANALYSIS_STATUS -ne 0 || -s "$WORK/stderr.txt" ]]; then
    echo "RESULT: FAIL (fluxvoice exit $FV_STATUS, analysis exit $ANALYSIS_STATUS)"
    exit 1
fi
echo "RESULT: PASS"
