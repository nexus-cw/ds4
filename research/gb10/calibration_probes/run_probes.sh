#!/usr/bin/env bash
# run_probes.sh -- fire every probe in probes.jsonl through the ds4 one-shot CLI
# and capture raw output for later scoring by score_probes.py.
#
# Pattern follows the ds4 one-shot CLI invocations documented throughout
# research/gb10/MEASUREMENTS.md (e.g. `./ds4 -m MODEL --cuda --nothink -p PROMPT`),
# temp 0 for determinism (a "before we run this on the box" precedent: see
# MEASUREMENTS.md's discussion of `--temp 0` not always being explicit --
# this script always passes it explicitly).
#
# This script is AUTHORED ONLY. It is not run as part of this ticket -- no
# model load, no GPU, no server touch happens here. A separate unit that owns
# the box runs it.
#
# Usage:
#   DS4_MODEL=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
#   DS4_EXTRA_FLAGS="--cuda" \
#   ./run_probes.sh
#
#   DS4_MODEL=gguf/DeepSeek-V4-Flash-0731-MXFP4_MOE-Q8_0.gguf \
#   DS4_EXTRA_FLAGS="--cuda --ssd-streaming --ssd-streaming-cache-experts 100GB" \
#   DS4_SYSTEM_PROMPT="If you are not confident, say you don't know." \
#   ./run_probes.sh
#
# Env vars (all optional, sane defaults for the IQ2 baseline):
#   DS4_BIN            path to the ds4 binary (default: ./ds4, run from repo root)
#   DS4_MODEL          -m argument (default: ds4flash.gguf, i.e. the repo's default symlink)
#   DS4_EXTRA_FLAGS    extra flags passed through verbatim (backend selection, streaming
#                       flags, --ctx, etc.) -- space-separated, NOT quoted as one arg
#   DS4_SYSTEM_PROMPT  optional system/preamble text prepended to every probe prompt,
#                       used for the "abstention system prompt" arm of the protocol
#                       (see README.md). Empty by default (no system prompt arm).
#   DS4_MAX_TOKENS     -n argument (default: 300; probes expect short, direct answers)
#   PROBES_FILE        input JSONL (default: probes.jsonl next to this script)
#   RESULTS_DIR        output directory for one .txt file per probe (default:
#                       results/<run-label>/ next to this script)
#   RUN_LABEL          subdirectory name under RESULTS_DIR's parent, identifying this
#                       arm of the protocol, e.g. "iq2-baseline", "ga-fp4",
#                       "ga-fp4-abstention-prompt" (default: derived from date+pid)
#
# Requires: jq (for JSONL field extraction). Falls back to a python3 one-liner
# if jq is not on PATH.

set -euo pipefail

DS4_BIN="${DS4_BIN:-./ds4}"
DS4_MODEL="${DS4_MODEL:-ds4flash.gguf}"
DS4_EXTRA_FLAGS="${DS4_EXTRA_FLAGS:---cuda}"
DS4_SYSTEM_PROMPT="${DS4_SYSTEM_PROMPT:-}"
DS4_MAX_TOKENS="${DS4_MAX_TOKENS:-300}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBES_FILE="${PROBES_FILE:-${SCRIPT_DIR}/probes.jsonl}"
RUN_LABEL="${RUN_LABEL:-run-$(date +%Y%m%d-%H%M%S)}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/results/${RUN_LABEL}}"

if [[ ! -f "${PROBES_FILE}" ]]; then
    echo "probes file not found: ${PROBES_FILE}" >&2
    exit 1
fi

if [[ ! -x "${DS4_BIN}" ]]; then
    echo "ds4 binary not found or not executable: ${DS4_BIN}" >&2
    echo "run this script from the ds4 repo root, or set DS4_BIN" >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

echo "== calibration probe run =="
echo "  ds4 binary   : ${DS4_BIN}"
echo "  model        : ${DS4_MODEL}"
echo "  extra flags  : ${DS4_EXTRA_FLAGS}"
echo "  system prompt: ${DS4_SYSTEM_PROMPT:-<none>}"
echo "  max tokens   : ${DS4_MAX_TOKENS}"
echo "  probes file  : ${PROBES_FILE}"
echo "  results dir  : ${RESULTS_DIR}"
echo

# Manifest records the exact config this run used, for the scorer and for
# README-cited comparisons across the four protocol arms.
manifest="${RESULTS_DIR}/manifest.json"
python3 - "$manifest" "$DS4_BIN" "$DS4_MODEL" "$DS4_EXTRA_FLAGS" "$DS4_SYSTEM_PROMPT" "$DS4_MAX_TOKENS" "$RUN_LABEL" <<'PYEOF'
import json, sys, datetime
path, ds4_bin, model, extra_flags, system_prompt, max_tokens, run_label = sys.argv[1:8]
with open(path, "w") as f:
    json.dump({
        "run_label": run_label,
        "ds4_bin": ds4_bin,
        "model": model,
        "extra_flags": extra_flags,
        "system_prompt": system_prompt,
        "max_tokens": int(max_tokens),
        "started_at": datetime.datetime.now().isoformat(),
    }, f, indent=2)
    f.write("\n")
PYEOF

n=0
while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    id=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['id'])" "${line}")
    category=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['category'])" "${line}")
    prompt=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['prompt'])" "${line}")

    full_prompt="${prompt}"
    if [[ -n "${DS4_SYSTEM_PROMPT}" ]]; then
        full_prompt="${DS4_SYSTEM_PROMPT}

${prompt}"
    fi

    out_file="${RESULTS_DIR}/${id}.txt"
    n=$((n + 1))
    echo "[$n] ${id} (${category}) -> ${out_file}"

    # shellcheck disable=SC2086
    "${DS4_BIN}" -m "${DS4_MODEL}" ${DS4_EXTRA_FLAGS} --nothink --temp 0 \
        -n "${DS4_MAX_TOKENS}" \
        -p "${full_prompt}" \
        > "${out_file}" 2>"${out_file}.stderr"
done < "${PROBES_FILE}"

echo
echo "done: ${n} probes run, output in ${RESULTS_DIR}"
