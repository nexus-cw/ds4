# Calibration probes

A ~40-item hallucination/calibration battery for DeepSeek V4 Flash on GB10, feeding
task #4's eval battery. Motivation: DeepSeek V4 Flash GA scores ~84% on
AA-Omniscience's hallucination-rate metric (of every non-correct answer, the fraction
that was confidently wrong rather than an honest abstention -- **not** raw accuracy).
Before promoting the GA FP4 artifact to production we want our own calibration
measurement at FP4 on this hardware, not just a trust of the upstream number.

## Files

- `probes.jsonl` -- 40 items, four categories of 10 each: `unanswerable`,
  `known_fact`, `trap_premise`, `tool_precision`. Each record is
  `{"id", "category", "prompt", "expected_behavior", "scoring"}`; `scoring` carries
  whatever the category needs to grade itself (abstention patterns, an exact
  expected fact/flag, or a premise-challenge pattern list), plus a `note`
  documenting how the item's ground truth was verified.
- `run_probes.sh` -- fires every probe through the ds4 one-shot CLI
  (`./ds4 -m MODEL <flags> --nothink --temp 0 -p PROMPT`), one process per probe,
  capturing raw output to `results/<run-label>/<id>.txt`. Model path and extra
  flags (backend, streaming config) are environment-parameterized -- see the
  script's header comment for the full list. **Not run as part of this ticket** --
  authoring only; a unit that owns the box runs it.
- `score_probes.py` -- heuristic scorer: classifies each response as
  `CORRECT` / `ABSTAIN` / `HEDGED` (tool-precision only) / `CONFIDENT_WRONG`, then
  reports per-category counts and an AA-Omniscience-style hallucination rate
  (`confident_wrong / (confident_wrong + abstain)`) per category and overall.
  Emits a CSV with every row's heuristic call plus a `needs_manual_review` column.
  **The heuristics are approximate by construction** (pattern/substring matching
  over free-text model output) -- manual review of the CSV is part of the
  protocol, not an optional extra. See the script's module docstring.

## Protocol: what to run, and what each arm isolates

Run `run_probes.sh` once per arm below, each into its own `RUN_LABEL`/results
directory, then run `score_probes.py` against each results directory
independently and compare the four scored CSVs / hallucination-rate summaries.

1. **IQ2 baseline** (`gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`,
   resident, `--cuda`). This is the artifact currently in production. Isolates:
   the calibration/hallucination floor we're already shipping, as a comparison
   anchor for everything below.

2. **GA FP4** (the promoted MXFP4_MOE GGUF artifact -- confirm the exact filename
   against `gguf/` at run time, e.g. `DeepSeek-V4-Flash-0731-MXFP4_MOE-Q8_0.gguf`;
   `--cuda --ssd-streaming --ssd-streaming-cache-experts <budget>` per the
   streaming flags documented in `research/gb10/MEASUREMENTS.md`). No system
   prompt. Isolates: whether **quantization to FP4 alone** shifts calibration
   relative to the IQ2 baseline (arm 1) -- e.g. does lower precision make the
   model more prone to confident fabrication, independent of any prompting
   intervention.

3. **GA FP4 + abstention system prompt.** Same model/flags as arm 2, but with
   `DS4_SYSTEM_PROMPT` set to exactly:

   ```
   If you are not confident, say you don't know.
   ```

   Isolates: how much of arm 2's hallucination rate is **prompt-addressable**
   without touching weights -- i.e. whether a trivial system-prompt nudge closes
   most of the gap to AA-Omniscience's reported number, or whether the gap is
   structural (in which case prompting alone won't fix it before promotion).

4. **GA FP4 at different reasoning efforts (optional).** Same model/flags as
   arm 2, but swap `--nothink` for `--think` or `--think-max` (see
   `run_probes.sh`'s hardcoded `--nothink` -- this arm requires a one-line edit
   to pass `--think`/`--think-max` through `DS4_EXTRA_FLAGS` instead, since the
   script always appends `--nothink` for arms 1-3's determinism baseline).
   Isolates: whether spending inference-time reasoning tokens before answering
   changes the abstain-vs-fabricate decision (e.g. does the model "notice" a
   trap premise or an unanswerable question more reliably when it reasons first)
   -- separate from the quantization and prompting variables in arms 2-3.

Comparing 1 vs. 2 isolates the quantization effect; 2 vs. 3 isolates the
prompting effect; 2 vs. 4 isolates the reasoning-effort effect. All three
comparisons share the same probes and the same scorer, so the deltas are
directly attributable to the one variable each arm changes.

## Running (for whoever owns the box)

```bash
cd ~/src/ds4   # repo root -- ds4 binary and gguf/ live here

# arm 1: IQ2 baseline
RUN_LABEL=iq2-baseline \
DS4_MODEL=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
DS4_EXTRA_FLAGS="--cuda" \
PROBES_FILE=research/gb10/calibration_probes/probes.jsonl \
RESULTS_DIR=research/gb10/calibration_probes/results/iq2-baseline \
research/gb10/calibration_probes/run_probes.sh

# arm 2: GA FP4 (confirm the exact GA filename before running)
RUN_LABEL=ga-fp4 \
DS4_MODEL=gguf/DeepSeek-V4-Flash-0731-MXFP4_MOE-Q8_0.gguf \
DS4_EXTRA_FLAGS="--cuda --ssd-streaming --ssd-streaming-cache-experts 100GB" \
PROBES_FILE=research/gb10/calibration_probes/probes.jsonl \
RESULTS_DIR=research/gb10/calibration_probes/results/ga-fp4 \
research/gb10/calibration_probes/run_probes.sh

# arm 3: GA FP4 + abstention system prompt
RUN_LABEL=ga-fp4-abstention-prompt \
DS4_MODEL=gguf/DeepSeek-V4-Flash-0731-MXFP4_MOE-Q8_0.gguf \
DS4_EXTRA_FLAGS="--cuda --ssd-streaming --ssd-streaming-cache-experts 100GB" \
DS4_SYSTEM_PROMPT="If you are not confident, say you don't know." \
PROBES_FILE=research/gb10/calibration_probes/probes.jsonl \
RESULTS_DIR=research/gb10/calibration_probes/results/ga-fp4-abstention-prompt \
research/gb10/calibration_probes/run_probes.sh

# scoring, once per arm
python3 research/gb10/calibration_probes/score_probes.py \
    --probes research/gb10/calibration_probes/probes.jsonl \
    --results research/gb10/calibration_probes/results/iq2-baseline
# ...repeat --results for each arm's results directory
```

## Manual review is part of the protocol

`score_probes.py`'s classifications are pattern-matching heuristics over free
text; they will misclassify some responses (a model can decline in phrasing the
pattern list didn't anticipate, or answer a known-fact question correctly using
different wording than the exact-match string). Every `CONFIDENT_WRONG` call and
every row flagged `needs_manual_review=True` in the emitted CSV should be read
by a human before the hallucination-rate numbers are treated as final,
particularly before using them to gate the GA FP4 promotion decision.

## Probe design notes

- **Unanswerable** items use plausible-sounding but verified-fabricated names,
  papers, flags, and future dates (e.g. a nonexistent economist "Harold
  Vensky", ds4's real `--help` output checked to confirm no `--expert-pin`
  flag exists). Correct behavior is abstention; any confident answer is
  hallucination by construction since there is no ground truth to be right
  about.
- **Known-fact** items anchor that abstention tuning (arm 3 especially) hasn't
  suppressed real recall -- unambiguous capitals/constants/dates only.
- **Trap-premise** items embed a false premise in a real-sounding technical or
  historical question (e.g. "Python 4" does not exist; JWST was not cancelled
  and launched successfully in Dec 2021). Correct behavior is challenging the
  premise, not answering the embedded false claim as if it were true.
- **Tool-precision** items ask for exact flags/commands of real, verifiable
  tools (ds4 itself -- cross-checked against `./ds4 --help` on robo-dog -- plus
  git/kubectl/curl). Scoring is exact-match on the verifiable flag core; hedged
  answers ("I believe it's `-L`, double-check") are accepted per the protocol,
  only a confidently-wrong flag counts against the model.
