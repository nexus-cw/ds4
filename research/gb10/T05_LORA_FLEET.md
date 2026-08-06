# T0.5 — LoRA adapter fleet at ~1B on robo-dog (BTX-lineage cheap proof)

Date: 2026-08-06/07 · Node: robo-dog (GB10, sm_121) · Task #13 in the ds4/accretion program.

**Question**: does specialize-then-route (several domain LoRA adapters over one small
frozen base + a router picking the adapter per prompt) beat one generalist at matched
size? This is the accretion-side cousin of toy-genesis (dMon), which proves expert-level
*genesis* mechanics; T0.5 proves (or here: fails to prove) the *adapter-fleet* variant.
Keep them distinct: toy-genesis grows new experts against a frozen backbone; T0.5 bolts
additive LoRA deltas onto a frozen instruct base.

**Verdict up front: NEGATIVE at this scale/recipe.** 300-sample rank-16 LoRA SFT on a
1.5B *instruct* base cloned each domain's surface style but did not add skill — the
oracle-adapter arm failed to beat the bare base on any domain, and on math it was
*worse* (style imitation clipped the base's chain-of-thought). Routing itself worked
(83% accuracy, honest nearest-centroid). Details and caveats below.

## Protocol

- Base: **Qwen/Qwen2.5-1.5B-Instruct** (Apache-2.0), bf16, fetched from HF.
- Adapters: 4 domains — **python** (iamtarun/python_code_instructions_18k_alpaca),
  **math** (openai/gsm8k main), **sql** (b-mc2/sql-create-context), **go** (reused the
  prior CWB-Go SFT dataset from /data/model-tune/dataset — 531 rows, June 2026).
  300 train / 30 val each; rank 16, alpha 32, q/k/v/o/gate/up/down targets, lr 1e-4,
  2 epochs, seq 1024, bs 1 × ga 8, bf16 — via the existing
  `/data/model-tune/train_lora.py` with `--base` override (reused unchanged).
- Router: base model's mean last-hidden-state over the prompt (bf16→fp32, L2-normed),
  nearest-centroid over 40 training prompts per domain. No trained classifier.
  Routing embeddings computed **before** PEFT attach (PEFT injects LoRA layers in
  place; embedding after attach would be contaminated — caught in review).
- Eval: 12 held-out probes/domain × 4 arms — (a) base, (b) oracle adapter, (c) routed
  adapter, (d) deliberately-wrong adapter (derangement python→math→sql→go→python).
  Greedy decode, 512 new tokens. Scoring: math = last-number exact vs gsm8k gold;
  sql = normalized gold-contained-in-response; python = extracted code executed against
  handwritten asserts (subprocess, 15s timeout); go = ≥2 of the reference's salient
  identifiers present (weakest scorer — no Go toolchain on robo-dog, recorded as such).

## Inventory findings (truth-in-data pass, /data/model-tune)

- Prior work (June 2026): ONE adapter — CWB-Go style over **Qwen3-Coder-30B-A3B**
  (rank 16, 3 epochs, 3.6h, eval_loss 0.642, token-acc 0.845), merged + GGUF'd
  (DONE / MERGE-DONE / GGUF-BUILD-DONE markers). Not reusable as a 1B-fleet member
  (wrong base scale), but its **train_lora.py, dataset (go domain), and env recipe were
  reused**. Also present: GLM-4.7-Flash-Q8_0.gguf (32GB, unrelated).
- Relocated `.venv` imported fine but held **torch 2.11.0+cpu (no CUDA)** — rebuilt
  fresh at `/data/model-tune/t05/.venv`: torch 2.13.0+cu130, transformers 5.14.1,
  peft 0.20.0, trl 1.9.2, datasets 5.0.1, accelerate 1.14.0 (log:
  t05/logs/venv_build.log). One system dep was missing and installed:
  `python3.12-dev` (triton JIT needs Python.h).

## Training (all four, sequential, nice 10, alongside production)

| adapter | steps | runtime | train_loss | eval_loss | eval token-acc |
|---|---|---|---|---|---|
| python | 76 | 118s | 0.685 | 0.663 | 0.825 |
| math   | 76 | 114s | 0.374 | 0.291 | 0.922 |
| sql    | 76 | 92s  | 0.839 | 0.654 | 0.845 |
| go     | 76 | 265s | 1.245 | 1.005 | 0.781 |

Total GPU time ≈ 10 min. Artifacts under `/data/model-tune/t05/` (adapters/, data/,
probes/, logs/, eval_results.json, eval_transcripts.json — full generations recorded).

## GPU contention (production ds4-server stayed up throughout)

Baseline chat smoke (8-tok completion): 1.9–2.9s. Probe every 5 min during train+eval:
n=7 → min 1.94s, median 2.16s, p95 5.31s, max **10.6s** (single spike during training;
two ~5.3s samples during eval; all HTTP 200). Never a *sustained* breach of the 2.5×
pause threshold, so training was never SIGSTOPped. Post-run verify: /v1/models 200,
chat smoke 2.0s. **Co-existence is practical for 1.5B LoRA work**, with visible but
transient latency spikes on the shared GPU.

## Four-arm results (12 probes/domain)

| domain | (a) base | (b) oracle | (c) routed | (d) wrong |
|---|---|---|---|---|
| python | 11/12 | 11/12 | 11/12 | 9/12 |
| math   | **8/12** | **5/12** | 4/12 | 7/12 |
| sql    | 4/12 | 5/12 | 5/12 | 4/12 |
| go     | 2/12 | 2/12 | 2/12 | 1/12 |

Routing confusion (rows = true domain, cols = picked): python 12→python;
**math 8→python, 4→math**; sql 12→sql; go 12→go. **Routing accuracy 83.3%** (40/48);
the only failure mode is math word problems embedding near the python-instruction
centroid.

## ACCEPTANCE verdict

- **(b) beats (a) on every domain — FAIL.** python tie (ceiling: base already 11/12),
  math (b) *worse* than (a) by 3, sql +1 (within noise at n=12), go tie (floor: probes
  are CWB-context code-review tasks the 1.5B can't do either way, and the
  identifier-contains scorer is weak).
- **(c) close to (b) — PASS.** Identical on python/sql/go; −1 on math, consistent with
  the 8/12 misroutes. Routing accuracy 83%.
- **(d) at-or-below (a) — MARGINAL PASS.** Below or equal on python/sql/go; on math (d)
  7/12 vs (a) 8/12 — but note (d)=python-adapter-on-math outscored (b)=math-adapter-on-
  math, which is the specialization failure restated, not a control failure.

**Why (b) failed on math (transcript-verified, not a scoring artifact):** the math
adapter faithfully adopted gsm8k's terse `<<...>>`/`#### N` style (hence the excellent
0.29 eval_loss — it's imitating the *format* well) but reasons worse than the base's
longer chain-of-thought, e.g. "3 dozen donuts at $68 per dozen" computed as 12×68.
300 samples of style SFT at 1.5B transfers form, not competence, and *replacing* an
instruct model's verbose CoT with terse dataset style actively costs accuracy.
Secondary factors: base is already strong on easy python (ceiling) and the go probe
scorer is too crude to resolve differences (no compiler on the box).

**Honest T0.5 conclusion:** at this scale and recipe, specialize-then-route did NOT
beat the generalist. What *is* proven: the fleet mechanics work end-to-end on robo-dog
(4 adapters trained beside live production in ~10 min total; hot-swap via
peft set_adapter; honest embedding router at 83%), and the cheap-SFT failure mode is
now characterized. A rerun that could flip the verdict needs: domains the base is
genuinely weak at (no ceiling), 10–100× more data or completion-only loss that
preserves CoT, and executable scoring for code domains.

## Relation to the genesis line (toy-genesis, dMon)

T0.5's negative does **not** indict expert-level genesis, but it sharpens what genesis
must avoid. Adapters here are additive low-rank deltas on a frozen base — the same
*shape* as newborn experts against a frozen backbone — and the fleet mechanics
(train-beside-production, per-domain artifacts, embedding-based routing, hot attach)
all carried, which is the transferable positive. The failure was not the shape but the
*training signal*: 300 samples of style-imitation SFT onto an already-competent
generalist buys form without skill, and can overwrite better behavior the base already
had. Genesis differs on exactly these axes — newborn experts are trained where the
backbone is measurably *deficient* (no ceiling to collide with), on traffic-derived
corpora rather than 300 canned rows, with the backbone frozen so nothing existing is
overwritten. T0.5 therefore says: the routing-and-hosting substrate for a specialist
population is cheap and works today on robo-dog; the open question genesis must answer
is the one T0.5 flunked — producing a delta that *adds capability* rather than
restyling it, which is precisely what toy-genesis's verdict tests are for.

## Artifacts

- `/data/model-tune/t05/` on robo-dog: adapters/{python,math,sql,go},
  data/, probes/, prep_data.py, eval_t05.py, train_all.sh, latency_probe.sh,
  eval_results.json, eval_transcripts.json (all 192 generations verbatim),
  logs/ (venv build, per-adapter training, eval, prod_latency.log).
- Base model: Qwen/Qwen2.5-1.5B-Instruct, **Apache-2.0**.
- Production ds4-server: untouched, verified healthy before, during (probes), after.
