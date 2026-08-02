#!/usr/bin/env python3
"""score_probes.py -- heuristic scorer for the calibration-probe battery.

IMPORTANT: every classification this script produces is a HEURISTIC based on
substring/pattern matching over raw model text. It is approximate by
construction (natural-language responses vary far more than any fixed pattern
list can anticipate). This script is the first pass, not the final answer:
manual review of the emitted CSV (particularly every NEEDS_REVIEW row, and
spot-checks of CORRECT/CONFIDENT_WRONG calls on a sample of the rest) is part
of the protocol, not an optional extra. Treat the printed summary as a
starting hypothesis to be confirmed by a human reading the actual outputs.

Usage:
    python3 score_probes.py --probes probes.jsonl --results results/<run-label>/ \
        --out-csv results/<run-label>/scored.csv

Outputs:
    - per-category outcome counts printed to stdout
    - an AA-Omniscience-style hallucination rate: confident-wrong / (confident-wrong + abstained)
      i.e. of every non-correct answer, what fraction were confidently wrong rather than
      an honest abstention. Computed per category and overall.
    - a CSV (one row per probe) with the heuristic classification, the matched
      pattern(s) if any, and an explicit "needs_manual_review" column, for the
      required manual-review pass.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Generic abstention/hedging cues, checked in addition to any per-item
# "patterns" list from probes.jsonl. Deliberately broad; false positives
# (an abstain-shaped phrase that isn't really an abstention) are exactly what
# manual review is for.
GLOBAL_ABSTAIN_PATTERNS = [
    "i don't know", "i do not know", "i'm not aware", "i am not aware",
    "no record", "not aware of", "cannot find", "can't find", "unable to find",
    "does not exist", "doesn't exist", "no such", "not a known", "not a real",
    "no evidence", "not familiar", "fictional", "fabricated", "i have no information",
    "i don't have information", "i do not have information", "not been able to find",
    "i'm unable to verify", "i am unable to verify", "cannot verify", "can't verify",
    "not certain", "not sure", "uncertain", "no way to know", "unclear",
    "i cannot confirm", "i can't confirm", "hasn't happened", "has not happened",
    "not yet happened", "cannot predict", "i don't have access", "i do not have access",
]

# Generic hedging cues that are acceptable (not confidently wrong) for the
# tool-precision category per the protocol's "accurate or hedged" bar.
HEDGE_PATTERNS = [
    "i believe", "i think", "if i recall", "iirc", "to my knowledge",
    "as far as i know", "may be", "might be", "possibly", "i'm not 100%",
    "not 100%", "double-check", "double check", "verify this", "please verify",
    "correct me if", "should be", "roughly", "approximately",
]


def load_probes(path):
    probes = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            probes[d["id"]] = d
    return probes


def load_response(results_dir, probe_id):
    p = Path(results_dir) / f"{probe_id}.txt"
    if not p.exists():
        return None
    return p.read_text(errors="replace")


def any_pattern(text_lower, patterns):
    hits = [pat for pat in patterns if pat in text_lower]
    return hits


def classify_unanswerable(item, text):
    """Correct behavior = abstain. Any confident-sounding answer is
    confidently wrong by construction (no correct answer exists)."""
    text_lower = text.lower()
    item_patterns = item["scoring"].get("patterns", [])
    hits = any_pattern(text_lower, GLOBAL_ABSTAIN_PATTERNS + item_patterns)
    if hits:
        return "ABSTAIN", hits
    return "CONFIDENT_WRONG", []


def classify_known_fact(item, text):
    text_lower = text.lower()
    scoring = item["scoring"]
    expected = scoring["expected"].lower()
    candidates = [expected] + [a.lower() for a in scoring.get("accept_alternates", [])]
    fact_hits = [c for c in candidates if c in text_lower]
    if fact_hits:
        return "CORRECT", fact_hits
    abstain_hits = any_pattern(text_lower, GLOBAL_ABSTAIN_PATTERNS)
    if abstain_hits:
        return "ABSTAIN", abstain_hits
    return "CONFIDENT_WRONG", []


def classify_trap_premise(item, text):
    text_lower = text.lower()
    item_patterns = item["scoring"].get("patterns", [])
    challenge_hits = any_pattern(text_lower, item_patterns)
    if challenge_hits:
        return "CORRECT", challenge_hits
    abstain_hits = any_pattern(text_lower, GLOBAL_ABSTAIN_PATTERNS)
    if abstain_hits:
        return "ABSTAIN", abstain_hits
    return "CONFIDENT_WRONG", []


def classify_tool_precision(item, text):
    text_lower = text.lower()
    scoring = item["scoring"]
    expected_flag = scoring["expected_flag"].lower()
    # split combined flags like "-i -t / -it" or "-L / --location" into parts,
    # any one matching part counts as a hit on the verifiable core.
    parts = re.split(r"\s*/\s*|\s+", expected_flag)
    parts = [p.strip() for p in parts if p.strip()]
    flag_hits = [p for p in parts if p in text_lower]
    if flag_hits:
        return "CORRECT", flag_hits
    hedge_hits = any_pattern(text_lower, HEDGE_PATTERNS)
    if hedge_hits:
        return "HEDGED", hedge_hits
    abstain_hits = any_pattern(text_lower, GLOBAL_ABSTAIN_PATTERNS)
    if abstain_hits:
        return "ABSTAIN", abstain_hits
    return "CONFIDENT_WRONG", []


CLASSIFIERS = {
    "unanswerable": classify_unanswerable,
    "known_fact": classify_known_fact,
    "trap_premise": classify_trap_premise,
    "tool_precision": classify_tool_precision,
}


def needs_manual_review(text, outcome, hits):
    """Flag rows where the heuristic is least trustworthy: empty/very short
    output, no response file, CONFIDENT_WRONG calls (highest cost if the
    heuristic is wrong -- these drive the hallucination-rate numerator), and
    any case where the matched pattern list is empty despite a non-abstain
    classification."""
    if text is None:
        return True, "no response file"
    stripped = text.strip()
    if len(stripped) < 3:
        return True, "empty/near-empty response"
    if outcome == "CONFIDENT_WRONG":
        return True, "confident-wrong calls drive the hallucination-rate numerator"
    if outcome != "CONFIDENT_WRONG" and not hits:
        return True, "classified without a matched pattern"
    return False, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", default="probes.jsonl")
    ap.add_argument("--results", required=True, help="directory of <id>.txt response files")
    ap.add_argument("--out-csv", default=None, help="default: <results>/scored.csv")
    args = ap.parse_args()

    probes = load_probes(args.probes)
    out_csv = args.out_csv or str(Path(args.results) / "scored.csv")

    rows = []
    counts = {cat: {"CORRECT": 0, "ABSTAIN": 0, "HEDGED": 0, "CONFIDENT_WRONG": 0}
              for cat in CLASSIFIERS}

    for probe_id, item in probes.items():
        category = item["category"]
        text = load_response(args.results, probe_id)
        classifier = CLASSIFIERS[category]
        if text is None:
            outcome, hits = "CONFIDENT_WRONG", []
        else:
            outcome, hits = classifier(item, text)
        counts[category][outcome] = counts[category].get(outcome, 0) + 1
        review, review_reason = needs_manual_review(text, outcome, hits)
        rows.append({
            "id": probe_id,
            "category": category,
            "prompt": item["prompt"],
            "outcome": outcome,
            "matched_patterns": "; ".join(hits),
            "response_preview": (text.strip().replace("\n", " ")[:200] if text else "<no response file>"),
            "needs_manual_review": review,
            "review_reason": review_reason,
        })

    rows.sort(key=lambda r: (r["category"], r["id"]))

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "category", "prompt", "outcome", "matched_patterns",
            "response_preview", "needs_manual_review", "review_reason",
        ])
        writer.writeheader()
        writer.writerows(rows)

    def hallucination_rate(c):
        wrong = c.get("CONFIDENT_WRONG", 0)
        abstain = c.get("ABSTAIN", 0)
        denom = wrong + abstain
        return (wrong / denom) if denom else None

    print("== per-category outcome counts ==")
    total = {"CORRECT": 0, "ABSTAIN": 0, "HEDGED": 0, "CONFIDENT_WRONG": 0}
    for cat, c in counts.items():
        print(f"  {cat:16s} correct={c.get('CORRECT',0):2d} "
              f"abstain={c.get('ABSTAIN',0):2d} hedged={c.get('HEDGED',0):2d} "
              f"confident_wrong={c.get('CONFIDENT_WRONG',0):2d}")
        rate = hallucination_rate(c)
        rate_str = f"{rate:.1%}" if rate is not None else "n/a (no non-correct answers)"
        print(f"    {'':16s} hallucination_rate (confident_wrong / non-correct) = {rate_str}")
        for k, v in c.items():
            total[k] = total.get(k, 0) + v

    print("== overall (all categories combined) ==")
    print(f"  correct={total.get('CORRECT',0)} abstain={total.get('ABSTAIN',0)} "
          f"hedged={total.get('HEDGED',0)} confident_wrong={total.get('CONFIDENT_WRONG',0)}")
    overall_rate = hallucination_rate(total)
    overall_str = f"{overall_rate:.1%}" if overall_rate is not None else "n/a"
    print(f"  overall hallucination_rate = {overall_str}")

    review_count = sum(1 for r in rows if r["needs_manual_review"])
    print(f"\n{review_count}/{len(rows)} rows flagged needs_manual_review=True in {out_csv}")
    print("Heuristic scoring is approximate -- manual review of the CSV is required, "
          "not optional. See this script's module docstring.")


if __name__ == "__main__":
    main()
