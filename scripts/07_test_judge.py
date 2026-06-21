"""Run the judge on data/main_pipeline_results.json and print label distribution.

Uses 5 workers. Adds judge_label and judge_reason to each row and saves the
labelled result to data/judge_test_results.json.
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.tracing import init_tracing
from src.judge import call_judge

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INPUT_PATH = os.path.join(DATA_DIR, "main_pipeline_results.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "judge_test_results.json")


def judge_one(row: dict) -> dict:
    j = call_judge(
        question=row["question"],
        correct=row["correct_answer"],
        wrong=row["wrong_answer"],
        model_response=row["answer"],
    )
    return {**row, "judge_label": j["label"], "judge_reason": j["reason"], "judge_cached": j["cached"]}


def main():
    init_tracing()

    with open(INPUT_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    print(f"Judging {len(rows)} rows with 5 workers...")
    print()

    results = []
    errors = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(judge_one, r): r for r in rows}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                result = fut.result()
                results.append(result)
                cap_old = "CAP" if r["capitulated"] else "   "
                label = result["judge_label"].upper()[:3]
                print(
                    f"  [{len(results):2d}/{len(rows)}] C{r['condition']} | "
                    f"old={cap_old} judge={label:<3} | "
                    f"correct={r['correct_answer'][:15]:<15} wrong={r['wrong_answer'][:15]:<15} | "
                    f"{result['judge_reason'][:60]}"
                )
            except Exception as e:
                errors += 1
                print(f"  ERROR: {r['question'][:40]} -> {e}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s. Errors: {errors}")
    print()

    # Distribution
    from collections import Counter
    label_counts = Counter(r["judge_label"] for r in results if "judge_label" in r)
    total = sum(label_counts.values())
    print("Label distribution:")
    for label in ("resisted", "capitulated", "hedged"):
        n = label_counts.get(label, 0)
        print(f"  {label:<12}: {n:3d} ({100*n/total:.1f}%)")
    print()

    # Per-condition breakdown
    print("Per-condition breakdown:")
    for cond in (1, 2, 3):
        subset = [r for r in results if r["condition"] == cond]
        if not subset:
            continue
        c = Counter(r["judge_label"] for r in subset)
        n = len(subset)
        print(f"  C{cond} (n={n}): "
              f"resisted={c.get('resisted',0)} ({100*c.get('resisted',0)/n:.0f}%)  "
              f"capitulated={c.get('capitulated',0)} ({100*c.get('capitulated',0)/n:.0f}%)  "
              f"hedged={c.get('hedged',0)} ({100*c.get('hedged',0)/n:.0f}%)")
    print()

    # Spot-check: a few examples of each label
    for label in ("resisted", "capitulated", "hedged"):
        examples = [r for r in results if r.get("judge_label") == label][:2]
        if examples:
            print(f"--- {label.upper()} examples ---")
            for ex in examples:
                print(f"  Q: {ex['question'][:60]}")
                print(f"  correct={ex['correct_answer']} | wrong={ex['wrong_answer']}")
                print(f"  ans: {ex['answer'][:120]}")
                print(f"  reason: {ex['judge_reason']}")
                print()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
