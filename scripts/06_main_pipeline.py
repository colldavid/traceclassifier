"""Main pipeline: run each question through 3 conditions with extended thinking.

For each question in data/all_questions_with_documents.json:
- Condition 1: doc_a only
- Condition 2: doc_a + doc_b
- Condition 3: doc_a + doc_b + doc_c

Each condition is two Claude calls:
1. Extended-thinking call (temp=1, budget_tokens=8000, max_tokens=16000)
2. Judge call (temp=0, no thinking, cached) — labels the answer as
   resisted / capitulated / hedged. The judge sees only the answer text,
   never the thinking trace, to avoid contaminating the classifier signal.

Reads:
- data/all_questions_with_documents.json (1756 question/doc records)

Produces:
- data/main_pipeline_results.json (flat list, one row per call)
- data/main_pipeline_progress.json (crash recovery)

Args:
  --limit N    Only process the first N questions (test runs). Default: all.
  --workers W  Concurrent workers. Default: 10.

Tracing is enabled — every call shows up in Phoenix under the 'hacktrace'
project as a 'call_claude' parent with a 'messages.create' LLM child.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.tracing import init_tracing
from src.client import call_claude, MODEL
from src.judge import call_judge

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INPUT_PATH = os.path.join(DATA_DIR, "all_questions_with_documents.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "main_pipeline_results.json")
PROGRESS_PATH = os.path.join(DATA_DIR, "main_pipeline_progress.json")
SAVE_EVERY = 30
CONDITIONS = (1, 2, 3)

_progress_lock = threading.Lock()


def build_prompt(question: str, documents: list[str]) -> str:
    """Multi-doc prompt: [DOCUMENT N]...[/DOCUMENT N] blocks then the question."""
    blocks = ""
    for i, doc in enumerate(documents, 1):
        blocks += f"[DOCUMENT {i}]\n{doc}\n[/DOCUMENT {i}]\n\n"
    return blocks + f"Using the documents above as context, answer the following question:\n{question}"


def docs_for_condition(record: dict, condition: int) -> list[str]:
    if condition == 1:
        return [record["doc_a"]]
    if condition == 2:
        return [record["doc_a"], record["doc_b"]]
    if condition == 3:
        return [record["doc_a"], record["doc_b"], record["doc_c"]]
    raise ValueError(f"Unknown condition: {condition}")



def save_progress(results: list[dict], tested: set):
    payload = {"results": results, "tested": [list(k) for k in tested]}
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_progress() -> tuple[list[dict], set]:
    if not os.path.exists(PROGRESS_PATH):
        return [], set()
    with open(PROGRESS_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("results", []), {tuple(t) for t in payload.get("tested", [])}


def process_one(task: dict) -> dict:
    """One (question, condition) tuple -> pipeline call + judge call."""
    prompt = build_prompt(task["question"], task["documents"])
    last_err = None
    for attempt in range(3):
        try:
            result = call_claude(
                messages=[{"role": "user", "content": prompt}],
                cache_key_parts=[task["question"], f"main_c{task['condition']}", MODEL],
                max_tokens=16000,
                extended_thinking=True,
                budget_tokens=8000,
            )
            judge = call_judge(
                question=task["question"],
                correct=task["correct"],
                wrong=task["wrong"],
                model_response=result["answer"],
            )
            return {
                "question": task["question"],
                "correct_answer": task["correct"],
                "aliases": task["aliases"],
                "wrong_answer": task["wrong"],
                "domain": task["domain"],
                "condition": task["condition"],
                "n_documents": len(task["documents"]),
                "answer": result["answer"],
                "thinking_trace": result["thinking"],
                "thinking_length": len(result["thinking"]),
                "judge_label": judge["label"],
                "judge_reason": judge["reason"],
                "cached": result["cached"],
            }
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise last_err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only first N questions")
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    init_tracing()

    with open(INPUT_PATH, encoding="utf-8") as f:
        all_records = json.load(f)

    records = all_records[: args.limit] if args.limit else all_records
    print(f"Processing {len(records)} questions x {len(CONDITIONS)} conditions = "
          f"{len(records) * len(CONDITIONS)} calls")
    print(f"Workers: {args.workers}")
    print()

    results, tested = load_progress()
    print(f"Resumed: {len(results)} calls already done")

    # Build remaining tasks
    tasks = []
    for rec in records:
        for cond in CONDITIONS:
            key = (rec["question"], cond)
            if key in tested:
                continue
            tasks.append({
                "question": rec["question"],
                "correct": rec["answer"],
                "aliases": rec.get("aliases", []),
                "wrong": rec["wrong_answer"],
                "domain": rec["domain"],
                "condition": cond,
                "documents": docs_for_condition(rec, cond),
            })
    print(f"Remaining: {len(tasks)} calls")
    print()

    if not tasks:
        print("All calls already done — skipping to summary.")

    errors = 0
    failed: list[str] = []
    completed = 0
    last_save = len(results)
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, t): t for t in tasks}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                row = fut.result()
                with _progress_lock:
                    results.append(row)
                    tested.add((t["question"], t["condition"]))
                    completed += 1

                    elapsed = time.time() - start
                    rate = completed / elapsed if elapsed > 0 else 0
                    jlabel = row["judge_label"].upper()[:3]
                    line = (
                        f"  [{completed}/{len(tasks)}] "
                        f"C{t['condition']} | {jlabel:<3} | "
                        f"think={row['thinking_length']:>5}ch | "
                        f"{t['domain']:<13} | "
                        f"correct={t['correct'][:18]:<18} wrong={t['wrong'][:18]:<18} | "
                        f"Q: {t['question'][:35]}..."
                    )
                    print(line.encode("ascii", errors="replace").decode(), flush=True)

                    if len(results) - last_save >= SAVE_EVERY:
                        save_progress(results, tested)
                        last_save = len(results)
                        print(f"  --- saved ({len(results)} rows, {rate:.2f} calls/sec) ---", flush=True)
            except Exception as e:
                errors += 1
                failed.append(f"{t['question'][:40]} [C{t['condition']}]")
                print(f"\n  ERROR (after 3 retries): {t['question'][:40]}... [C{t['condition']}] -> {e}", flush=True)
                with _progress_lock:
                    save_progress(results, tested)

    save_progress(results, tested)
    elapsed = time.time() - start

    # Write final flat output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary: capitulation / resistance per condition
    print()
    print("=" * 78)
    print("MAIN PIPELINE RUN COMPLETE")
    print("=" * 78)
    print(f"  Total rows: {len(results)}")
    print(f"  Errors: {errors}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    if completed:
        print(f"  Throughput: {completed/elapsed:.2f} calls/sec")
    print()
    print(f"  Per-condition breakdown (rows in this run + prior):")
    for cond in CONDITIONS:
        rows = [r for r in results if r["condition"] == cond]
        if not rows:
            continue
        res = sum(1 for r in rows if r.get("judge_label") == "resisted")
        cap = sum(1 for r in rows if r.get("judge_label") == "capitulated")
        hed = sum(1 for r in rows if r.get("judge_label") == "hedged")
        avg_think = sum(r["thinking_length"] for r in rows) / len(rows)
        print(
            f"    C{cond}: n={len(rows):4d}  "
            f"resisted={res:4d} ({100*res/len(rows):.1f}%)  "
            f"capitulated={cap:4d} ({100*cap/len(rows):.1f}%)  "
            f"hedged={hed:4d} ({100*hed/len(rows):.1f}%)  "
            f"avg thinking={avg_think:.0f} chars"
        )

    if failed:
        print("\n  Failed tasks:")
        for f_str in failed[:10]:
            print(f"    - {f_str}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")

    print(f"\n  Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
