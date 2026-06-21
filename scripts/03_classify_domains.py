"""Step 3: Classify each of the 1757 questions into a domain bucket.

Reads:
- data/all_questions_with_wrong_answers.json (1757 questions)

Produces:
- data/domain_classifications.json (1757 questions + 'domain' field)
- data/domain_progress.json (crash recovery, temporary)
- data/domain_fallbacks.json (questions that defaulted to 'general' because
  the model returned an out-of-set label even after retry — spot-check these)

Uses src.document_gen.classify_question_domain:
- Single Claude call, temp=0, max_tokens=20, cached in Redis
- Out-of-set labels trigger one retry then default to 'general' (logged)
- Concurrent execution with ThreadPoolExecutor (5 workers, same as Step 2)
"""

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.tracing import init_tracing
from src.document_gen import classify_question_domain, DOMAIN_BUCKETS, VENUE_POOLS

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INPUT_PATH = os.path.join(DATA_DIR, "all_questions_with_wrong_answers.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "domain_classifications.json")
PROGRESS_PATH = os.path.join(DATA_DIR, "domain_progress.json")
FALLBACK_PATH = os.path.join(DATA_DIR, "domain_fallbacks.json")
SAVE_EVERY = 50
MAX_WORKERS = 5

_progress_lock = threading.Lock()


def save_progress(completed: list[dict], tested: set[str], fallbacks: list[dict]):
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"completed": completed, "tested_questions": list(tested), "fallbacks": fallbacks},
            f,
            indent=2,
            ensure_ascii=False,
        )


def load_progress() -> tuple[list[dict], set[str], list[dict]]:
    if not os.path.exists(PROGRESS_PATH):
        return [], set(), []
    with open(PROGRESS_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    return (
        payload.get("completed", []),
        set(payload.get("tested_questions", [])),
        payload.get("fallbacks", []),
    )


def process_one(q: dict) -> tuple[dict, bool]:
    """Classify a single question. Returns (record_with_domain, fell_back)."""
    last_err = None
    for attempt in range(3):
        try:
            bucket, fell_back = classify_question_domain(q["question"], q["answer"])
            return {**q, "domain": bucket}, fell_back
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise last_err


def main():
    init_tracing()

    with open(INPUT_PATH, encoding="utf-8") as f:
        all_questions = json.load(f)

    print(f"Total questions: {len(all_questions)}")

    progress_completed, tested, fallbacks = load_progress()
    print(f"Resumed: {len(progress_completed)} classified, {len(fallbacks)} fallbacks so far")

    todo = [q for q in all_questions if q["question"] not in tested]
    print(f"Still need classification: {len(todo)}")
    print()

    if not todo:
        print("All questions already classified.")
    else:
        print("=" * 60)
        print(f"Classifying {len(todo)} questions ({MAX_WORKERS} workers)")
        print("=" * 60)

    new_completed = list(progress_completed)
    last_save = len(new_completed)
    errors = 0
    failed_questions: list[str] = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one, q): q for q in todo}

        for future in as_completed(futures):
            q = futures[future]
            try:
                record, fell_back = future.result()
                with _progress_lock:
                    tested.add(q["question"])
                    new_completed.append(record)
                    if fell_back:
                        fallbacks.append({"question": q["question"], "answer": q["answer"]})
                    count = len(new_completed)

                marker = " [FALLBACK]" if fell_back else ""
                line = (
                    f"  [{count}/{len(todo) + len(progress_completed)}] "
                    f"{record['domain']:<14}{marker} | "
                    f"Q: {q['question'][:50]}..."
                )
                print(line.encode("ascii", errors="replace").decode(), flush=True)

                with _progress_lock:
                    if len(new_completed) - last_save >= SAVE_EVERY:
                        save_progress(new_completed, tested, fallbacks)
                        last_save = len(new_completed)
                        print(f"  --- Progress saved: {len(new_completed)} ---", flush=True)

            except Exception as e:
                errors += 1
                failed_questions.append(q["question"][:60])
                print(
                    f"\n  ERROR (after 3 retries): {q['question'][:50]}... -> {e}",
                    flush=True,
                )
                with _progress_lock:
                    save_progress(new_completed, tested, fallbacks)

    save_progress(new_completed, tested, fallbacks)
    elapsed = time.time() - start

    # Persist final dataset preserving original question order
    by_question = {q["question"]: q for q in new_completed}
    final = [by_question[q["question"]] for q in all_questions if q["question"] in by_question]

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    with open(FALLBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(fallbacks, f, indent=2, ensure_ascii=False)

    distribution = Counter(q["domain"] for q in final)

    print()
    print("=" * 60)
    print("STEP 3 COMPLETE")
    print("=" * 60)
    print(f"  Total classified: {len(final)}")
    print(f"  Errors (will retry on next run): {errors}")
    print(f"  Fallbacks to 'general' (out-of-set after retry): {len(fallbacks)}")
    print(f"  Elapsed: {elapsed:.0f}s")
    print()
    print("  Bucket distribution:")
    for bucket in DOMAIN_BUCKETS:
        n = distribution.get(bucket, 0)
        pct = (100.0 * n / len(final)) if final else 0.0
        venues = ", ".join(VENUE_POOLS[bucket])
        print(f"    {bucket:<14} {n:>5} ({pct:5.1f}%)  venues: {venues}")

    if failed_questions:
        print("\n  Failed questions:")
        for fq in failed_questions[:10]:
            print(f"    - {fq}")
        if len(failed_questions) > 10:
            print(f"    ... and {len(failed_questions) - 10} more")

    print(f"\n  Saved classifications to {OUTPUT_PATH}")
    print(f"  Saved fallback list to {FALLBACK_PATH}")


if __name__ == "__main__":
    main()
