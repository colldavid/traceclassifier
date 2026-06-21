"""Step 2: Generate wrong answers for the 1557 new questions.

Reads:
- data/all_validated_questions.json (1757 merged questions)
- data/all_questions_with_wrong_answers.json (200 with existing wrong answers)

Produces:
- data/all_questions_with_wrong_answers.json (1757 with wrong answers)
- data/wrong_answer_progress.json (crash recovery, temporary)

Uses the same wrong answer generation logic as modeltraceprep:
- Prompt asks for plausible wrong answer by type (person swap, date shift, etc.)
- temp=0, no thinking, max_tokens=1000
- All calls cached in Redis
- Concurrent execution with ThreadPoolExecutor
"""

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
from src.wrong_answer import generate_wrong_answer

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
PROGRESS_PATH = os.path.join(DATA_DIR, "wrong_answer_progress.json")
SAVE_EVERY = 50
MAX_WORKERS = 5

# Thread-safe progress tracking
_progress_lock = threading.Lock()


def save_progress(completed: list[dict], tested_questions: set[str]):
    payload = {
        "completed": completed,
        "tested_questions": list(tested_questions),
    }
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_progress() -> tuple[list[dict], set[str]]:
    if not os.path.exists(PROGRESS_PATH):
        return [], set()
    with open(PROGRESS_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["completed"], set(payload["tested_questions"])


def process_one(q: dict) -> dict:
    """Generate a wrong answer for a single question with retry logic."""
    question = q["question"]
    correct = q["answer"]
    last_err = None
    for attempt in range(3):
        try:
            wrong = generate_wrong_answer(question, correct)
            return {**q, "wrong_answer": wrong}
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise last_err


def main():
    init_tracing()

    # Load all 1757 questions
    with open(os.path.join(DATA_DIR, "all_validated_questions.json"), encoding="utf-8") as f:
        all_questions = json.load(f)

    # Load existing wrong answers (200 from modeltraceprep)
    with open(os.path.join(DATA_DIR, "all_questions_with_wrong_answers.json"), encoding="utf-8") as f:
        existing_wrong = json.load(f)

    existing_lookup = {q["question"]: q["wrong_answer"] for q in existing_wrong}
    print(f"Total questions: {len(all_questions)}")
    print(f"Existing wrong answers: {len(existing_lookup)}")

    # Load crash recovery progress
    progress_completed, progress_tested = load_progress()
    progress_lookup = {q["question"]: q["wrong_answer"] for q in progress_completed}
    print(f"Resumed progress: {len(progress_completed)} completed, {len(progress_tested)} tested")

    # Merge existing + progress lookups
    all_existing = {**existing_lookup, **progress_lookup}

    # Find questions that still need wrong answers
    need_wrong = [q for q in all_questions if q["question"] not in all_existing]
    print(f"Still need wrong answers: {len(need_wrong)}")
    print()

    if not need_wrong:
        print("All questions already have wrong answers!")
    else:
        print("=" * 60)
        print(f"Generating wrong answers for {len(need_wrong)} questions ({MAX_WORKERS} workers)")
        print("=" * 60)

    new_completed = list(progress_completed)  # start from progress
    tested = set(progress_tested)
    last_save = len(new_completed)
    errors = 0
    failed_questions = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks, skipping already-tested questions
        futures = {}
        for q in need_wrong:
            if q["question"] in tested:
                continue
            future = executor.submit(process_one, q)
            futures[future] = q

        for future in as_completed(futures):
            q = futures[future]
            try:
                result = future.result()
                with _progress_lock:
                    tested.add(q["question"])
                    new_completed.append(result)
                    count = len(new_completed)

                line = (
                    f"  [{count}/{len(need_wrong)}] "
                    f"Q: {q['question'][:50]}... | "
                    f"Correct: {q['answer']} | "
                    f"Wrong: {result['wrong_answer']}"
                )
                print(line.encode("ascii", errors="replace").decode(), flush=True)

                # Save progress periodically
                with _progress_lock:
                    if len(new_completed) - last_save >= SAVE_EVERY:
                        save_progress(new_completed, tested)
                        last_save = len(new_completed)
                        print(f"  --- Progress saved: {len(new_completed)} completed ---", flush=True)

            except Exception as e:
                errors += 1
                failed_questions.append(q["question"][:60])
                print(
                    f"\n  ERROR (after 3 retries): {q['question'][:50]}... -> {e}",
                    flush=True,
                )
                # Do NOT mark as tested — will be retried on next run
                with _progress_lock:
                    save_progress(new_completed, tested)

    # Final save of progress
    save_progress(new_completed, tested)

    elapsed = time.time() - start

    # Merge: existing 200 + newly generated
    final = list(existing_wrong)  # start with original 200
    new_lookup = {q["question"]: q["wrong_answer"] for q in new_completed}

    for q in all_questions:
        if q["question"] in existing_lookup:
            continue  # already in final from existing_wrong
        if q["question"] in new_lookup:
            final.append({**q, "wrong_answer": new_lookup[q["question"]]})

    # Save final dataset
    output_path = os.path.join(DATA_DIR, "all_questions_with_wrong_answers.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    # Verification
    print()
    print("=" * 60)
    print("STEP 2 COMPLETE")
    print("=" * 60)
    print(f"  Total with wrong answers: {len(final)}")
    print(f"  Errors (will retry on next run): {errors}")
    print(f"  Elapsed: {elapsed:.0f}s")

    if failed_questions:
        print(f"  Failed questions:")
        for fq in failed_questions[:10]:
            print(f"    - {fq}")
        if len(failed_questions) > 10:
            print(f"    ... and {len(failed_questions) - 10} more")

    # Check: all non-empty, wrong != correct
    empty = sum(1 for q in final if not q.get("wrong_answer", "").strip())
    same = sum(1 for q in final if q.get("wrong_answer", "").strip().lower() == q["answer"].strip().lower())
    print(f"  Empty wrong answers: {empty}")
    print(f"  Wrong == correct: {same}")

    if empty > 0 or same > 0:
        print("  WARNING: issues found!")
    elif errors > 0:
        print(f"  Run again to retry {errors} failed questions.")
    else:
        print("  All checks passed.")

    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
