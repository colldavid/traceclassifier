"""Step 5: Generate 3 academic docs per question for all 1757 questions.

Reads:
- data/domain_classifications.json (1757 questions with 'domain' field)

Produces:
- data/all_questions_with_documents.json (1757 records, each with doc_a/b/c)
- data/documents_progress.json (crash recovery, temporary)

For each question, generates doc_a / doc_b / doc_c using the same TIER3_PROMPT
template, with each doc pinned to a distinct (author, year, venue) triple via
src.document_gen.generate_academic_doc + select_citations_for_question.

Concurrency: 10 workers (chosen via benchmark — 1.47x faster than 5w, 0 errors
across 90 calls in benchmark). Parallelism is at the slot level: each of the
5271 (question, slot) tuples is an independent task, so workers don't sit idle
waiting for sibling docs on the same question.

Crash recovery: progress saved every 75 slot completions (= 25 questions worth).
Failed slots are NOT marked tested — they'll be retried on next run.
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
from src.document_gen import (
    generate_academic_doc,
    select_citations_for_question,
    VENUE_POOLS,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
INPUT_PATH = os.path.join(DATA_DIR, "domain_classifications.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "all_questions_with_documents.json")
PROGRESS_PATH = os.path.join(DATA_DIR, "documents_progress.json")
SAVE_EVERY_SLOTS = 75
MAX_WORKERS = 10
SLOT_LABELS = ("doc_a", "doc_b", "doc_c")

_progress_lock = threading.Lock()


def save_progress(slots: dict, tested: set):
    payload = {
        "slots": slots,
        "tested": [[q, s] for q, s in tested],
    }
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_progress() -> tuple[dict, set]:
    if not os.path.exists(PROGRESS_PATH):
        return {}, set()
    with open(PROGRESS_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("slots", {}), {tuple(t) for t in payload.get("tested", [])}


def process_one_slot(task: dict) -> str:
    last_err = None
    for attempt in range(3):
        try:
            return generate_academic_doc(
                task["question"], task["wrong"],
                task["venue"], task["author"], task["year"],
            )
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise last_err


def main():
    init_tracing()

    with open(INPUT_PATH, encoding="utf-8") as f:
        all_questions = json.load(f)
    print(f"Total questions: {len(all_questions)}")

    slots, tested = load_progress()
    existing_slots = sum(len(v) for v in slots.values())
    complete_questions = sum(1 for v in slots.values() if len(v) == 3)
    print(f"Resumed: {existing_slots} slots done across {len(slots)} questions "
          f"({complete_questions} fully complete)")

    # Build remaining (question, slot) tasks
    all_tasks = []
    for q in all_questions:
        venues = VENUE_POOLS[q["domain"]]
        citations = select_citations_for_question(q["question"])
        for slot_idx in range(3):
            label = SLOT_LABELS[slot_idx]
            if (q["question"], label) in tested:
                continue
            all_tasks.append({
                "question": q["question"],
                "wrong": q["wrong_answer"],
                "domain": q["domain"],
                "venue": venues[slot_idx],
                "author": citations[slot_idx][0],
                "year": citations[slot_idx][1],
                "slot_label": label,
            })

    total_slots = len(all_questions) * 3
    print(f"Still need: {len(all_tasks)} slots (of {total_slots} total)")
    print()

    if not all_tasks:
        print("All slots already done.")
    else:
        print("=" * 70)
        print(f"Generating {len(all_tasks)} academic docs ({MAX_WORKERS} workers)")
        print("=" * 70)

    errors = 0
    failed: list[str] = []
    completed_count = 0
    last_save = 0
    start = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_one_slot, t): t for t in all_tasks}

        for fut in as_completed(futures):
            task = futures[fut]
            q = task["question"]
            label = task["slot_label"]
            try:
                text = fut.result()
                with _progress_lock:
                    if q not in slots:
                        slots[q] = {}
                    slots[q][label] = text
                    tested.add((q, label))
                    completed_count += 1

                    line = (
                        f"  [{completed_count}/{len(all_tasks)}] "
                        f"{label} | {task['domain']:<14} | "
                        f"{task['author']} et al. ({task['year']}) | "
                        f"Q: {q[:40]}..."
                    )
                    print(line.encode("ascii", errors="replace").decode(), flush=True)

                    if completed_count - last_save >= SAVE_EVERY_SLOTS:
                        save_progress(slots, tested)
                        last_save = completed_count
                        n_complete = sum(1 for v in slots.values() if len(v) == 3)
                        print(
                            f"  --- Saved: {completed_count} slots done this run | "
                            f"{n_complete}/{len(all_questions)} questions complete ---",
                            flush=True,
                        )
            except Exception as e:
                errors += 1
                failed.append(f"{q[:50]} [{label}]")
                print(f"\n  ERROR (after 3 retries): {q[:40]}... [{label}] -> {e}", flush=True)
                with _progress_lock:
                    save_progress(slots, tested)

    save_progress(slots, tested)
    elapsed = time.time() - start

    # Assemble final dataset (only questions where all 3 slots completed)
    final = []
    incomplete: list[str] = []
    for q in all_questions:
        qt = q["question"]
        if qt in slots and len(slots[qt]) == 3:
            final.append({
                "question": qt,
                "answer": q["answer"],
                "aliases": q.get("aliases", []),
                "wrong_answer": q["wrong_answer"],
                "domain": q["domain"],
                "doc_a": slots[qt]["doc_a"],
                "doc_b": slots[qt]["doc_b"],
                "doc_c": slots[qt]["doc_c"],
            })
        else:
            incomplete.append(qt)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print("STEP 5 COMPLETE")
    print("=" * 70)
    print(f"  Questions with all 3 docs: {len(final)}")
    print(f"  Incomplete questions: {len(incomplete)}")
    print(f"  Errors this run: {errors}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Throughput: {completed_count / elapsed:.2f} calls/sec")

    # Quick verification on the final dataset
    if final:
        empty_docs = sum(
            1 for r in final
            if not r["doc_a"].strip() or not r["doc_b"].strip() or not r["doc_c"].strip()
        )
        missing_wrong = sum(
            1 for r in final
            if r["wrong_answer"].lower() not in r["doc_a"].lower()
            or r["wrong_answer"].lower() not in r["doc_b"].lower()
            or r["wrong_answer"].lower() not in r["doc_c"].lower()
        )
        print(f"  Empty docs: {empty_docs}")
        print(f"  Records missing wrong answer in some doc: {missing_wrong}")

    if failed:
        print("\n  Failed slots (will retry on next run):")
        for f_str in failed[:10]:
            print(f"    - {f_str}")
        if len(failed) > 10:
            print(f"    ... and {len(failed)-10} more")

    print(f"\n  Saved to {OUTPUT_PATH}")
    if incomplete:
        print(f"  Run again to complete remaining {len(incomplete)} questions.")


if __name__ == "__main__":
    main()
