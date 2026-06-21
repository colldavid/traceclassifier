"""Benchmark: 5 vs 10 workers for academic document generation.

Picks 60 questions from the classified dataset, splits into two disjoint
groups of 30 (so neither group hits cache from the other arm). Runs each
group at one worker setting, measures wall time, throughput, and errors.

To keep arms comparable, both groups are picked from the same domain mix
(round-robin across domains so each group has the same distribution).
"""

import json
import os
import sys
import time
from collections import defaultdict
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

SANITY_CHECK_DOMAINS_FIRST = {"entertainment", "sports", "science", "history", "music"}


def pick_balanced(all_questions, exclude_questions: set, per_arm: int = 30):
    """Pick 2 disjoint groups of per_arm questions, both with similar domain mix."""
    by_domain = defaultdict(list)
    for q in all_questions:
        if q["question"] in exclude_questions:
            continue
        by_domain[q["domain"]].append(q)

    # Round-robin across domains
    domains = sorted(by_domain.keys())
    pool_a, pool_b = [], []
    idx_per_domain = {d: 0 for d in domains}
    target = 2 * per_arm
    while len(pool_a) + len(pool_b) < target:
        progress = False
        for d in domains:
            if len(pool_a) + len(pool_b) >= target:
                break
            i = idx_per_domain[d]
            if i < len(by_domain[d]):
                q = by_domain[d][i]
                if len(pool_a) <= len(pool_b):
                    pool_a.append(q)
                else:
                    pool_b.append(q)
                idx_per_domain[d] = i + 1
                progress = True
        if not progress:
            break
    return pool_a[:per_arm], pool_b[:per_arm]


def build_tasks(questions):
    tasks = []
    for q in questions:
        venues = VENUE_POOLS[q["domain"]]
        citations = select_citations_for_question(q["question"])
        for slot in range(3):
            venue = venues[slot]
            author, year = citations[slot]
            tasks.append({
                "question": q["question"], "wrong": q["wrong_answer"],
                "venue": venue, "author": author, "year": year, "slot": slot,
            })
    return tasks


def run_arm(label, tasks, workers):
    errors = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(generate_academic_doc, t["question"], t["wrong"],
                      t["venue"], t["author"], t["year"])
            for t in tasks
        ]
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                errors += 1
                print(f"  [{label}] error: {e}", flush=True)
    elapsed = time.time() - start
    return elapsed, errors


def main():
    init_tracing()

    with open(INPUT_PATH, encoding="utf-8") as f:
        all_questions = json.load(f)

    # Exclude the 5 sanity-check questions to keep this measurement on fresh keys
    exclude = set()
    for q in all_questions:
        if q["domain"] in SANITY_CHECK_DOMAINS_FIRST and q["domain"] not in {x["domain"] for x in []}:
            exclude.add(q["question"])
            break  # just one per domain
    # Actually the sanity check picked the FIRST one per domain — exclude those
    seen = set()
    exclude = set()
    for q in all_questions:
        if q["domain"] in SANITY_CHECK_DOMAINS_FIRST and q["domain"] not in seen:
            exclude.add(q["question"])
            seen.add(q["domain"])
        if len(seen) == len(SANITY_CHECK_DOMAINS_FIRST):
            break

    pool_a, pool_b = pick_balanced(all_questions, exclude, per_arm=30)
    tasks_a = build_tasks(pool_a)
    tasks_b = build_tasks(pool_b)
    assert len(tasks_a) == len(tasks_b) == 90, f"got {len(tasks_a)} / {len(tasks_b)}"

    # Domain distribution sanity check
    dist_a = defaultdict(int)
    for q in pool_a:
        dist_a[q["domain"]] += 1
    dist_b = defaultdict(int)
    for q in pool_b:
        dist_b[q["domain"]] += 1
    print("Domain mix per arm (questions):")
    print(f"  Arm A: {dict(dist_a)}")
    print(f"  Arm B: {dict(dist_b)}")
    print()

    print("=" * 70)
    print(f"ARM A: {len(tasks_a)} fresh calls at 5 workers")
    print("=" * 70)
    elapsed_a, errors_a = run_arm("A-5w", tasks_a, workers=5)
    print(f"  Wall time: {elapsed_a:.1f}s  |  Errors: {errors_a}  |  "
          f"Throughput: {len(tasks_a)/elapsed_a:.2f} calls/sec")
    print()

    # Cool-down between arms to avoid one arm benefiting from warmed-up state
    time.sleep(3)

    print("=" * 70)
    print(f"ARM B: {len(tasks_b)} fresh calls at 10 workers")
    print("=" * 70)
    elapsed_b, errors_b = run_arm("B-10w", tasks_b, workers=10)
    print(f"  Wall time: {elapsed_b:.1f}s  |  Errors: {errors_b}  |  "
          f"Throughput: {len(tasks_b)/elapsed_b:.2f} calls/sec")
    print()

    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    rps_a = len(tasks_a) / elapsed_a
    rps_b = len(tasks_b) / elapsed_b
    speedup = rps_b / rps_a if rps_a > 0 else 0
    print(f"  5 workers : {rps_a:.2f} calls/sec ({elapsed_a:.0f}s for 90 calls, {errors_a} errors)")
    print(f"  10 workers: {rps_b:.2f} calls/sec ({elapsed_b:.0f}s for 90 calls, {errors_b} errors)")
    print(f"  Speedup at 10w: {speedup:.2f}x")
    print()

    # Project to full Step 5
    total_step5 = 5271
    proj_5w = total_step5 / rps_a / 60
    proj_10w = total_step5 / rps_b / 60
    print(f"  Projected Step 5 wall time (5271 calls):")
    print(f"    5 workers : ~{proj_5w:.0f} min")
    print(f"    10 workers: ~{proj_10w:.0f} min")


if __name__ == "__main__":
    main()
