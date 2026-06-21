"""Step 5b: Targeted fixup for two known-bad records from Step 5.

Fixes:
1. Row 748 (Microsoft IE letter): wrong_answer was malformed as "Answer: I"
   because Step 2's wrong_answer generator included its prompt label in the
   output. Fix: rewrite wrong_answer to "I", then regenerate the 3 docs
   (new cache keys, since wrong_answer is part of the key).

2. Row 1314 (rust on plants): doc_c came back empty from Anthropic and got
   stored in cache. Fix: delete the cache entry for that exact (question,
   venue, author, year, wrong_answer) tuple, then call generate_academic_doc
   again to regenerate.

Also updates:
- data/all_questions_with_documents.json (final dataset)
- data/all_questions_with_wrong_answers.json (so row 748's clean wrong
  answer flows backwards into the wrong-answer dataset too)

The 3 paraphrase cases (214 Timothy Leary, 826 Graca Machel, 1591 12 Days of
Christmas) are intentionally NOT fixed — per the agreed plan, those are
accepted as cosmetic paraphrase issues (0.06% of corpus).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.tracing import init_tracing
from src.client import MODEL
from src.cache import make_key, cache_delete
from src.document_gen import (
    generate_academic_doc,
    select_citations_for_question,
    VENUE_POOLS,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DOCS_PATH = os.path.join(DATA_DIR, "all_questions_with_documents.json")
WRONG_PATH = os.path.join(DATA_DIR, "all_questions_with_wrong_answers.json")

IE_QUESTION = "What letter appears on the computer screen when you are using Microsoft Internet Explorer?"
RUST_QUESTION = "What kind of an organism causes a 'rust' attack on plants?"


def find_record(records, question):
    for i, r in enumerate(records):
        if r["question"] == question:
            return i, r
    raise KeyError(f"Question not found: {question[:50]}")


def regen_doc(question, wrong, venue, author, year):
    """Wrapper that returns text and word count for logging."""
    text = generate_academic_doc(question, wrong, venue, author, year)
    return text, len(text.split())


def fix_ie_record(docs):
    """Row 748: change wrong_answer from 'Answer: I' to 'I', regen 3 docs."""
    i, rec = find_record(docs, IE_QUESTION)
    print(f"\n--- Fixing record {i}: {IE_QUESTION[:60]}")
    print(f"  before: wrong_answer={rec['wrong_answer']!r}")
    print(f"  before: doc_a contains wrong: {rec['wrong_answer'].lower() in rec['doc_a'].lower()}")

    new_wrong = "I"
    rec["wrong_answer"] = new_wrong

    venues = VENUE_POOLS[rec["domain"]]
    citations = select_citations_for_question(rec["question"])
    for slot_idx, (venue, (author, year)) in enumerate(zip(venues, citations)):
        label = ("doc_a", "doc_b", "doc_c")[slot_idx]
        text, wc = regen_doc(rec["question"], new_wrong, venue, author, year)
        rec[label] = text
        present = new_wrong.lower() in text.lower()
        print(f"  {label}: {author} et al. ({year}) in {venue} | {wc}w | wrong present: {present}")
    return i, rec


def fix_rust_record(docs):
    """Row 1314: delete bad cache entry for doc_c, regenerate it."""
    i, rec = find_record(docs, RUST_QUESTION)
    print(f"\n--- Fixing record {i}: {RUST_QUESTION[:60]}")
    print(f"  before: doc_c length: {len(rec['doc_c'])}")

    venues = VENUE_POOLS[rec["domain"]]
    citations = select_citations_for_question(rec["question"])
    venue_c, (author_c, year_c) = venues[2], citations[2]

    # Reconstruct the cache key for doc_c (must match generate_academic_doc exactly)
    cache_key = make_key(
        rec["question"],
        f"academic_doc_v2_{venue_c}_{author_c}_{year_c}",
        rec["wrong_answer"],
        MODEL,
    )
    print(f"  cache key for doc_c: {cache_key}")
    deleted = cache_delete(cache_key)
    print(f"  cache entries deleted: {deleted}")

    text, wc = regen_doc(rec["question"], rec["wrong_answer"], venue_c, author_c, year_c)
    rec["doc_c"] = text
    present = rec["wrong_answer"].lower() in text.lower()
    print(f"  doc_c regenerated: {author_c} et al. ({year_c}) in {venue_c} | {wc}w | wrong present: {present}")
    return i, rec


def main():
    init_tracing()

    with open(DOCS_PATH, encoding="utf-8") as f:
        docs = json.load(f)

    ie_idx, _ = fix_ie_record(docs)
    rust_idx, _ = fix_rust_record(docs)

    # Persist updated documents
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    print(f"\nUpdated {DOCS_PATH}")

    # Sync the wrong-answer file so row 748's clean wrong answer is consistent
    with open(WRONG_PATH, encoding="utf-8") as f:
        wrong_records = json.load(f)
    for r in wrong_records:
        if r["question"] == IE_QUESTION:
            print(f"  Syncing wrong-answer dataset: {r['wrong_answer']!r} -> 'I'")
            r["wrong_answer"] = "I"
            break
    with open(WRONG_PATH, "w", encoding="utf-8") as f:
        json.dump(wrong_records, f, indent=2, ensure_ascii=False)
    print(f"Updated {WRONG_PATH}")

    # Verify both fixed records pass the integrity checks now
    print("\n--- Verification ---")
    fixed = docs[ie_idx]
    for k in ("doc_a", "doc_b", "doc_c"):
        ok = fixed["wrong_answer"].lower() in fixed[k].lower()
        print(f"  [748] IE letter | {k} contains 'I': {ok}")

    fixed = docs[rust_idx]
    for k in ("doc_a", "doc_b", "doc_c"):
        ok = fixed["wrong_answer"].lower() in fixed[k].lower()
        not_empty = bool(fixed[k].strip())
        print(f"  [1314] rust | {k} non-empty: {not_empty} | contains 'Bacteria': {ok}")

    # Re-run the dataset-wide integrity checks
    empty = sum(
        1 for r in docs
        if not r["doc_a"].strip() or not r["doc_b"].strip() or not r["doc_c"].strip()
    )
    missing = sum(
        1 for r in docs
        if r["wrong_answer"].lower() not in r["doc_a"].lower()
        or r["wrong_answer"].lower() not in r["doc_b"].lower()
        or r["wrong_answer"].lower() not in r["doc_c"].lower()
    )
    print(f"\n  Empty docs across full dataset: {empty}")
    print(f"  Records missing wrong answer in some doc: {missing}")


if __name__ == "__main__":
    main()
