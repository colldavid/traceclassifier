"""Step 4: 5-question sanity check for academic document generation.

Picks 5 questions across high-volume domains (entertainment, sports, science,
history, music) and generates doc_a / doc_b / doc_c for each. Uses
generate_three_academic_docs, which pins author + year + venue per doc via
select_citations_for_question + VENUE_POOLS.

Concurrent: 5 workers across all 15 calls (5 questions x 3 docs).

Verifies:
- Author names actually vary across doc_a/b/c (the key fix being tested)
- Years vary
- Journals match the expected venues from the domain pool
- Word counts land 80-120
- Wrong answer appears in each doc, ideally in the first sentence

Writes full output to data/sanity_check_step4.json for the record.
"""

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

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
OUTPUT_PATH = os.path.join(DATA_DIR, "sanity_check_step4.json")
MAX_WORKERS = 5

TARGET_DOMAINS = ["entertainment", "sports", "science", "history", "music"]


# Catches both formats:
#   "Henderson et al. (1998)"   -> Henderson et al., 1998
#   "(Henderson et al., 1998)"  -> Henderson et al., 1998
CITATION_RE_PRE = re.compile(
    r"([A-Z][A-Za-z\-']+(?:\s+(?:and|&)\s+[A-Z][A-Za-z\-']+)?(?:\s+et\s+al\.?)?)"
    r"\s*\(\s*(\d{4})\s*\)"
)
CITATION_RE_PAREN = re.compile(
    r"\(\s*([A-Z][A-Za-z\-']+(?:\s+(?:and|&)\s+[A-Z][A-Za-z\-']+)?(?:\s+et\s+al\.?)?),\s*(\d{4})\s*\)"
)


def extract_citations(text: str) -> list[tuple[str, str]]:
    hits = [(m.group(1).strip(), m.group(2)) for m in CITATION_RE_PRE.finditer(text)]
    hits += [(m.group(1).strip(), m.group(2)) for m in CITATION_RE_PAREN.finditer(text)]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            unique.append(h)
    return unique


def first_sentence(text: str) -> str:
    for end in [". ", ".\n"]:
        if end in text:
            return text.split(end, 1)[0] + "."
    return text


def word_count(text: str) -> int:
    return len(text.split())


def main():
    init_tracing()

    with open(INPUT_PATH, encoding="utf-8") as f:
        all_questions = json.load(f)

    picks = []
    seen_domains = set()
    for q in all_questions:
        d = q["domain"]
        if d in TARGET_DOMAINS and d not in seen_domains:
            picks.append(q)
            seen_domains.add(d)
        if len(seen_domains) == len(TARGET_DOMAINS):
            break

    # Build the 15 generation tasks
    tasks = []
    for q in picks:
        venues = VENUE_POOLS[q["domain"]]
        citations = select_citations_for_question(q["question"])
        for slot, (venue, (author, year)) in enumerate(zip(venues, citations)):
            label = ["doc_a", "doc_b", "doc_c"][slot]
            tasks.append({
                "q": q,
                "label": label,
                "venue": venue,
                "author": author,
                "year": year,
            })

    print("=" * 78)
    print(f"STEP 4 SANITY CHECK (v2) - {len(picks)} questions x 3 docs = {len(tasks)} generations")
    print(f"Concurrent workers: {MAX_WORKERS}")
    print("=" * 78)
    print()

    def run_task(t):
        return t["label"], generate_academic_doc(
            t["q"]["question"], t["q"]["wrong_answer"],
            t["venue"], t["author"], t["year"],
        )

    start = time.time()
    results: dict[tuple[str, str], str] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(run_task, t): t for t in tasks}
        for fut in futures:
            t = futures[fut]
            label, text = fut.result()
            results[(t["q"]["question"], label)] = text
    elapsed = time.time() - start

    print(f"All {len(tasks)} generations done in {elapsed:.1f}s")
    print()

    records = []
    summary_rows = []

    for idx, q in enumerate(picks, 1):
        domain = q["domain"]
        venues = VENUE_POOLS[domain]
        citations = select_citations_for_question(q["question"])

        print(f"### {idx}. [{domain}] {q['question']}")
        print(f"    correct: {q['answer']}  |  wrong: {q['wrong_answer']}")
        for slot, ((author, year), venue) in enumerate(zip(citations, venues)):
            label = ["doc_a", "doc_b", "doc_c"][slot]
            print(f"    {label}: assigned {author} et al. ({year}) in {venue}")
        print()

        slot_data = []
        for slot in range(3):
            label = ["doc_a", "doc_b", "doc_c"][slot]
            author, year = citations[slot]
            venue = venues[slot]
            text = results[(q["question"], label)]
            wc = word_count(text)
            found_citations = extract_citations(text)
            wrong_in = q["wrong_answer"].lower() in text.lower()
            wrong_in_first = q["wrong_answer"].lower() in first_sentence(text).lower()
            venue_in = venue.lower() in text.lower()
            author_in = author.lower() in text.lower()
            year_in = str(year) in text

            print(f"--- {label}  (assigned: {author} et al. ({year}) in {venue}) ---")
            print(text)
            print()
            print(
                f"  words: {wc}  |  venue verbatim: {venue_in}  |  "
                f"author verbatim: {author_in}  |  year verbatim: {year_in}  |  "
                f"wrong in 1st sentence: {wrong_in_first}"
            )
            print(f"  citations found by regex: {found_citations}")
            print()

            slot_data.append({
                "label": label, "venue": venue, "assigned_author": author,
                "assigned_year": year, "text": text, "word_count": wc,
                "venue_verbatim": venue_in, "author_verbatim": author_in,
                "year_verbatim": year_in, "wrong_in_first_sentence": wrong_in_first,
                "extracted_citations": found_citations,
            })

        authors_used = [s["assigned_author"] for s in slot_data]
        years_used = [s["assigned_year"] for s in slot_data]
        all_verbatim = all(
            s["author_verbatim"] and s["year_verbatim"] and s["venue_verbatim"]
            for s in slot_data
        )

        print(f"  >>> assigned authors: {authors_used} (all distinct: {len(set(authors_used)) == 3})")
        print(f"  >>> assigned years:   {years_used} (all distinct: {len(set(years_used)) == 3})")
        print(f"  >>> all citations honored verbatim across all 3 docs: {all_verbatim}")
        print()
        print("=" * 78)
        print()

        records.append({
            "domain": domain, "question": q["question"], "answer": q["answer"],
            "wrong_answer": q["wrong_answer"], "slots": slot_data,
            "all_citations_verbatim": all_verbatim,
        })
        summary_rows.append({
            "domain": domain, "all_verbatim": all_verbatim,
            "details": [
                (s["label"], s["venue_verbatim"], s["author_verbatim"], s["year_verbatim"])
                for s in slot_data
            ],
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Wall time for {len(tasks)} generations ({MAX_WORKERS} workers): {elapsed:.1f}s")
    print()
    for row in summary_rows:
        marker = "OK" if row["all_verbatim"] else "FAIL"
        print(f"  [{marker}] {row['domain']:<14}", end="")
        for label, v, a, y in row["details"]:
            flags = ""
            if not v: flags += "V"
            if not a: flags += "A"
            if not y: flags += "Y"
            flags = flags or "-"
            print(f"  {label}:{flags}", end="")
        print()
    print()
    print("  Legend: V=venue missing, A=author missing, Y=year missing, -=all 3 verbatim")
    all_ok = all(r["all_verbatim"] for r in summary_rows)
    print(f"\n  All 5 questions have all 3 citations honored verbatim: {all_ok}")
    print(f"\n  Saved full output to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
