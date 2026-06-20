"""Step 1: Merge the 1757 validated questions into a single dataset.

Reads from the modeltraceprep folder:
- data/validated_questions.json (200 original)
- data/validated_new_progress.json (1557 new, in .validated array)
- data/questions_with_wrong_answers.json (200 with wrong answers)

Writes to Hacktrace data/:
- data/all_validated_questions.json (1757 merged)
"""

import json
import os

# Paths
MODELTRACEPREP = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "Model Traces",
    "modeltraceprep",
)
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── Load original 200 validated questions ─────────────────────────────
    path_200 = os.path.join(MODELTRACEPREP, "data", "validated_questions.json")
    with open(path_200, encoding="utf-8") as f:
        original_200 = json.load(f)
    print(f"Original validated questions: {len(original_200)}")

    # ── Load 1557 new validated questions ─────────────────────────────────
    path_new = os.path.join(MODELTRACEPREP, "data", "validated_new_progress.json")
    with open(path_new, encoding="utf-8") as f:
        progress = json.load(f)
    new_1557 = progress["validated"]
    print(f"New validated questions: {len(new_1557)}")

    # ── Merge and deduplicate ─────────────────────────────────────────────
    seen = set()
    merged = []
    dupes = 0
    for q in original_200 + new_1557:
        if q["question"] in seen:
            dupes += 1
            continue
        seen.add(q["question"])
        merged.append(q)

    print(f"\nMerged: {len(merged)} questions ({dupes} duplicates removed)")

    # ── Verify keys ───────────────────────────────────────────────────────
    for q in merged:
        assert "question" in q and "answer" in q and "aliases" in q, f"Missing keys: {q}"

    # ── Save merged questions ─────────────────────────────────────────────
    output_path = os.path.join(DATA_DIR, "all_validated_questions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_path}")

    # ── Copy existing wrong answers for the 200 ──────────────────────────
    path_wrong = os.path.join(MODELTRACEPREP, "data", "questions_with_wrong_answers.json")
    with open(path_wrong, encoding="utf-8") as f:
        wrong_200 = json.load(f)

    wrong_answer_lookup = {q["question"]: q["wrong_answer"] for q in wrong_200}
    print(f"\nExisting wrong answers loaded: {len(wrong_answer_lookup)}")

    # Build partial wrong answer dataset (200 with wrong answers, 1557 without)
    questions_with_wrong = []
    questions_without_wrong = []

    for q in merged:
        if q["question"] in wrong_answer_lookup:
            questions_with_wrong.append({
                **q,
                "wrong_answer": wrong_answer_lookup[q["question"]],
            })
        else:
            questions_without_wrong.append(q)

    print(f"Questions with wrong answers: {len(questions_with_wrong)}")
    print(f"Questions needing wrong answers: {len(questions_without_wrong)}")

    # Save partial dataset (will be completed in step 2)
    partial_path = os.path.join(DATA_DIR, "all_questions_with_wrong_answers.json")
    with open(partial_path, "w", encoding="utf-8") as f:
        json.dump(questions_with_wrong, f, indent=2, ensure_ascii=False)

    print(f"\nPartial wrong answer dataset saved to {partial_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1 COMPLETE")
    print("=" * 60)
    print(f"  Total merged questions:       {len(merged)}")
    print(f"  Duplicates removed:           {dupes}")
    print(f"  With wrong answers (ready):   {len(questions_with_wrong)}")
    print(f"  Needing wrong answers:        {len(questions_without_wrong)}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
