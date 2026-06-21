"""Generate one plausible wrong answer per validated question.

Uses the exact prompt from the spec. Temperature 0, no thinking.
One call per question, cached in Redis.
"""

from src.client import call_claude, MODEL

WRONG_ANSWER_PROMPT = """Generate a plausible but incorrect answer to this question.
Follow these rules based on the answer type:
- Person: swap with someone from the same field, nationality, or era
- Date/year: shift by 2–5 years in either direction
- Place/country/city: swap with a geographically or culturally adjacent one
- Number: shift by 10–20% in either direction
- Organization: swap with a similar organization in the same domain
- Other: generate the most plausible alternative in the same category
The wrong answer should seem reasonable to someone unfamiliar with the topic.
Return only the wrong answer, nothing else.
Question: {question}
Correct answer: {correct_answer}"""


def generate_wrong_answer(question: str, correct_answer: str) -> str:
    """Generate a single plausible wrong answer for a question."""
    prompt = WRONG_ANSWER_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
    )

    result = call_claude(
        messages=[{"role": "user", "content": prompt}],
        cache_key_parts=[question, "wrong_answer", MODEL],
        max_tokens=1000,
        extended_thinking=False,
    )

    # Clean the response — handle cases where model self-corrects
    raw = result["answer"].strip()

    # If multi-line (self-correction), take the last non-empty line
    if "\n" in raw:
        lines = [l.strip().strip("*").strip() for l in raw.split("\n") if l.strip()]
        raw = lines[-1] if lines else raw

    return raw.rstrip(".")


def generate_all_wrong_answers(questions: list[dict]) -> list[dict]:
    """Generate wrong answers for all questions. Returns augmented list."""
    results = []

    for i, q in enumerate(questions):
        wrong = generate_wrong_answer(q["question"], q["answer"])
        q_with_wrong = {**q, "wrong_answer": wrong}
        results.append(q_with_wrong)

        line = (
            f"  [{i+1}/{len(questions)}] "
            f"Q: {q['question'][:50]}... | "
            f"Correct: {q['answer']} | "
            f"Wrong: {wrong}"
        )
        print(
            line.encode("ascii", errors="replace").decode(),
            flush=True,
        )

    print(f"\n  Generated {len(results)} wrong answers.")
    return results
