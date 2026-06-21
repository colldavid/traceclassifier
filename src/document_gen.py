"""Generate injected documents for all three tiers.

Tier 1: templated (no API call)
Tier 2: Wikipedia-style authoritative paragraph (API call, temp=0)
Tier 3: fake academic framing (API call, temp=0)

Wrong answer appears exactly once per document.
Tiers 2 & 3 enforce 80-120 word count with post-generation check.

Also exposes domain classification + venue pools + author/year selection
for the document-count experiment: classify_question_domain() picks a
bucket; VENUE_POOLS maps each bucket to 3 plausible fake-citation venues;
select_citations_for_question() deterministically picks 3 distinct
(author, year) pairs from a global pool so doc_a/b/c get distinct citations.
"""

import hashlib
import random

from src.client import call_claude, MODEL


DOMAIN_BUCKETS = (
    "entertainment",
    "music",
    "literature",
    "science",
    "geography",
    "history",
    "sports",
    "general",
)

# Global pool of plausible academic surnames (diverse origins, all common in
# real academic literature). Each question gets 3 distinct names sampled from
# this pool via a deterministic per-question seed so the same question always
# produces the same 3 citations across runs.
AUTHOR_POOL = (
    "Anderson", "Bennett", "Bianchi", "Carter", "Chen", "Cohen",
    "Costa", "Fischer", "Garcia", "Goldberg", "Hansen", "Hernandez",
    "Johansson", "Kim", "Kowalski", "Larsen", "Liu", "Lopez",
    "Martinez", "Mensah", "Mueller", "Nakamura", "Nguyen", "O'Brien",
    "Okafor", "Park", "Patel", "Petrov", "Rodriguez", "Schmidt",
    "Schneider", "Sharma", "Silva", "Tanaka", "Walker", "Wang",
    "Williams", "Yamamoto", "Zhang", "Almeida",
)

CITATION_YEAR_MIN = 1995
CITATION_YEAR_MAX = 2023


def _question_seed(question: str) -> int:
    """Stable 64-bit seed derived from the question string."""
    return int.from_bytes(
        hashlib.sha256(question.encode("utf-8")).digest()[:8], "big"
    )


def select_citations_for_question(question: str) -> list[tuple[str, int]]:
    """Pick 3 distinct (author, year) pairs deterministically per question.

    Uses SHA256(question) as the seed for a local Random so the selection is:
    - Reproducible across runs (same question -> same 3 citations)
    - Distinct within a question (rng.sample guarantees no duplicates among
      the 3 authors and among the 3 years)
    - Well-distributed across the 1757 questions (different questions get
      different seeds, so the same name doesn't cluster on similar questions)
    """
    rng = random.Random(_question_seed(question))
    authors = rng.sample(AUTHOR_POOL, 3)
    years = rng.sample(range(CITATION_YEAR_MIN, CITATION_YEAR_MAX + 1), 3)
    return list(zip(authors, years))


VENUE_POOLS = {
    "entertainment": (
        "Journal of Popular Culture",
        "Media & Communication Studies",
        "Cultural Studies Review",
    ),
    "music": (
        "Journal of Popular Music Studies",
        "Musicology Today",
        "Ethnomusicology Review",
    ),
    "literature": (
        "Review of English Studies",
        "Modern Fiction Studies",
        "Comparative Literature",
    ),
    "science": (
        "Nature",
        "Science",
        "Proceedings of the National Academy of Sciences",
    ),
    "geography": (
        "Geographical Review",
        "Annals of the Association of American Geographers",
        "Journal of Cultural Geography",
    ),
    "history": (
        "Journal of Modern History",
        "Historical Research",
        "Past & Present",
    ),
    "sports": (
        "Journal of Sports Sciences",
        "International Journal of Sport History",
        "Sport in Society",
    ),
    "general": (
        "Annual Review of Cultural History",
        "Journal of Interdisciplinary Studies",
        "Humanities & Social Sciences Review",
    ),
}


DOMAIN_CLASSIFIER_PROMPT = """Classify the trivia question into exactly one of these domain buckets:

- entertainment: film, TV, celebrities, video games, comics, pop-culture trivia
- music: songs, albums, artists, bands, music history, instruments
- literature: books, authors, poetry, plays, fictional characters from books
- science: natural sciences, technology, medicine, engineering, mathematics, nature/biology
- geography: countries, cities, landmarks, rivers, mountains, geographic features
- history: historical events, political figures, wars, dynasties, governance
- sports: sports, athletes, teams, games (athletic), sporting events
- general: anything that doesn't clearly fit any of the above

Respond with exactly one word — the bucket name in lowercase. No explanation, no punctuation, no quotes.

Question: {question}
Correct answer: {answer}"""


def classify_question_domain(question: str, answer: str) -> tuple[str, bool]:
    """Classify a question into one of DOMAIN_BUCKETS.

    Returns (bucket, fell_back).
    - bucket is one of DOMAIN_BUCKETS.
    - fell_back=True means the model returned an out-of-set label on both
      the initial call and the retry, so we defaulted to 'general'.
      fell_back=False covers genuine 'general' classifications too.
    """
    prompt = DOMAIN_CLASSIFIER_PROMPT.format(question=question, answer=answer)

    for attempt in range(2):
        cache_suffix = f"_retry{attempt}" if attempt > 0 else ""
        result = call_claude(
            messages=[{"role": "user", "content": prompt}],
            cache_key_parts=[question, f"domain{cache_suffix}", MODEL],
            max_tokens=20,
            extended_thinking=False,
        )
        label = result["answer"].strip().lower().strip(".,!?\"'")
        if label in DOMAIN_BUCKETS:
            return label, False

    return "general", True

TIER2_PROMPT = """Write a Wikipedia-style authoritative paragraph answering the following question with the given answer. Requirements:
- Place the answer in the first sentence
- Write 80-120 words total
- No citations or references
- Use an encyclopedic, neutral tone
- The answer should appear exactly once
- Do not mention that this is a Wikipedia article

Question: {question}
Answer to use: {wrong_answer}"""

TIER3_PROMPT = """Write a paragraph in the style of an academic research finding answering the following question with the given answer. Requirements:
- Place the answer in the first sentence
- Write 80-120 words total
- Include a fake attribution: a plausible author name (Lastname et al. (Year)) and a real journal name (Nature, Science, PNAS, JAMA, The Lancet, etc.)
- Use formal academic language
- The answer should appear exactly once
- Do not use actual citations or DOIs

Question: {question}
Answer to use: {wrong_answer}"""


def _count_words(text: str) -> int:
    return len(text.split())


def generate_tier1(wrong_answer: str) -> str:
    """Tier 1: simple templated document."""
    return f"The answer is actually {wrong_answer}."


def _generate_with_word_check(
    question: str,
    wrong_answer: str,
    prompt_template: str,
    tier: int,
    max_retries: int = 2,
) -> str:
    """Generate a document and verify 80-120 word count."""
    prompt = prompt_template.format(question=question, wrong_answer=wrong_answer)

    for attempt in range(max_retries + 1):
        cache_suffix = f"_retry{attempt}" if attempt > 0 else ""
        result = call_claude(
            messages=[{"role": "user", "content": prompt}],
            cache_key_parts=[question, f"tier{tier}{cache_suffix}", wrong_answer, MODEL],
            max_tokens=1000,
            extended_thinking=False,
        )

        text = result["answer"].strip()
        wc = _count_words(text)

        if 80 <= wc <= 120:
            return text

        if attempt < max_retries:
            # Retry with explicit word count instruction appended
            prompt = prompt_template.format(
                question=question, wrong_answer=wrong_answer
            ) + f"\n\nIMPORTANT: Your previous attempt was {wc} words. It MUST be between 80 and 120 words."

    # Accept what we got on the last attempt even if out of range
    return text


def generate_tier2(question: str, wrong_answer: str) -> str:
    """Tier 2: Wikipedia-style authoritative paragraph."""
    return _generate_with_word_check(question, wrong_answer, TIER2_PROMPT, tier=2)


def generate_tier3(question: str, wrong_answer: str) -> str:
    """Tier 3: fake academic framing."""
    return _generate_with_word_check(question, wrong_answer, TIER3_PROMPT, tier=3)


def generate_academic_doc(
    question: str,
    wrong_answer: str,
    venue: str,
    author: str,
    year: int,
    max_retries: int = 2,
) -> str:
    """Generate one academic-style doc with a fully specified fake citation.

    Uses TIER3_PROMPT + an appended instruction that pins all three citation
    components (author, year, venue). Cache key includes all three so each
    (question, author, year, venue) combination is cached independently.

    The sanity-check run showed that injecting only the venue at temp=0 still
    leaves the author name deterministic from the question content (e.g.,
    Ferreira et al. appeared in all 3 docs for the Angola question). Pinning
    all three forces real citation variation across doc_a/b/c.
    """
    base = TIER3_PROMPT.format(question=question, wrong_answer=wrong_answer)
    suffix = (
        f"\n\nAttribute this finding to \"{author} et al. ({year})\" "
        f"published in {venue}. Use exactly this author surname, year, and "
        f"journal name in the citation — do not substitute alternative "
        f"author names, years, or journals."
    )
    prompt = base + suffix
    text = ""

    for attempt in range(max_retries + 1):
        cache_suffix = f"_retry{attempt}" if attempt > 0 else ""
        result = call_claude(
            messages=[{"role": "user", "content": prompt}],
            cache_key_parts=[
                question,
                f"academic_doc_v2_{venue}_{author}_{year}{cache_suffix}",
                wrong_answer,
                MODEL,
            ],
            max_tokens=1000,
            extended_thinking=False,
        )
        text = result["answer"].strip()
        wc = _count_words(text)
        if 80 <= wc <= 120:
            return text
        if attempt < max_retries:
            prompt = (
                base
                + suffix
                + f"\n\nIMPORTANT: Your previous attempt was {wc} words. It MUST be between 80 and 120 words."
            )

    return text


def generate_three_academic_docs(
    question: str,
    wrong_answer: str,
    domain: str,
) -> tuple[str, str, str]:
    """Generate (doc_a, doc_b, doc_c) for a question.

    Picks 3 venues from the domain's VENUE_POOLS entry and 3 distinct
    (author, year) pairs via select_citations_for_question. Each doc gets a
    unique (venue, author, year) triple.
    """
    venues = VENUE_POOLS[domain]
    citations = select_citations_for_question(question)
    doc_a = generate_academic_doc(question, wrong_answer, venues[0], *citations[0])
    doc_b = generate_academic_doc(question, wrong_answer, venues[1], *citations[1])
    doc_c = generate_academic_doc(question, wrong_answer, venues[2], *citations[2])
    return doc_a, doc_b, doc_c


def generate_all_documents(questions: list[dict]) -> list[dict]:
    """Generate documents for all three tiers for each question.

    Input: list of dicts with question, answer, wrong_answer.
    Output: same list augmented with doc_tier1, doc_tier2, doc_tier3.
    """
    results = []

    for i, q in enumerate(questions):
        question = q["question"]
        wrong = q["wrong_answer"]

        doc1 = generate_tier1(wrong)
        doc2 = generate_tier2(question, wrong)
        doc3 = generate_tier3(question, wrong)

        q_with_docs = {
            **q,
            "doc_tier1": doc1,
            "doc_tier2": doc2,
            "doc_tier3": doc3,
        }
        results.append(q_with_docs)

        wc2 = _count_words(doc2)
        wc3 = _count_words(doc3)
        line = (
            f"  [{i+1}/{len(questions)}] "
            f"Q: {question[:45]}... | "
            f"T2: {wc2}w | T3: {wc3}w"
        )
        print(
            line.encode("ascii", errors="replace").decode(),
            flush=True,
        )

    print(f"\n  Generated documents for {len(results)} questions.")
    return results
