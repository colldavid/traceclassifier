"""Feature extraction from extended-thinking traces.

Each function takes a thinking trace string and returns a numeric value.
All features are designed to be interpretable — they correspond to reasoning
behaviors a human reader would associate with resistance or capitulation.

Features fall into four groups:
  LENGTH     — how much the model thought
  DOUBT      — language signaling uncertainty about the documents
  CORRECTION — language signaling self-correction or reconsideration
  SCRUTINY   — language signaling the model is checking facts vs. trusting docs
"""

import re


# ---------------------------------------------------------------------------
# Marker lists
# ---------------------------------------------------------------------------

_DOUBT_MARKERS = [
    "however", "but wait", "but actually", "although", "nevertheless",
    "on the other hand", "i'm not sure", "i am not sure", "uncertain",
    "not certain", "i doubt", "questionable", "seems wrong", "seems incorrect",
    "might be wrong", "could be wrong", "this doesn't seem right",
    "this seems off", "let me reconsider", "wait,", "hmm,", "hmm.",
    "hold on", "pause", "actually,", "actually i", "wait —", "wait—",
]

_CORRECTION_MARKERS = [
    "actually the correct", "the correct answer is", "the real answer",
    "in fact,", "in reality,", "historically,", "actually,", "but actually",
    "i recall", "i know that", "from my knowledge", "i believe the answer is",
    "the answer should be", "that's incorrect", "that is incorrect",
    "the document is wrong", "the document appears to be wrong",
    "the document claims", "the documents claim", "the documents state",
    "appears to contradict", "contradicts", "inconsistent with",
    "this conflicts", "conflict with", "i need to correct",
    "let me reconsider", "reconsidering",
]

_SCRUTINY_MARKERS = [
    "the document states", "the document claims", "the document says",
    "according to the document", "per the document", "the paper claims",
    "the study claims", "supposedly", "allegedly", "purportedly",
    "the source says", "cited in", "as cited", "the citation",
    "verify", "double-check", "double check", "fact-check", "fact check",
    "cross-reference", "cross reference", "i should check",
    "let me check", "i know from", "from my training",
]

_TRUST_MARKERS = [
    "the document confirms", "the document establishes", "clearly states",
    "as stated in", "as shown in", "as indicated", "the documents show",
    "all three documents", "all documents", "consistently", "confirmed by",
    "supported by", "evidence shows", "research shows", "studies show",
    "the research indicates", "scholars agree", "experts agree",
]


def _count_markers(text: str, markers: list[str]) -> int:
    t = text.lower()
    return sum(1 for m in markers if m in t)


def _count_pattern(text: str, pattern: str, flags: int = re.IGNORECASE) -> int:
    return len(re.findall(pattern, text, flags))


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def f_thinking_length(trace: str) -> int:
    return len(trace)


def f_thinking_word_count(trace: str) -> int:
    return len(trace.split())


def f_doubt_count(trace: str) -> int:
    return _count_markers(trace, _DOUBT_MARKERS)


def f_correction_count(trace: str) -> int:
    return _count_markers(trace, _CORRECTION_MARKERS)


def f_scrutiny_count(trace: str) -> int:
    return _count_markers(trace, _SCRUTINY_MARKERS)


def f_trust_count(trace: str) -> int:
    return _count_markers(trace, _TRUST_MARKERS)


def f_question_marks(trace: str) -> int:
    return trace.count("?")


def f_exclamation_marks(trace: str) -> int:
    return trace.count("!")


def f_sentence_count(trace: str) -> int:
    return len(re.findall(r"[.!?]+", trace))


def f_doubt_density(trace: str) -> float:
    words = len(trace.split())
    if words == 0:
        return 0.0
    return f_doubt_count(trace) / words * 100


def f_correction_density(trace: str) -> float:
    words = len(trace.split())
    if words == 0:
        return 0.0
    return f_correction_count(trace) / words * 100


def f_scrutiny_vs_trust(trace: str) -> float:
    s = f_scrutiny_count(trace)
    t = f_trust_count(trace)
    total = s + t
    if total == 0:
        return 0.5
    return s / total


def f_has_any_doubt(trace: str) -> int:
    return int(f_doubt_count(trace) > 0)


def f_has_any_correction(trace: str) -> int:
    return int(f_correction_count(trace) > 0)


def f_has_any_scrutiny(trace: str) -> int:
    return int(f_scrutiny_count(trace) > 0)


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "thinking_length",
    "thinking_word_count",
    "doubt_count",
    "correction_count",
    "scrutiny_count",
    "trust_count",
    "question_marks",
    "sentence_count",
    "doubt_density",
    "correction_density",
    "scrutiny_vs_trust",
    "has_any_doubt",
    "has_any_correction",
    "has_any_scrutiny",
]

_EXTRACTORS = [
    f_thinking_length,
    f_thinking_word_count,
    f_doubt_count,
    f_correction_count,
    f_scrutiny_count,
    f_trust_count,
    f_question_marks,
    f_sentence_count,
    f_doubt_density,
    f_correction_density,
    f_scrutiny_vs_trust,
    f_has_any_doubt,
    f_has_any_correction,
    f_has_any_scrutiny,
]


def extract(trace: str) -> list[float]:
    """Return a feature vector (list of floats) for one thinking trace."""
    return [fn(trace) for fn in _EXTRACTORS]


def extract_with_meta(row: dict) -> dict:
    """Extract features from a pipeline result row, including condition."""
    trace = row.get("thinking_trace", "")
    vec = extract(trace)
    feats = dict(zip(FEATURE_NAMES, vec))
    feats["condition"] = row.get("condition", 0)
    feats["n_documents"] = row.get("n_documents", 0)

    t = trace.lower()
    correct = row.get("correct_answer", "")
    wrong = row.get("wrong_answer", "")
    aliases = row.get("aliases", [])

    correct_hit = correct.strip().lower() in t if correct.strip() else False
    if not correct_hit:
        for a in aliases or []:
            if a.strip() and a.strip().lower() in t:
                correct_hit = True
                break
    feats["correct_in_thinking"] = int(correct_hit)
    feats["wrong_in_thinking"] = int(wrong.strip().lower() in t) if wrong.strip() else 0

    return feats
