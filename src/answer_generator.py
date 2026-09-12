"""
Optional final stage: summarizes an already-retrieved RetrievalResult using
a local Ollama model.

This is deliberately the LAST stage, not a replacement for retrieval:

    USER QUERY -> GRAPH RETRIEVAL -> RETRIEVED FACTS -> OLLAMA -> SUMMARY

The LLM is only ever shown the facts retrieval.py already found (see
build_prompt/format_facts_for_prompt below) - it is explicitly instructed
not to use outside medical knowledge, and the retrieved facts are always
available to the caller/UI independently of whether generation succeeds,
so nothing is lost if Ollama is unavailable (see AnswerResult.note).

Per the brief's medical-safety requirements, the prompt instructs the
model to: answer using only the retrieved facts, say so explicitly when
those facts don't fully answer the question, and avoid diagnostic or
prescriptive language. A lightweight keyword scan
(_contains_prescriptive_language) logs a warning if the model's output
looks prescriptive anyway - this is best-effort defensive logging, not a
guarantee, since a real safety boundary can't be enforced by string
matching alone.
"""

from pydantic import BaseModel

from src import config
from src.retrieval import RetrievalResult


class AnswerResult(BaseModel):
    query: str
    summary: str | None = None
    facts_used: int = 0
    ollama_available: bool = False
    model: str | None = None
    note: str = ""  # explains *why* summary is None, or "" on success


_SYSTEM_INSTRUCTIONS = """You are a medical information retrieval assistant. You summarize facts that were retrieved from a curated drug-interaction and drug-disease knowledge graph. You are NOT a diagnostic tool and must NEVER give clinical advice, dosing instructions, or tell the user what to do.

Rules you MUST follow:
1. Base your answer ONLY on the "Retrieved facts" listed below. Do not use outside medical knowledge, and do not invent facts that are not listed.
2. If the retrieved facts do not fully answer the question, say so explicitly rather than filling the gap with a guess.
3. Do not diagnose, recommend treatment, or state anything with clinical certainty (avoid phrases like "you should", "this will", "definitely"). Use neutral, informational language such as "the retrieved records indicate...".
4. Each fact is labeled [direct] (directly connected to the entity the user asked about) or [2-hop] (reached through one intermediate entity, given as extra context). Prioritize [direct] facts as the main answer; only mention [2-hop] facts briefly if they add clearly relevant context.
5. Keep the answer concise (3-5 sentences).
6. This is for an academic information-retrieval demonstration, not medical advice.
"""

_PRESCRIPTIVE_PHRASES = (
    "you should take", "i recommend", "you must", "diagnos", "prescribe",
    "definitely", "you have ", "take this",
)


def format_facts_for_prompt(result: RetrievalResult) -> str:
    """Numbered, human-readable list of retrieved facts - this exact text
    is what the model is told to restrict itself to.
    """
    lines = [
        f"{i}. [{'direct' if f.hop == 1 else '2-hop'}] {f.source_name} --[{f.relationship}]--> "
        f"{f.target_name} (evidence: {f.evidence})"
        for i, f in enumerate(result.facts, 1)
    ]
    return "\n".join(lines)


def build_prompt(query: str, result: RetrievalResult) -> str:
    matched = result.matched_entity.canonical_name if result.matched_entity else "unknown"
    return f"""{_SYSTEM_INSTRUCTIONS}
User question: "{query}"
Matched entity in the knowledge graph: {matched}

Retrieved facts:
{format_facts_for_prompt(result)}

Write a concise summary answering the user's question using ONLY the facts above.
"""


def _contains_prescriptive_language(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _PRESCRIPTIVE_PHRASES)


def generate_answer(query: str, result: RetrievalResult, model: str = config.OLLAMA_MODEL) -> AnswerResult:
    """Summarize `result` with Ollama. Never raises: any failure (Ollama
    package missing, service not running, model not pulled, empty
    response) degrades to `ollama_available=False` with an explanatory
    `note`, so the caller can always fall back to showing the raw
    retrieved facts (result.facts) instead of the summary.
    """
    if not result.facts:
        print("[ANSWER] No facts retrieved - skipping LLM call, nothing to summarize")
        return AnswerResult(
            query=query, facts_used=0, ollama_available=False,
            note="No relevant information was found in the knowledge graph for this query.",
        )

    try:
        import ollama
    except ImportError:
        print("[ANSWER] 'ollama' package not installed - showing retrieved facts only")
        return AnswerResult(
            query=query, facts_used=len(result.facts), ollama_available=False,
            note="Ollama is not installed; showing retrieved facts only.",
        )

    prompt = build_prompt(query, result)
    try:
        response = ollama.generate(model=model, prompt=prompt)
        summary = response["response"].strip()
    except Exception as exc:
        print(f"[ANSWER] Ollama call failed ({exc}) - showing retrieved facts only")
        return AnswerResult(
            query=query, facts_used=len(result.facts), ollama_available=False,
            note=f"Ollama is unavailable ({exc}); showing retrieved facts only.",
        )

    if not summary:
        print("[ANSWER] Ollama returned an empty response - showing retrieved facts only")
        return AnswerResult(
            query=query, facts_used=len(result.facts), ollama_available=False,
            note="Ollama returned an empty response; showing retrieved facts only.",
        )

    if _contains_prescriptive_language(summary):
        print(f"[ANSWER] WARNING: generated summary may contain prescriptive language: {summary[:120]!r}")

    print(f"[ANSWER] Generated a {len(summary)}-character summary using {len(result.facts)} retrieved facts")
    return AnswerResult(query=query, summary=summary, facts_used=len(result.facts), ollama_available=True, model=model)


if __name__ == "__main__":
    # Quick manual smoke test: `python -m src.answer_generator`
    # Ollama is not installed in the dev environment, so this exercises
    # (and demonstrates) the graceful-fallback path for real, not a mock.
    from src.retrieval import load_retrieval_engine, retrieve_relationships

    G, index = load_retrieval_engine()

    print("\n--- Prompt construction (pure string building, no Ollama needed) ---")
    result = retrieve_relationships("What diseases does Metformin treat?", G, index)
    print(build_prompt(result.query, result))

    print("\n--- generate_answer() with real (empty) Ollama installation ---")
    answer = generate_answer(result.query, result)
    print(answer.model_dump_json(indent=2))

    print("\n--- generate_answer() on a no-match query (should skip Ollama entirely) ---")
    empty_result = retrieve_relationships("xyzabc123", G, index)
    answer = generate_answer(empty_result.query, empty_result)
    print(answer.model_dump_json(indent=2))
