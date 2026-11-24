import json
import os
from typing import Any, Dict, List

PROMPT_TEMPLATE = """You are a legal contract analyst.
Task: Answer the QUESTION using only the provided CONTEXT from a contract/policy.
Return STRICT JSON (no markdown) with keys:
- answered: boolean
- answer: string (concise, cite section names if present)
- evidence: array of EXACT quotes from context (each <= 300 chars)
- risk_notes: string (risks/ambiguities if answered, else why missing)

QUESTION:
{question}

CONTEXT:
{context}

If the context does not contain enough information, set answered=false and keep evidence empty.
"""

def _try_openai():
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    except Exception:
        return None

def ask_llm(question: str, context: str) -> Dict[str, Any]:
    llm = _try_openai()
    if llm is None:
        # Fallback heuristic: mark answered if any keyword overlap is high
        q_terms = set(t.lower() for t in question.split() if len(t) > 4)
        ctx = context.lower()
        hit = sum(1 for t in q_terms if t in ctx)
        answered = hit >= max(2, int(0.15 * len(q_terms) + 1))
        return {
            "answered": bool(answered),
            "answer": "Heuristic fallback (no LLM configured).",
            "evidence": [],
            "risk_notes": "Configure OPENAI_API_KEY to enable clause extraction."
        }

    prompt = PROMPT_TEMPLATE.format(question=question, context=context)
    resp = llm.invoke(prompt).content

    # Strict JSON extraction
    try:
        return json.loads(resp)
    except Exception:
        # Try to recover if model wrapped JSON with text
        start = resp.find("{")
        end = resp.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(resp[start:end+1])
            except Exception:
                pass
        return {
            "answered": False,
            "answer": "",
            "evidence": [],
            "risk_notes": f"LLM returned non-JSON output; could not parse. Raw: {resp[:300]}"
        }
