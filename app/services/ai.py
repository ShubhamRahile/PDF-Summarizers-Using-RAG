import json
import urllib.request
import urllib.error
from flask import current_app


def ask_ollama(prompt):
    payload = json.dumps({"model": current_app.config["OLLAMA_MODEL"], "prompt": prompt, "stream": False}).encode()
    request = urllib.request.Request(current_app.config["OLLAMA_URL"].rstrip("/") + "/api/generate", data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.loads(response.read().decode()).get("response", "").strip()
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def grounded_answer(question, evidence):
    if not evidence:
        return "Insufficient evidence: the indexed PDFs do not contain enough information to answer this question."
    joined = "\n\n".join(f"[{r['title']}, page {r['page']}] {r['text']}" for r in evidence)
    prompt = f"""You are a document analyst. Answer only from the evidence below. If it does not support an answer, say exactly that there is insufficient evidence. Separate confirmed facts from reasonable inferences and cite document title/page.
Question: {question}
Evidence:
{joined}
Provide a concise, useful answer."""
    answer = ask_ollama(prompt)
    if answer:
        return answer
    snippets = " ".join(r["text"][:300] for r in evidence[:3])
    return f"Evidence-grounded fallback (Ollama unavailable): {snippets} [Sources: " + ", ".join(f"{r['title']}, p.{r['page']}" for r in evidence) + "]"


def summary(doc, detail=False):
    evidence = [{"title": doc["title"], "page": 1, "text": doc["content"][:6000]}]
    question = "Give a structured summary with purpose, key findings, facts, recommendations, and conclusion." if detail else "Summarize this document in two concise sentences."
    return grounded_answer(question, evidence)
