"""
(b) Lynx-8B (PatronusAI via Ollama) sanity check.
Uses the exact prompt template from the main pipeline.
Run: python test_b_lynx.py
"""
import json as json_mod
import requests
from test_cases import CASES

OLLAMA_BASE = "http://localhost:11434"
LYNX_MODEL = "hf.co/PatronusAI/Llama-3-Patronus-Lynx-8B-Instruct-Q4_K_M-GGUF:latest"

LYNX_PROMPT_TEMPLATE = (
    "Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer "
    "and determine whether it is faithful to the contents of the DOCUMENT. "
    "The ANSWER must not offer new information beyond the context provided in the DOCUMENT. "
    "The ANSWER also must not contradict information provided in the DOCUMENT. "
    "Output your final verdict by strictly following this format: \"PASS\" if the answer is "
    "faithful to the DOCUMENT and \"FAIL\" if the answer is not faithful to the DOCUMENT. "
    "Show your reasoning.\n\n"
    "-- QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION): {question}\n"
    "-- DOCUMENT: {context}\n"
    "-- ANSWER: {answer}\n\n"
    "Your output should be in JSON FORMAT with the keys \"REASONING\" and \"SCORE\": "
    '{{"REASONING": <your reasoning as bullet points>, "SCORE": <your final score>}}'
)

for case in CASES:
    prompt = LYNX_PROMPT_TEMPLATE.format(
        question=case["query"], context=case["document"], answer=case["response"]
    )
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": LYNX_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_gpu": 33, "num_predict": 500},
        },
        timeout=600,
    )
    content = resp.json().get("message", {}).get("content", "")

    score_val = None
    try:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        parsed = json_mod.loads(cleaned)
        score_val = parsed.get("SCORE", "")
    except Exception:
        upper = content.upper()
        score_val = "PASS" if "PASS" in upper else ("FAIL" if "FAIL" in upper else None)

    pred_label = "hallucinated" if str(score_val).upper() == "FAIL" else (
        "faithful" if str(score_val).upper() == "PASS" else "error")

    status = "PASS" if pred_label == case["expected"] else "FAIL"
    print(f"[{status}] {case['id']}: expected={case['expected']}  got={pred_label}")
    print(f"    raw content: {content[:300]!r}\n")
