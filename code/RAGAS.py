"""
RAGAS.py - standalone RAGAS-style Faithfulness evaluator.
"""
import argparse
import csv
import json
import os
import sys
import time
import traceback

import requests as req_lib

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
OLLAMA_BASE = "http://127.0.0.1:11434"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
USE_GROQ = bool(GROQ_API_KEY)
DEBUG = os.environ.get("DEBUG", "") == "1"

DATA_FILES = [
    "data/ragtruth.jsonl",
    "data/ragbench.jsonl",
    "data/delucionqa.jsonl",
    "data/mock.jsonl",
]

OUT_CSV = "results/ragas_only.csv"
OUT_DETAIL = "results/ragas_only_claims_detail.jsonl"

GROQ_RPM = int(os.environ.get("GROQ_RPM", "18"))
GROQ_TPM = int(os.environ.get("GROQ_TPM", "7000"))

_groq_min_interval = 60.0 / GROQ_RPM
_groq_last_call_time = 0.0
_groq_call_timestamps = []          
_groq_token_events = []             


def _estimate_tokens(prompt, max_tokens):
    return len(prompt) // 4 + max_tokens


def _rate_limit_wait(estimated_tokens):
    global _groq_last_call_time
    now = time.time()

    since_last = now - _groq_last_call_time
    if since_last < _groq_min_interval:
        time.sleep(_groq_min_interval - since_last)
        now = time.time()

    while _groq_call_timestamps and now - _groq_call_timestamps[0] > 60:
        _groq_call_timestamps.pop(0)
        
    if len(_groq_call_timestamps) >= GROQ_RPM:
        sleep_for = 60 - (now - _groq_call_timestamps[0]) + 0.5
        if sleep_for > 0:
            print(f"    [Groq RPM limiter] Max requests reached, sleeping {sleep_for:.1f}s...")
            time.sleep(sleep_for)
        now = time.time()

    while _groq_token_events and now - _groq_token_events[0][0] > 60:
        _groq_token_events.pop(0)
        
    tokens_used = sum(t for _, t in _groq_token_events)
    if tokens_used + estimated_tokens > GROQ_TPM:
        if _groq_token_events:
            sleep_for = 60 - (now - _groq_token_events[0][0]) + 0.5
            if sleep_for > 0:
                print(f"    [Groq TPM limiter] ~{tokens_used} tokens used in last 60s, sleeping {sleep_for:.1f}s...")
                time.sleep(sleep_for)
            now = time.time()
            while _groq_token_events and now - _groq_token_events[0][0] > 60:
                _groq_token_events.pop(0)

    _groq_last_call_time = now
    _groq_call_timestamps.append(now)


def _record_token_usage(actual_tokens):
    _groq_token_events.append((time.time(), actual_tokens))


def groq_chat_completion(prompt, model=None, temperature=0, max_tokens=1200, max_retries=5, timeout=90):
    model = model or GROQ_MODEL
    last_err = None
    for attempt in range(max_retries):
        _rate_limit_wait(_estimate_tokens(prompt, max_tokens))
        try:
            resp = req_lib.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=timeout,
            )
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(2 ** attempt * 2, 30)
                print(f"    [Groq 429] API forced limit, waiting {wait:.1f}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            actual_tokens = usage.get("total_tokens", _estimate_tokens(prompt, max_tokens))
            _record_token_usage(actual_tokens)
            
            finish_reason = data["choices"][0].get("finish_reason", "")
            if finish_reason == "length" and DEBUG:
                print(f"    [Groq] WARNING: response truncated by max_tokens={max_tokens}.")
                
            return content
        except req_lib.exceptions.RequestException as e:
            last_err = e
            wait = min(2 ** attempt, 20)
            print(f"    [Groq Error] Network/timeout error: {e}, retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Groq API failed after {max_retries} attempts: {last_err}")


def ollama_faithfulness_call(prompt, model="qwen2.5-coder:7b", timeout=120, max_retries=3):
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = req_lib.post(
                f"{OLLAMA_BASE}/api/chat",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0},
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "")
        except Exception as e:
            last_err = e
            time.sleep(3)
    raise RuntimeError(f"Ollama call failed after {max_retries} attempts: {last_err}")


def llm_judge_call(prompt):
    if USE_GROQ:
        return groq_chat_completion(prompt)
    return ollama_faithfulness_call(prompt)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------
RAGAS_CLAIM_EXTRACTION_PROMPT = """Given a QUESTION and an ANSWER, break the ANSWER down into a list of simple, atomic, standalone factual claims (statements).

Rules:
- Each claim must be a single verifiable statement.
- Respond with ONLY a JSON array of strings and nothing else, e.g. ["claim one", "claim two"].

QUESTION: {question}
ANSWER: {answer}"""

RAGAS_VERIFICATION_PROMPT = """You will be given a CONTEXT and a numbered list of CLAIMS. For each claim, decide whether it can be directly inferred from or is explicitly supported by the CONTEXT alone.

QUESTION: {question}
CONTEXT: {context}

CLAIMS:
{numbered_claims}

Respond with ONLY a valid JSON array of objects, one object per claim, in claim-number order, and nothing else. Every object must use the string key "i" for the claim number index, "verdict" (0 or 1), and "reason" (max 8 words).
Example format:
[[{{"i": 1, "verdict": 1, "reason": "supported"}}, ...]]"""


def _extract_json_array(text):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        if len(parts) > 1:
            cleaned = parts[1]
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    start = cleaned.find("[")
    if start == -1:
        try:
            obj_start = cleaned.find("{")
            obj_end = cleaned.rfind("}")
            if obj_start != -1 and obj_end != -1:
                loaded_obj = json.loads(cleaned[obj_start:obj_end+1])
                for k, v in loaded_obj.items():
                    if isinstance(v, list):
                        return v
        except:
            pass
        return []

    end = cleaned.rfind("]")
    if end != -1 and end > start:
        try:
            res = json.loads(cleaned[start:end + 1])
            if isinstance(res, list):
                return res
        except json.JSONDecodeError:
            pass

    # Salvage pass for truncated responses
    elements = []
    body = cleaned[start + 1:]
    depth = 0
    in_string = False
    escape = False
    elem_start = None
    
    for i, ch in enumerate(body):
        if elem_start is None and ch.strip() in ("", ","):
            continue
        if elem_start is None:
            elem_start = i
        
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        
        if ch == '"':
            in_string = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0 and elem_start is not None:
                chunk = body[elem_start:i + 1]
                try:
                    parsed_chunk = json.loads(chunk)
                    elements.append(parsed_chunk)
                except json.JSONDecodeError:
                    pass
                elem_start = None
        elif ch == "," and depth == 0:
            elem_start = None

    if DEBUG and elements:
        print(f"    [JSON repair] response was truncated; salvaged {len(elements)} items.")
        
    return elements


def compute_ragas_faithfulness(question, answer, context, row_id=""):
    extraction_prompt = RAGAS_CLAIM_EXTRACTION_PROMPT.format(question=question, answer=answer)
    extraction_raw = llm_judge_call(extraction_prompt)
        
    raw_claims = _extract_json_array(extraction_raw)
    claims = []
    if isinstance(raw_claims, list):
        for c in raw_claims:
            if isinstance(c, str):
                claims.append(c)
            elif isinstance(c, dict):
                val = c.get("claim") or c.get("statement") or list(c.values())[0]
                claims.append(str(val))

    if not claims:
        return 1.0, False, []

    numbered_claims = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    verification_prompt = RAGAS_VERIFICATION_PROMPT.format(
        question=question,
        context=context[:1200],
        numbered_claims=numbered_claims,
    )
    
    verify_max_tokens = min(120 * len(claims) + 200, 3500)
    verification_raw = groq_chat_completion(verification_prompt, max_tokens=verify_max_tokens) if USE_GROQ else ollama_faithfulness_call(verification_prompt)
        
    raw_verdicts = _extract_json_array(verification_raw)

    # BULLETPROOF FLATTENER: deeply flatten any nested lists the model spits out
    flat_verdicts = []
    if isinstance(raw_verdicts, list):
        for item in raw_verdicts:
            if isinstance(item, list):
                flat_verdicts.extend(item)
            else:
                flat_verdicts.append(item)
    
    verdict_by_index = {}
    for v in flat_verdicts:
        if isinstance(v, dict):
            idx_val = v.get("i") or v.get("claim_number") or v.get("index")
            if idx_val is not None:
                try:
                    verdict_by_index[int(idx_val)] = v
                except (TypeError, ValueError):
                    pass

    claims_detail = []
    supported = 0
    missing = 0
    has_any_hallucination = False

    print(f"    [MATH {row_id}] Checking {len(claims)} claims...")

    for i, c in enumerate(claims):
        verdict_obj = verdict_by_index.get(i + 1, {})
        raw_v = verdict_obj.get("verdict", 0) # default to 0 if missing
        
        # Aggressive conversion to integer 1 or 0
        try:
            verdict = int(raw_v)
        except (TypeError, ValueError):
            if str(raw_v).strip().lower() in ("1", "true", "yes", "supported"):
                verdict = 1
            else:
                verdict = 0
        
        if (i + 1) not in verdict_by_index:
            missing += 1
            verdict = 0 # Punish missing verdicts by treating them as hallucinations
            
        reason = str(verdict_obj.get("reason", ""))[:300]
        claims_detail.append({"claim": c, "verdict": verdict, "reason": reason})
        supported += verdict

        print(f"      -> Claim {i+1}: verdict={verdict} (Raw: {raw_v})")

        if verdict == 0:
            has_any_hallucination = True

    score = supported / len(claims)
    is_hallucinated = has_any_hallucination 
    
    print(f"    [STRICT LOGIC {row_id}] Any 0s found? {has_any_hallucination} --> IS_HALLUCINATED: {is_hallucinated}\n")
    
    return score, is_hallucinated, claims_detail

def main():
    print("\n🚀 RUNNING THE NEW STRICT MODE RAGAS CODE!\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Only process N records")
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    records = []
    for path in DATA_FILES:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line.strip()))

    if args.limit:
        records = records[:args.limit]

    print(f"Total records loaded: {len(records)}")
    if not records:
        sys.exit(1)

    tool_rows = []
    detail_records = []

    for idx, rec in enumerate(records):
        print(f"  [RAGAS {idx+1}/{len(records)}] Processing {rec['id']}...", flush=True)
        t0 = time.time()
        pred_label = "error"
        flagged_span_text = ""
        try:
            score, is_hallucinated, claims_detail = compute_ragas_faithfulness(
                question=rec["query"],
                answer=rec["response"],
                context=rec["document"],
                row_id=rec["id"],
            )
            pred_label = "hallucinated" if is_hallucinated else "faithful"
            flagged_span_text = f"score={score:.3f}; claims={len(claims_detail)}"

            detail_records.append({
                "id": rec["id"],
                "dataset": rec["dataset"],
                "question": rec["query"],
                "faithfulness_score": round(score, 4),
                "num_claims": len(claims_detail),
                "claims": claims_detail,
            })
        except Exception as e:
            print(f"  RAGAS execution crash trace:")
            traceback.print_exc()
            pred_label = "error"
            flagged_span_text = str(e)[:300]

        latency = time.time() - t0
        row = {
            "dataset":            rec["dataset"],
            "id":                 rec["id"],
            "tool":               "RAGAS",
            "predicted_label":    pred_label,
            "ground_truth_label": "hallucinated" if rec["label_hallucinated"] else "faithful",
            "flagged_span_text":  flagged_span_text,
            "latency_seconds":    round(latency, 4),
        }
        tool_rows.append(row)

    with open(OUT_DETAIL, "w", encoding="utf-8") as f:
        for d in detail_records:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    fieldnames = ["dataset", "id", "tool", "predicted_label", "ground_truth_label", "flagged_span_text", "latency_seconds"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tool_rows)

    print(f"\nFirst 5 rows:")
    for r in tool_rows[:5]:
        print(f"  [{r['id']}] pred={r['predicted_label']}  gt={r['ground_truth_label']}")
        
    scored = [r for r in tool_rows if r["predicted_label"] in ("faithful", "hallucinated")]
    n_error = len(tool_rows) - len(scored)
    if scored:
        correct = sum(1 for r in scored if r["predicted_label"] == r["ground_truth_label"])
        acc = correct / len(scored)
        print(f"\nAccuracy: {acc:.4f}  ({len(scored)}/{len(tool_rows)} scored, {n_error} errored)")


if __name__ == "__main__":
    main()