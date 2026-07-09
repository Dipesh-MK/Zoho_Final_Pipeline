"""
phoenix.py - standalone Arize Phoenix-style Hallucination evaluator.

This isolates the Phoenix prompt and runs it manually row-by-row through
our custom rate limiter, completely bypassing LiteLLM (which was causing
the instant 429 throttling errors in the main pipeline).

Fix applied: openai/gpt-oss-120b is a reasoning model. By default
(reasoning_effort="medium") it spends tokens on an internal reasoning pass
before emitting the final answer, and those reasoning tokens count against
max_tokens even though Groq returns them in a separate `reasoning` field.
With the original max_tokens=50, the model could burn its whole budget on
reasoning and leave nothing for the actual "factual"/"hallucinated" word,
producing "Unrecognized output from model" errors. Fixed by setting
reasoning_effort="low" (this is a simple one-word classification - it
doesn't need deep reasoning) and raising max_tokens for margin.
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

# Reasoning effort only applies to GPT-OSS models on Groq; harmless to omit
# for others, but we only send it when it's actually supported.
IS_GPT_OSS = "gpt-oss" in GROQ_MODEL.lower()
GROQ_REASONING_EFFORT = os.environ.get("GROQ_REASONING_EFFORT", "low")

DATA_FILES = [
    "data/ragtruth.jsonl",
    "data/ragbench.jsonl",
    "data/delucionqa.jsonl",
    "data/mock.jsonl",
]

OUT_CSV = "results/phoenix_only.csv"
OUT_DETAIL = "results/phoenix_only_detail.jsonl"

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


def groq_chat_completion(prompt, model=None, temperature=0, max_tokens=150, max_retries=5, timeout=90):
    model = model or GROQ_MODEL
    last_err = None
    for attempt in range(max_retries):
        _rate_limit_wait(_estimate_tokens(prompt, max_tokens))
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if IS_GPT_OSS:
                # Simple one-word classification doesn't need deep reasoning;
                # "low" keeps reasoning-token spend small so it doesn't eat
                # the max_tokens budget before the model emits its verdict.
                payload["reasoning_effort"] = GROQ_REASONING_EFFORT

            resp = req_lib.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json=payload,
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
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
            reasoning = message.get("reasoning") or ""
            finish_reason = data["choices"][0].get("finish_reason", "")
            usage = data.get("usage", {})
            actual_tokens = usage.get("total_tokens", _estimate_tokens(prompt, max_tokens))
            _record_token_usage(actual_tokens)

            if DEBUG:
                print(f"    [Groq] usage: {usage}  finish_reason: {finish_reason}")
                if reasoning:
                    print(f"    [Groq] reasoning ({len(reasoning)} chars): {reasoning[:200]!r}")

            if not content.strip() and finish_reason == "length":
                # Budget was fully spent on reasoning with nothing left for
                # content - surface this clearly instead of a mystery empty string.
                print(f"    [Groq] WARNING: empty content, finish_reason=length "
                      f"(max_tokens={max_tokens} likely consumed entirely by reasoning). "
                      f"Consider raising max_tokens or lowering GROQ_REASONING_EFFORT further.")

            return content, reasoning
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
                    "options": {"temperature": 0, "num_predict": 50},
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", ""), ""
        except Exception as e:
            last_err = e
            time.sleep(3)
    raise RuntimeError(f"Ollama call failed after {max_retries} attempts: {last_err}")


# --------------------------------------------------------------------------
# Phoenix Prompt & Logic
# --------------------------------------------------------------------------
PHOENIX_PROMPT = """In this task, you will be given a reference text and a conclusion. Your task is to determine if the conclusion is fully supported by the reference text. If it is, output 'factual'. If it contains any information that is not supported by the reference text, output 'hallucinated'.

Reference text: {reference}

Conclusion: {conclusion}

Answer with exactly one word ('factual' or 'hallucinated'), no explanation:"""


def _parse_verdict(text):
    """Search from the end of the text first - if the model leaked any
    reasoning/preamble into content, the actual verdict is almost always the
    last word, not the first mention (e.g. "...not hallucinated... factual")."""
    cleaned = text.strip().lower()
    last_factual = cleaned.rfind("factual")
    last_hallucinated = cleaned.rfind("hallucinated")
    if last_factual == -1 and last_hallucinated == -1:
        return None
    # "hallucinated" contains no "factual" substring and vice versa, so
    # whichever keyword occurs LATER in the text is the actual verdict.
    if last_hallucinated > last_factual:
        return "hallucinated"
    return "faithful"


def compute_phoenix_eval(reference, conclusion, row_id=""):
    prompt = PHOENIX_PROMPT.format(reference=reference, conclusion=conclusion)

    if USE_GROQ:
        content, reasoning = groq_chat_completion(prompt, max_tokens=150)
    else:
        content, reasoning = ollama_faithfulness_call(prompt)

    if DEBUG:
        print(f"    [DEBUG {row_id}] content: {content!r}")

    verdict = _parse_verdict(content)
    if verdict is None and reasoning:
        # Fall back to the reasoning trace if content came back empty/unparseable.
        if DEBUG:
            print(f"    [DEBUG {row_id}] content unparseable, falling back to reasoning trace")
        verdict = _parse_verdict(reasoning)

    if verdict is None:
        raise ValueError(f"Unrecognized output from model: content={content!r} reasoning={reasoning[:200]!r}")

    return verdict, content or reasoning


# --------------------------------------------------------------------------
# Main Loop
# --------------------------------------------------------------------------
def main():
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

    if USE_GROQ:
        try:
            _check = req_lib.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={"model": GROQ_MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
                timeout=15,
            )
            if _check.status_code == 401:
                print("FATAL: GROQ_API_KEY is set but was rejected (401 Unauthorized).")
                sys.exit(1)
            if _check.status_code == 404:
                print(f"FATAL: model '{GROQ_MODEL}' not found on Groq. Set $env:GROQ_MODEL to a valid model.")
                sys.exit(1)
            _check.raise_for_status()
        except req_lib.exceptions.RequestException as e:
            print(f"FATAL: could not reach Groq API to verify key: {e}")
            sys.exit(1)
        print(f"Using Groq judge model: {GROQ_MODEL} (reasoning_effort={GROQ_REASONING_EFFORT if IS_GPT_OSS else 'n/a'})")
    else:
        print("Using local Ollama judge.")

    tool_rows = []
    detail_records = []

    for idx, rec in enumerate(records):
        print(f"  [Phoenix {idx+1}/{len(records)}] Processing {rec['id']}...", flush=True)
        t0 = time.time()
        pred_label = "error"
        flagged_span_text = ""

        try:
            pred_label, raw_output = compute_phoenix_eval(
                reference=rec["document"],
                conclusion=rec["response"],
                row_id=rec["id"]
            )
            flagged_span_text = raw_output[:300]

            detail_records.append({
                "id": rec["id"],
                "dataset": rec["dataset"],
                "predicted_label": pred_label,
                "raw_output": raw_output
            })

        except Exception as e:
            if DEBUG:
                print(f"  Phoenix execution crash trace:")
                traceback.print_exc()
            print(f"  Phoenix error on {rec['id']}: {e}")
            pred_label = "error"
            flagged_span_text = str(e)[:300]

        latency = time.time() - t0

        if latency < 0.3:
            time.sleep(0.3 - latency)
            latency = 0.301

        row = {
            "dataset":            rec["dataset"],
            "id":                 rec["id"],
            "tool":               "Phoenix",
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

    print(f"\nSaved {len(tool_rows)} rows to {OUT_CSV}")
    print(f"First 5 rows:")
    for r in tool_rows[:5]:
        print(f"  [{r['id']}] pred={r['predicted_label']:12s} gt={r['ground_truth_label']:12s} lat={r['latency_seconds']:.2f}s")

    scored = [r for r in tool_rows if r["predicted_label"] in ("faithful", "hallucinated")]
    n_error = len(tool_rows) - len(scored)
    if scored:
        correct = sum(1 for r in scored if r["predicted_label"] == r["ground_truth_label"])
        acc = correct / len(scored)
        print(f"\nAccuracy: {acc:.4f}  ({len(scored)}/{len(tool_rows)} scored, {n_error} errored)")


if __name__ == "__main__":
    main()