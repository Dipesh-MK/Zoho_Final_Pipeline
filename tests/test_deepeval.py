"""
(c) DeepEval FaithfulnessMetric sanity check.
NOTE: uses the SAME judge model your pipeline currently points at (llama3.2).
If DeepEval also struggles with structured output like RAGAS did, consider
swapping to "qwen2.5-coder:7b" here too (same fix that worked for RAGAS).
Run: python test_c_deepeval.py
"""
from test_cases import CASES
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel

OLLAMA_BASE = "http://localhost:11434"
JUDGE_MODEL = "qwen2.5-coder:7b"

judge = OllamaModel(model=JUDGE_MODEL, base_url=OLLAMA_BASE)
metric = FaithfulnessMetric(threshold=0.5, include_reason=True, model=judge)

for case in CASES:
    test_case = LLMTestCase(
        input=case["query"],
        actual_output=case["response"],
        retrieval_context=[case["document"]],
    )
    try:
        metric.measure(test_case)
        pred_label = "faithful" if metric.score >= 0.5 else "hallucinated"
        status = "PASS" if pred_label == case["expected"] else "FAIL"
        print(f"[{status}] {case['id']}: expected={case['expected']}  got={pred_label}  score={metric.score:.3f}")
        print(f"    reason: {metric.reason}\n")
    except Exception as e:
        print(f"[ERROR] {case['id']}: {type(e).__name__}: {e}\n")
