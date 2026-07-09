"""
(a) LettuceDetect sanity check.
Run: python test_a_lettucedetect.py
"""
from test_cases import CASES
from lettucedetect.models.inference import HallucinationDetector

detector = HallucinationDetector(
    method="transformer",
    model_path="KRLabsOrg/lettucedect-base-modernbert-en-v1"
)
print("LettuceDetect model loaded.\n")

for case in CASES:
    preds = detector.predict(
        context=[case["document"]],
        question=case["query"],
        answer=case["response"],
        output_format="spans"
    )
    flagged = preds if isinstance(preds, list) and preds and isinstance(preds[0], dict) else []
    is_hallucinated = len(flagged) > 0
    pred_label = "hallucinated" if is_hallucinated else "faithful"

    status = "PASS" if pred_label == case["expected"] else "FAIL"
    print(f"[{status}] {case['id']}: expected={case['expected']}  got={pred_label}")
    if flagged:
        print(f"    flagged spans: {[s.get('text','')[:80] for s in flagged]}")
