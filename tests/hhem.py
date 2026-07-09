"""
(g) Vectara HHEM-2.1-Open sanity check.
Run: python test_g_hhem.py
"""

import transformers

class AllTiedWeightsKeysDescriptor:
    def __get__(self, instance, owner):
        return {}
    def __set__(self, instance, value):
        pass

transformers.PreTrainedModel.all_tied_weights_keys = AllTiedWeightsKeysDescriptor()

import torch
from transformers import AutoModelForSequenceClassification
from test_cases import CASES

print("Loading Vectara HHEM-2.1-Open model...")
hhem_model = AutoModelForSequenceClassification.from_pretrained(
    "vectara/hallucination_evaluation_model", trust_remote_code=True
).to("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    hhem_model.t5.transformer.encoder.embed_tokens.weight.data.copy_(
        hhem_model.t5.transformer.shared.weight.data
    )

for case in CASES:
    hhem_score = float(hhem_model.predict([(case["document"], case["response"])])[0].item())
    pred_label = "faithful" if hhem_score >= 0.5 else "hallucinated"
    status = "PASS" if pred_label == case["expected"] else "FAIL"
    print(f"[{status}] {case['id']}: expected={case['expected']}  got={pred_label}  raw_score={hhem_score:.4f}")
