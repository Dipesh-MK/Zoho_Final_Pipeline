"""
(f) AlignScore sanity check.
Run: python test_f_alignscore.py
(requires ckpts/AlignScore-large.ckpt already downloaded by the main pipeline)
"""
import torch
import transformers
import torch.optim
transformers.AdamW = torch.optim.AdamW  # same shim the main pipeline uses

from alignscore import AlignScore
from test_cases import CASES

align_model = AlignScore(
    model='roberta-large',
    batch_size=16,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    ckpt_path='ckpts/AlignScore-large.ckpt',
    evaluation_mode='nli_sp'
)

for case in CASES:
    score = align_model.score(contexts=[case["document"]], claims=[case["response"]])[0]
    pred_label = "faithful" if score >= 0.5 else "hallucinated"
    status = "PASS" if pred_label == case["expected"] else "FAIL"
    print(f"[{status}] {case['id']}: expected={case['expected']}  got={pred_label}  raw_score={score:.4f}")
