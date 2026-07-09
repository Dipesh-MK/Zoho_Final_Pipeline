"""
(d) Arize Phoenix HallucinationEvaluator sanity check.
Run: python test_d_phoenix.py
This prints the RAW result_df row so you can see exactly what label/score
field is being parsed -- useful since this section already had brittle
column-name-guessing logic in the main pipeline.
"""
import os as _os
import pandas as pd
from test_cases import CASES

OLLAMA_BASE = "http://localhost:11434"
_os.environ["OLLAMA_API_BASE"] = OLLAMA_BASE

from phoenix.evals import LLM, evaluate_dataframe, bind_evaluator
from phoenix.evals.metrics import HallucinationEvaluator

eval_model = LLM(provider="litellm", model="ollama/llama3.2")
hallucination_evaluator = HallucinationEvaluator(eval_model)

df = pd.DataFrame([
    {"id": c["id"], "input": c["query"], "output": c["response"], "context": c["document"]}
    for c in CASES
])

bound = bind_evaluator(hallucination_evaluator, {"output": "output", "context": "context", "input": "input"})
res_df = evaluate_dataframe(dataframe=df, evaluators=[bound])

print("Columns returned:", list(res_df.columns))
result_cols = [c for c in res_df.columns if c.endswith("_score") and "hallucination" in c.lower()]

for i, (_, row) in enumerate(res_df.iterrows()):
    case = CASES[i]
    print(f"\n--- {case['id']} (expected={case['expected']}) ---")
    print("RAW ROW:", row.to_dict())
    for c in result_cols:
        print(f"  field '{c}': {row.get(c)}")
