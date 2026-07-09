"""
(h) RAGAS Faithfulness sanity check.
Includes the vertexai stub + format="json" + qwen2.5-coder:7b judge,
matching your current fixed pipeline config.
Run: python test_h_ragas.py
"""
import sys, types, asyncio

# stub for langchain_community's missing vertexai submodule
_fake_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI: pass
_fake_vertexai.ChatVertexAI = ChatVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _fake_vertexai

from langchain_ollama import ChatOllama
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness
from ragas.dataset_schema import SingleTurnSample
from test_cases import CASES

OLLAMA_BASE = "http://localhost:11434"
JUDGE_MODEL = "qwen2.5-coder:7b"

chat = ChatOllama(model=JUDGE_MODEL, base_url=OLLAMA_BASE, temperature=0, format="json")
judge = LangchainLLMWrapper(chat)
metric = Faithfulness(llm=judge)

for case in CASES:
    sample = SingleTurnSample(
        user_input=case["query"],
        response=case["response"],
        retrieved_contexts=[case["document"]],
    )
    try:
        score = asyncio.run(metric.single_turn_ascore(sample))
        pred_label = "faithful" if score >= 0.5 else "hallucinated"
        status = "PASS" if pred_label == case["expected"] else "FAIL"
        print(f"[{status}] {case['id']}: expected={case['expected']}  got={pred_label}  raw_score={score:.4f}")
    except Exception as e:
        print(f"[ERROR] {case['id']}: {type(e).__name__}: {e}")
