"""
Master Runner Script: run_all.py
Executes the entire evaluation pipeline sequentially:
  1. setup_and_verify.py
  2. load_datasets.py
  3. relevance_scoring.py
  4. hallucination_detection.py
  5. semantic_similarity_eval.py
  6. ragas_eval.py
"""
import os
import sys
import subprocess

SCRIPTS = [
    "setup_and_verify.py",
    "load_datasets.py",
    "relevance_scoring.py",
    "hallucination_detection.py",
    "semantic_similarity_eval.py",
    "ragas_eval.py",
]

print("="*80)
print("STARTING FULL EVALUATION PIPELINE RUN")
print("="*80)

for script in SCRIPTS:
    if not os.path.exists(script):
        print(f"Error: {script} not found in workspace root.")
        sys.exit(1)
        
    print(f"\n>>> Running {script}...")
    # Run using the local virtual env python
    python_exe = os.path.join(".venv", "Scripts", "python.exe")
    if not os.path.exists(python_exe):
        python_exe = "python"
        
    res = subprocess.run([python_exe, script], capture_output=False)
    if res.returncode != 0:
        print(f"\n[FAIL] {script} failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[SUCCESS] {script} completed successfully.")

print("\n" + "="*80)
print("FULL PIPELINE EXECUTED SUCCESSFULLY")
print("="*80)
