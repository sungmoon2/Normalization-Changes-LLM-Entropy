"""
Token-Position 재실행: exp31과 동일 샘플/프롬프트/추출 정렬
==========================================================
근본 원인: run_phase1_token_position.py가 run_mmlu_entropy.py와 다른 샘플링 사용
  - exp31: np.random.choice(14042, 1000, seed=42) → 겹침 80/1000 (8%)
  - 구 token-pos: random.shuffle(all) → [:1000]

수정 내용 (run_phase1_token_position.py에 반영 완료):
  1. 샘플링: np.random.choice (exp31과 동일)
  2. 프롬프트: exp31과 동일한 user content
  3. 답변 추출: extract_mcq_answer 6패턴 (exp31과 동일)

이 스크립트: 3모델 순차 실행 + 검증
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from _paths import POT_DIR

BASE = Path(__file__).parent.parent
EXP_BASE = POT_DIR / "experiments"
OUTPUT_BASE = EXP_BASE / "44_GPT25_Experiments" / "phase2_gpu_tokenpos_v2_aligned"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable
SCRIPT = BASE / "scripts" / "run_phase1_token_position.py"

LOG_FILE = OUTPUT_BASE / "rerun_log.txt"

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def run_model(model_key, output_dir):
    """Run token-position experiment for one model."""
    log(f"Starting {model_key} MMLU token-position (aligned with exp31)")
    cmd = [
        PYTHON, str(SCRIPT),
        "--dataset", "mmlu",
        "--num_samples", "1000",
        "--model", model_key,
        "--output_dir", str(output_dir),
    ]
    log(f"  Command: {' '.join(cmd)}")

    t0 = time.time()
    result = subprocess.run(
        cmd,
        stdout=open(OUTPUT_BASE / f"subprocess_{model_key}.log", "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    )
    elapsed = time.time() - t0

    if result.returncode == 0:
        log(f"  {model_key} completed ({elapsed:.0f}s = {elapsed/3600:.2f}h)")
        # Verify results
        fr = output_dir / "final_results.json"
        if fr.exists():
            with open(fr) as f:
                d = json.load(f)
            log(f"  Results: n={d['n_total']}, acc={d['accuracy']}, errors={d['n_errors']}")
        else:
            log(f"  WARNING: final_results.json not found!")
    else:
        log(f"  FAILED (rc={result.returncode}, {elapsed:.0f}s)")

    return result.returncode == 0

def verify_alignment(model_key, output_dir):
    """Verify the rerun data matches exp31 samples."""
    checkpoint = output_dir / "data" / "checkpoint.json"
    if not checkpoint.exists():
        log(f"  VERIFY SKIP: no checkpoint for {model_key}")
        return

    with open(checkpoint) as f:
        new_data = json.load(f)

    # Load exp31 data for same model
    exp31_map = {
        "qwen": EXP_BASE / "31_MMLU_Domain_Extension" / "EXP_20260219_053638_mmlu_qwen" / "data" / "sample_results.json",
        "llama": EXP_BASE / "31_MMLU_Domain_Extension" / "EXP_20260219_171237_mmlu_llama" / "data" / "sample_results.json",
        "mistral": EXP_BASE / "31_MMLU_Domain_Extension" / "EXP_20260220_000610_mmlu_mistral" / "data" / "sample_results.json",
    }

    exp31_path = exp31_map.get(model_key)
    if not exp31_path or not exp31_path.exists():
        log(f"  VERIFY SKIP: exp31 data not found for {model_key}")
        return

    with open(exp31_path) as f:
        exp31_data = json.load(f)

    # Compare ground truth distributions
    from collections import Counter
    gt_new = Counter(s['ground_truth'] for s in new_data)
    gt_31 = Counter(s['ground_truth'] for s in exp31_data)

    if gt_new == gt_31:
        log(f"  VERIFY OK: {model_key} ground truth distribution matches exp31 exactly")
    else:
        log(f"  VERIFY FAIL: {model_key} ground truth mismatch!")
        log(f"    exp31: {dict(gt_31)}")
        log(f"    new:   {dict(gt_new)}")

    # Compare per-subject counts
    subj_new = Counter(s.get('subject', '?') for s in new_data)
    subj_31 = Counter(s.get('subject', '?') for s in exp31_data)
    diff = sum(1 for k in set(subj_new) | set(subj_31) if subj_new.get(k, 0) != subj_31.get(k, 0))
    log(f"  Subject distribution diffs: {diff}/{len(set(subj_new) | set(subj_31))}")

    # Compare accuracy
    corr_new = sum(1 for s in new_data if s.get('is_correct', False))
    corr_31 = sum(1 for s in exp31_data if s.get('is_correct', False))
    log(f"  Accuracy: new={corr_new}/{len(new_data)} ({corr_new/len(new_data)*100:.1f}%), exp31={corr_31}/{len(exp31_data)} ({corr_31/len(exp31_data)*100:.1f}%)")


if __name__ == "__main__":
    log("=" * 60)
    log("TOKEN-POSITION RERUN: ALIGNED WITH EXP31 SAMPLES")
    log(f"Output: {OUTPUT_BASE}")
    log("=" * 60)

    models = [
        ("qwen", OUTPUT_BASE / "qwen_mmlu_aligned"),
        ("llama", OUTPUT_BASE / "llama_mmlu_aligned"),
        ("mistral", OUTPUT_BASE / "mistral_mmlu_aligned"),
    ]

    all_ok = True
    for model_key, out_dir in models:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "data").mkdir(exist_ok=True)

        success = run_model(model_key, out_dir)
        if success:
            verify_alignment(model_key, out_dir)
        else:
            all_ok = False
            log(f"STOPPING: {model_key} failed")
            break

    if all_ok:
        log("")
        log("ALL 3 MODELS COMPLETED SUCCESSFULLY")
        log("Next: re-run EXP-2c Step0 incremental with aligned data")
    else:
        log("")
        log("PIPELINE INCOMPLETE - check logs")
