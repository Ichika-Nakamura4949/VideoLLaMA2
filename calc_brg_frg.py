"""
BRG・FRG を計算するマスタースクリプト。

評価は以下の4回で完結する:
  ① θ_1          × image_eval.json  → 画像スコアA  (BRGの分母)
  ② θ_{2,vanilla} × audio_eval.json  → 音声スコアB  (FRGの分母)
  ③ CKPT_FINAL   × image_eval.json  → 画像スコアC  (BRGの分子)
  ④ CKPT_FINAL   × audio_eval.json  → 音声スコアD  (FRGの分子)

  BRG = C / A   (1.0 = 完全保持、低いほど忘却)
  FRG = D / B   (1.0 = naive FTと同等、低いほどMERAのコストが大きい)

使い方:
  python calc_brg_frg.py

チェックポイントパスは config.py から自動取得する。
各評価の予測結果は /workspace/output/eval/ 以下に JSONL で保存される。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from eval_vqa import run as eval_run
import argparse


EVAL_OUT_DIR  = "/workspace/output/eval"
IMAGE_EVAL    = "/workspace/data/image_eval.json"
AUDIO_EVAL    = "/workspace/data/audio_eval.json"
DATA_FOLDER   = "/workspace/data"


def make_args(model: str, eval_json: str, output: str, data_folder: str = DATA_FOLDER) -> argparse.Namespace:
    return argparse.Namespace(
        model       = model,
        eval_json   = eval_json,
        data_folder = data_folder,
        output      = output,
        limit       = None,
    )


if __name__ == "__main__":
    os.makedirs(EVAL_OUT_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("① θ_1 × image_eval.json  (BRG分母)")
    print("="*60)
    score_A = eval_run(make_args(
        model     = config.CKPT_PHASE1,
        eval_json = IMAGE_EVAL,
        output    = f"{EVAL_OUT_DIR}/phase1_image.jsonl",
    ))

    print("\n" + "="*60)
    print("② θ_{2,vanilla} × audio_eval.json  (FRG分母)")
    print("="*60)
    score_B = eval_run(make_args(
        model     = config.CKPT_AUDIO_VANILLA,
        eval_json = AUDIO_EVAL,
        output    = f"{EVAL_OUT_DIR}/vanilla_audio.jsonl",
    ))

    print("\n" + "="*60)
    print("③ CKPT_FINAL × image_eval.json  (BRG分子)")
    print("="*60)
    score_C = eval_run(make_args(
        model     = config.CKPT_FINAL,
        eval_json = IMAGE_EVAL,
        output    = f"{EVAL_OUT_DIR}/final_image.jsonl",
    ))

    print("\n" + "="*60)
    print("④ CKPT_FINAL × audio_eval.json  (FRG分子)")
    print("="*60)
    score_D = eval_run(make_args(
        model     = config.CKPT_FINAL,
        eval_json = AUDIO_EVAL,
        output    = f"{EVAL_OUT_DIR}/final_audio.jsonl",
    ))

    brg = score_C / score_A if score_A > 0 else float("nan")
    frg = score_D / score_B if score_B > 0 else float("nan")

    result = {
        "score_A_phase1_image":   score_A,
        "score_B_vanilla_audio":  score_B,
        "score_C_final_image":    score_C,
        "score_D_final_audio":    score_D,
        "BRG": brg,
        "FRG": frg,
    }

    result_path = f"{EVAL_OUT_DIR}/brg_frg.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "="*60)
    print("BRG / FRG 計算結果")
    print("="*60)
    print(f"  A: θ_1         × 画像  = {score_A:.4f}")
    print(f"  B: θ_{{2,vanilla}} × 音声  = {score_B:.4f}")
    print(f"  C: CKPT_FINAL  × 画像  = {score_C:.4f}")
    print(f"  D: CKPT_FINAL  × 音声  = {score_D:.4f}")
    print(f"")
    print(f"  BRG = C / A = {brg:.4f}  (1.0 = 画像性能を完全保持)")
    print(f"  FRG = D / B = {frg:.4f}  (1.0 = naive FT と同等の音声性能)")
    print(f"")
    print(f"  結果を保存: {result_path}")
    print("="*60)
