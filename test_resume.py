"""
途中保存・自動再開の実機試験ハーネス（本番 Pre-Training 前に RunPod で実行する）。

mscoco_train.json の先頭 2,000 件で Pre-Training（画像）と同じ引数の学習を短く回し、
「途中保存 → 強制終了 → 再実行で続きから → 完了後に checkpoint 削除」を確認する。
判定基準・設計の根拠は フェーズ3/進捗メモ/review-prompt-09-22.md への回答を参照。

使い方（すべて /workspace/MERAモデル で、venv 有効化済み・tmux 内）:

  python test_resume.py prep        # 試験用 JSON を作る（2,000 件）
  python test_resume.py r0          # 無中断の基準走（約16ステップ、約4分）
  python test_resume.py r2          # 試験走。進捗バーが 9〜13 の間に Ctrl+C で止める
  python test_resume.py r2          # もう一度 → checkpoint-7 から再開して完走するはず
  python test_resume.py fake        # 書きかけ checkpoint を偽造して再実行 → prune が消して即完了するはず
  python test_resume.py compare     # r0 と r2 の LR / loss / mm_projector.bin を比較し、cleanup を実行

  python test_resume.py lora-r0     # LoRA+DeepSpeed 経路の基準走（約125ステップ）
  python test_resume.py lora-r2     # 試験走。進捗バーが 25〜38 の間に Ctrl+C
  python test_resume.py lora-r2     # 再実行 → checkpoint-20 から再開
  python test_resume.py lora-compare

設定: save_steps=7（2,000件/128 = 16ステップなので 7・14 と最終16で保存。limit=1 で 7→14→16 と入れ替わる）
"""

import glob
import json
import os
import shutil
import subprocess
import sys

import config
from mera_train import (
    _cleanup_checkpoints,
    _ensure_pretrain_full_checkpoint,
    _merge_lora_and_save,
    _prune_incomplete_checkpoints,
)
from train_runner import REPO_DIR, train_image

TEST_JSON = "/workspace/data/intermediate/test_resume_2000.json"
TEST_ROOT = "/workspace/output/test_resume"
DEEPSPEED_CONFIG = os.path.join(REPO_DIR, "scripts", "zero2_cpu_offload.json")

PRETRAIN_KW = dict(
    model_path=config.LLM_BACKBONE,
    data_json=TEST_JSON,
    data_folder=config.IMAGE_DATA_FOLDER,
    vision_tower=config.VISION_TOWER,
    tune_mm_mlp_adapter=True,
    lora_enable=False,
    lr=1e-3,
    mm_projector_lr=1e-3,
    batch_size=8,
    grad_accum=16,
    num_gpus=config.NUM_GPUS,
    save_steps=7,
    logging_steps=1,
)

LORA_KW = dict(
    data_json=TEST_JSON,
    data_folder=config.IMAGE_DATA_FOLDER,
    vision_tower=config.VISION_TOWER,
    lora_enable=True,
    lora_r=128,
    lora_alpha=128,
    deepspeed_config=DEEPSPEED_CONFIG,
    num_gpus=config.NUM_GPUS,
    save_steps=20,
    logging_steps=1,
)


def _run_stage(output_dir: str, marker: str, **kw) -> None:
    """_prune → 学習 → （Ctrl+C なら中断を報告）。mera_train.py と同じ前後処理を明示的に呼ぶ。"""
    import accelerate
    print(f"[test] accelerate {accelerate.__version__}（0.24 以上なら SeedableRandomSampler で順序再現）")
    _prune_incomplete_checkpoints(output_dir)
    before = sorted(os.path.basename(p) for p in glob.glob(os.path.join(output_dir, "checkpoint-*")))
    print(f"[test] 開始前の checkpoint: {before or 'なし'}")
    try:
        train_image(output_dir=output_dir, **kw)
    except (KeyboardInterrupt, subprocess.CalledProcessError):
        after = sorted(os.path.basename(p) for p in glob.glob(os.path.join(output_dir, "checkpoint-*")))
        print(f"\n[test] 中断。残っている checkpoint: {after or 'なし'}")
        print("[test] 同じコマンドをもう一度実行すると再開する")
        sys.exit(1)
    after = sorted(os.path.basename(p) for p in glob.glob(os.path.join(output_dir, "checkpoint-*")))
    print(f"[test] 完走。完走直後の checkpoint（学習終了時保存の確認）: {after or 'なし'}")
    print(f"[test] root の完了マーカー {marker}: {'あり' if os.path.exists(os.path.join(output_dir, marker)) else 'なし'}")


def prep() -> None:
    with open(config.IMAGE_PRETRAIN_JSON) as f:
        data = json.load(f)
    subset = data[:2000]
    with open(TEST_JSON, "w") as f:
        json.dump(subset, f, ensure_ascii=False)
    print(f"[test] {len(subset)} 件を {TEST_JSON} に保存")


def fake(output_dir: str, marker: str, **kw) -> None:
    """完走後の最新 checkpoint をより大きい番号にコピーし trainer_state.json を消して「書きかけ」を偽造する。"""
    ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")), key=lambda p: int(p.split("-")[-1]))
    assert ckpts, "checkpoint-* が無い。r2 を完走させてから実行する"
    src = ckpts[-1]
    n = int(src.split("-")[-1])
    dst = os.path.join(output_dir, f"checkpoint-{n + 100}")
    shutil.copytree(src, dst)
    os.remove(os.path.join(dst, "trainer_state.json"))
    print(f"[test] 偽の書きかけを作成: {dst}（trainer_state.json 無し）")
    print(f"[test] 期待: prune が {os.path.basename(dst)} を消し、{os.path.basename(src)} から再開して即完了する")
    _run_stage(output_dir, marker, **kw)


def _log_history(output_dir: str) -> list[dict]:
    with open(os.path.join(output_dir, "trainer_state.json")) as f:
        return [e for e in json.load(f)["log_history"] if "loss" in e]


def compare(r0: str, r2: str, marker: str, resume_from: int) -> None:
    import torch

    h0 = {e["step"]: e for e in _log_history(r0)}
    h2 = {e["step"]: e for e in _log_history(r2)}
    print(f"[test] r0 steps: {min(h0)}..{max(h0)}  r2 steps: {min(h2)}..{max(h2)}")
    ok = True
    lr_bad, loss_bad = [], []
    for step in sorted(h0):
        if step <= resume_from or step not in h2:
            continue
        if h0[step]["learning_rate"] != h2[step]["learning_rate"]:
            lr_bad.append(step)
        l0, l2 = h0[step]["loss"], h2[step]["loss"]
        if abs(l0 - l2) > 1e-2 * max(abs(l0), 1e-6):
            loss_bad.append((step, l0, l2))
    print(f"[test] 学習率 完全一致: {'OK' if not lr_bad else 'NG ' + str(lr_bad)}")
    print(f"[test] loss 相対1e-2以内: {'OK' if not loss_bad else 'NG ' + str(loss_bad[:5])}")
    ok &= not lr_bad and not loss_bad

    if marker == "mm_projector.bin":
        w0 = torch.load(os.path.join(r0, marker), map_location="cpu")
        w2 = torch.load(os.path.join(r2, marker), map_location="cpu")
        same = all(torch.allclose(w0[k].float(), w2[k].float(), atol=1e-2) for k in w0)
        print(f"[test] mm_projector.bin allclose(atol=1e-2): {'OK' if same else 'NG'}")
        ok &= same

    _cleanup_checkpoints(r2, marker)
    left = glob.glob(os.path.join(r2, "checkpoint-*"))
    print(f"[test] cleanup 後の checkpoint-*: {'なし OK' if not left else 'NG ' + str(left)}")
    ok &= not left
    print(f"\n[test] 総合判定: {'合格' if ok else '不合格'}")


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    r0, r2 = f"{TEST_ROOT}/r0", f"{TEST_ROOT}/r2"
    lr0, lr2 = f"{TEST_ROOT}/lora_r0", f"{TEST_ROOT}/lora_r2"
    full = f"{TEST_ROOT}/r0_full"   # LoRA 試験の出発点（r0 をフルモデル化したもの）

    if mode == "prep":
        prep()
    elif mode == "r0":
        _run_stage(r0, "mm_projector.bin", **PRETRAIN_KW)
    elif mode == "r2":
        _run_stage(r2, "mm_projector.bin", **PRETRAIN_KW)
    elif mode == "fake":
        fake(r2, "mm_projector.bin", **PRETRAIN_KW)
    elif mode == "compare":
        compare(r0, r2, "mm_projector.bin", resume_from=7)
    elif mode in ("lora-r0", "lora-r2"):
        if not os.path.exists(os.path.join(full, "config.json")):
            shutil.copytree(r0, full, ignore=shutil.ignore_patterns("checkpoint-*", "runs"))
            _ensure_pretrain_full_checkpoint(config.LLM_BACKBONE, full, "image")
        _run_stage(lr0 if mode == "lora-r0" else lr2, "adapter_config.json", model_path=full, **LORA_KW)
    elif mode == "lora-compare":
        compare(lr0, lr2, "adapter_config.json", resume_from=20)
        _merge_lora_and_save(base_path=full, lora_path=lr2, save_path=f"{TEST_ROOT}/lora_r2_merged")
        print(f"[test] マージ成功: {TEST_ROOT}/lora_r2_merged")
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
