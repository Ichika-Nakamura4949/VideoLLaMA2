"""
途中保存・自動再開の実機試験ハーネス（本番 Pre-Training 前に RunPod で実行する）。

mscoco_train.json の先頭 2,000 件で Pre-Training（画像）と同じ引数の学習を短く回し、
「途中保存 → 強制終了 → 再実行で続きから → 完了後に checkpoint 削除」を確認する。
2,000件 ÷ batch8 = 250 micro-batch、250 // grad_accum16 = 15 optimizer step
（端数の10 micro-batchは処理されない。4.42.3の仕様）。保存は 7・14・15（終了時）。

判定基準・設計の根拠は フェーズ3/進捗メモ/review-prompt-09-22.md・review-prompt-09-22-v2.md を参照。

使い方（すべて /workspace/MERAモデル で、venv 有効化済み・tmux 内）:

  python test_resume.py prep        # 試験用 JSON を作る（2,000 件）
  python test_resume.py r0          # 無中断の基準走（15ステップ、約4分）
  python test_resume.py r2          # 試験走。進捗バーが 9〜13 の間に Ctrl+C で止める
  python test_resume.py r2          # もう一度 → checkpoint-7 から再開して完走するはず
  python test_resume.py fake        # 書きかけ checkpoint を偽造して再実行 → prune が消して即完了するはず
  python test_resume.py compare     # r0 と r2 の LR / loss / optimizer step / projector を比較し、cleanup

  python test_resume.py lora-r0     # LoRA+DeepSpeed 経路の基準走（125ステップ）
  python test_resume.py lora-r2     # 試験走。進捗バーが 25〜38 の間に Ctrl+C
  python test_resume.py lora-r2     # 再実行 → checkpoint-20 から再開
  python test_resume.py lora-compare   # LR/loss比較＋マージ実行＋projectorがマージに反映されたか検証

Ctrl+C で中断した直後は torchrun 配下のプロセスが後始末中なので、次のコマンドを打つ前に
`pgrep -f videollama2/train.py` が空になるのを確認する（ハーネス側でも待つが、長引く場合は手動確認）。
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time

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


def _wait_for_train_py_to_exit(timeout: int = 60) -> None:
    """Ctrl+C後、torchrun配下のtrain.pyが完全に終了するまで待つ。
    subprocess.runはSIGINTで子を0.25秒だけ待ってkillするため、
    train.py自身の後始末（DataLoader worker・DeepSpeedスレッド）は
    親プロセス消滅後も少し続く。これを待たずに次を実行するとGPUメモリが
    解放前でOOM、またはmaster_portが前のプロセスに握られたままになる。"""
    for _ in range(timeout):
        r = subprocess.run(["pgrep", "-f", "videollama2/train.py"], capture_output=True)
        if r.returncode != 0:
            return
        time.sleep(1)
    print("[test] 警告: train.py がまだ残っている可能性。`nvidia-smi`で確認し、"
          "GPUメモリが解放されていなければ手動でkillしてから次を実行する")


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
        _wait_for_train_py_to_exit()
        after = sorted(os.path.basename(p) for p in glob.glob(os.path.join(output_dir, "checkpoint-*")))
        print(f"\n[test] 中断。残っている checkpoint: {after or 'なし'}")
        print("[test] train.py の終了を確認できたら、同じコマンドをもう一度実行すると再開する")
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
    """完走後の最新 checkpoint をより大きい番号にコピーし trainer_state.json を消して「書きかけ」を偽造する。

    期待される動き：prune が偽物を消す → train.py が正常な checkpoint-N を拾って resume_from_checkpoint=True
    → epochs_trained = global_step // steps_per_epoch が既に1に達しているため学習ループは一度も回らず
    即座に「Training completed」で終了する → 新しい checkpoint は増えない（これは異常ではない）。
    """
    ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")), key=lambda p: int(p.split("-")[-1]))
    assert ckpts, "checkpoint-* が無い。r2 を完走させてから実行する"
    src = ckpts[-1]
    n = int(src.split("-")[-1])
    dst = os.path.join(output_dir, f"checkpoint-{n + 100}")
    shutil.copytree(src, dst)
    os.remove(os.path.join(dst, "trainer_state.json"))
    print(f"[test] 偽の書きかけを作成: {dst}（trainer_state.json 無し）")
    print(f"[test] 期待: prune が {os.path.basename(dst)} を消し、{os.path.basename(src)} から即終了する"
          f"（epochs_trained が既に1周分に達しているため学習ループは回らず、checkpoint は増えない）")
    _run_stage(output_dir, marker, **kw)


def _log_history(output_dir: str) -> list[dict]:
    with open(os.path.join(output_dir, "trainer_state.json")) as f:
        return [e for e in json.load(f)["log_history"] if "loss" in e]


def _optimizer_step(output_dir: str) -> set:
    """最新 checkpoint の optimizer.pt に記録された step 番号（Adam の内部カウンタ）を返す。
    optimizerが正しく復元されていれば最終step、初期化からやり直していれば小さい値になる。
    学習率・lossより直接的に「optimizer状態が本当に復元されたか」を見る決定的な指標。"""
    ckpts = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")), key=lambda p: int(p.split("-")[-1]))
    if not ckpts:
        return set()
    import torch
    opt = torch.load(os.path.join(ckpts[-1], "optimizer.pt"), map_location="cpu")
    return {int(s["step"]) for s in opt["state"].values() if "step" in s}


def compare(r0: str, r2: str, marker: str, resume_from: int, loose_loss: bool = False) -> None:
    import torch

    h0 = {e["step"]: e for e in _log_history(r0)}
    h2 = {e["step"]: e for e in _log_history(r2)}
    print(f"[test] r0 steps: {min(h0)}..{max(h0)}  r2 steps: {min(h2)}..{max(h2)}")
    ok = True

    # 中断前（1..resume_from）: 再開と無関係な非決定性の床。差があるのは正常
    pre_diffs = [abs(h0[s]["loss"] - h2[s]["loss"]) for s in range(1, resume_from + 1) if s in h0 and s in h2]
    if pre_diffs:
        print(f"[test] 再開と無関係な区間(step 1-{resume_from})のloss差（非決定性の床）: "
              f"max={max(pre_diffs):.4f}")

    lr_bad, loss_bad = [], []
    for step in sorted(h0):
        if step <= resume_from or step not in h2:
            continue
        if h0[step]["learning_rate"] != h2[step]["learning_rate"]:
            lr_bad.append(step)
        l0, l2 = h0[step]["loss"], h2[step]["loss"]
        # 再開直後は決定的に一致するはずだが、LoRA+DeepSpeedはbackwardの非決定性が
        # 蓄積するため、再開直後(+5ステップ)より先は緩い閾値にする
        tol = 5e-2 if (loose_loss and step > resume_from + 5) else 1e-2
        if abs(l0 - l2) > tol * max(abs(l0), 1e-6):
            loss_bad.append((step, l0, l2, tol))
    print(f"[test] 学習率 完全一致: {'OK' if not lr_bad else 'NG ' + str(lr_bad)}")
    print(f"[test] loss 一致: {'OK' if not loss_bad else 'NG ' + str(loss_bad[:5])}")
    ok &= not lr_bad and not loss_bad

    # optimizer state の直接確認（決定的）
    max_step = max(h0)
    opt0, opt2 = _optimizer_step(r0), _optimizer_step(r2)
    opt_ok = opt0 == {max_step} and opt2 == {max_step}
    print(f"[test] optimizer step: r0={opt0} r2={opt2}（期待: 両方とも {{{max_step}}}）: {'OK' if opt_ok else 'NG'}")
    ok &= opt_ok

    if marker == "mm_projector.bin":
        w0 = torch.load(os.path.join(r0, marker), map_location="cpu")
        w2 = torch.load(os.path.join(r2, marker), map_location="cpu")
        max_delta = max((w0[k].float() - w2[k].float()).abs().max().item() for k in w0)
        # projectorの1ステップ更新は要素あたり1e-3程度で、2e-3を大きく超えるのは
        # optimizerが初期化されてやり直しになった場合。ただしこれは補助指標で、
        # 主判定はlossとoptimizer stepの一致（上記）
        print(f"[test] mm_projector.bin max|Δ|={max_delta:.5f}（参考値。主判定はloss/optimizer stepを見る）")

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
        # ディレクトリ不在のときだけコピーする。_ensure自体は冪等なので毎回呼ぶ
        # （前回の_ensureがOOM/Ctrl+Cで途中終了した場合、config.jsonだけコピー済みで
        #  再実行をガードしてしまい、壊れたsafetensorsを読みに行く事故を防ぐ）
        if not os.path.isdir(full):
            shutil.copytree(r0, full, ignore=shutil.ignore_patterns("checkpoint-*", "runs"))
        _ensure_pretrain_full_checkpoint(config.LLM_BACKBONE, full, "image")
        _run_stage(lr0 if mode == "lora-r0" else lr2, "adapter_config.json", model_path=full, **LORA_KW)
    elif mode == "lora-compare":
        compare(lr0, lr2, "adapter_config.json", resume_from=20, loose_loss=True)
        merged_path = f"{TEST_ROOT}/lora_r2_merged"
        _merge_lora_and_save(base_path=full, lora_path=lr2, save_path=merged_path)
        print(f"[test] マージ成功: {merged_path}")

        # C-1: Phase1/2で学習したprojectorがマージ後モデルに反映されているかを直接検証。
        # 反映されていなければ、mera_train.pyの_merge_lora_and_saveのキー接頭辞の剥がしが
        # 機能していない（修正前バージョンではこのassertが落ちる）
        import torch
        from safetensors.torch import load_file
        extra = torch.load(os.path.join(lr2, "non_lora_trainables.bin"), map_location="cpu")
        merged = load_file(os.path.join(merged_path, "model.safetensors"))
        checked = 0
        for k, v in extra.items():
            kk = k
            if kk.startswith("base_model."):
                kk = kk[len("base_model."):]
            if kk.startswith("model.model.") and kk.replace("model.model.", "model.", 1) in merged:
                kk = kk.replace("model.model.", "model.", 1)
            assert kk in merged, f"マージ後モデルにキーが無い: {kk}（元: {k}）"
            assert torch.equal(merged[kk].float(), v.float()), f"マージ後にprojector等が反映されていない: {kk}"
            checked += 1
        print(f"[test] non_lora_trainables.bin の {checked} tensors がマージ後モデルに正しく反映されている: OK")
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main()
