"""
MERA 順次学習パイプライン（RunPod 上で実行するスクリプト）

論文 §4 に基づく 2 モダリティ（画像→音声）の MERA パイプライン:

  Phase1  : 画像モダリティで LoRA 学習（MSCOCO + OK-VQA）
             → CKPT_PHASE1_LORA/（LoRA adapter）
             → LoRA マージ → CKPT_PHASE1/（θ_1 完全モデル）
             ※ 評価①: eval_vqa.py --model CKPT_PHASE1 --eval-json image_eval.json
                → 画像スコアA（BRGの分母）

  Phase2  : θ_1 を出発点に音声モダリティを素直に LoRA FT（AudioCaps + Clotho-AQA）
             → CKPT_AUDIO_VANILLA_LORA/（LoRA adapter）
             → LoRA マージ → CKPT_AUDIO_VANILLA/（θ_{2,vanilla} 完全モデル）
             ※ 評価②: eval_vqa.py --model CKPT_AUDIO_VANILLA --eval-json audio_eval.json
                → 音声スコアB（FRGの分母）

  Step1   : θ_1 と θ_{2,vanilla} の LLM 層を式 (4.1) でマージ
             θ_{2,merged} = 1/2 × θ_1 + 1/2 × θ_{2,vanilla}
             → CKPT_MERGED

  Step2a  : CKPT_MERGED から画像コネクタを画像 replay で再整合
             LLM 凍結・LoRA 不要
             → CKPT_IMG_REALIGNED/mm_projector.bin のみ

  Step2b  : CKPT_MERGED から音声コネクタを音声 replay で再整合
             LLM 凍結・LoRA 不要
             → CKPT_AUD_REALIGNED/mm_projector_a.bin のみ

  Assemble: CKPT_MERGED に両コネクタを注入して最終モデルを組み立てる
             → CKPT_FINAL
             ※ 評価③④: eval_vqa.py --model CKPT_FINAL で画像・音声スコア
                → 画像スコアC（BRG分子）・音声スコアD（FRG分子）

  BRG/FRG計算: calc_brg_frg.py を実行
                BRG = C / A（1.0 = 画像性能を完全保持）
                FRG = D / B（1.0 = naive FT と同等の音声性能）

前提:
  - このファイルは MERAモデル/ 直下（videollama2/ の親ディレクトリ）から実行する
  - config.py / mera.py / train_runner.py が同じディレクトリにあること
  - 実行コマンド例: python mera_train.py
  - 学習完了後: python calc_brg_frg.py

再開ロジック:
  各ステップの出力ディレクトリが既に存在する場合はそのステップをスキップする。
  Phase1 LoRA 学習済み（CKPT_PHASE1_LORA）・未マージの場合はマージのみ再実行する。
"""

import os
import sys
import torch

import config
from mera import mera_step1, get_step2_training_flags, assemble_final_model
from train_runner import REPO_DIR, train_image, train_audio

# DeepSpeed 設定（Phase1/2 の LoRA 学習で使用）
DEEPSPEED_CONFIG = os.path.join(REPO_DIR, "scripts", "zero2_cpu_offload.json")


# ── 再開ロジック補助関数 ───────────────────────────────────────────────────────

def _already_done(path: str, name: str) -> bool:
    """
    path のディレクトリが既に存在する場合はスキップし True を返す。
    クラッシュ後の再開時に完了済みステップを飛ばすために使う。
    """
    if os.path.exists(path):
        print(f"[skip] {name} → 出力済みのためスキップ: {path}")
        return True
    return False


# ── LoRA マージ補助関数 ────────────────────────────────────────────────────────

def _merge_lora_and_save(base_path: str, lora_path: str, save_path: str) -> None:
    """
    LoRA チェックポイントを base モデルにマージして完全モデルとして保存する。

    videollama2/model/__init__.py の既存ロジック（PeftModel.merge_and_unload）を
    mera_train.py から直接呼ぶラッパー。

    引数:
        base_path : LoRA 学習の出発点（BASE_MODEL または CKPT_PHASE1）
        lora_path : train.py が出力した LoRA チェックポイントのディレクトリ
        save_path : マージ済み完全モデルの保存先
    """
    sys.path.insert(0, REPO_DIR)
    from peft import PeftModel
    from videollama2.model import Videollama2Qwen2ForCausalLM

    print(f"\n{'='*60}")
    print(f"[lora_merge] base  : {base_path}")
    print(f"[lora_merge] lora  : {lora_path}")
    print(f"[lora_merge] save  : {save_path}")
    print(f"{'='*60}\n")

    model = Videollama2Qwen2ForCausalLM.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # コネクタなど LoRA 対象外の更新済み重みを適用
    non_lora_path = os.path.join(lora_path, "non_lora_trainables.bin")
    if os.path.exists(non_lora_path):
        extra = torch.load(non_lora_path, map_location="cpu")
        model.load_state_dict(extra, strict=False)

    # LoRA adapter をマージして完全モデルに変換
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"[lora_merge] 保存完了: {save_path}")

    del model
    torch.cuda.empty_cache()


# ── Step1 補助関数 ────────────────────────────────────────────────────────────

def _apply_mera_step1_and_save(
    audio_vanilla_dir: str,
    prev_dir: str,
    save_dir: str,
    modality_index: int,
) -> None:
    """
    audio_vanilla_dir（θ_{i,vanilla}）を読み込み、
    prev_dir（θ_{i-1}）と MERA Step1 マージを適用して save_dir に保存する。

    引数:
        audio_vanilla_dir : θ_{i,vanilla}（新モダリティを素直に FT した結果）
        prev_dir          : θ_{i-1}（前段ステージの最終モデル）
        save_dir          : マージ済みモデルの保存先
        modality_index    : モダリティ番号 i（画像→音声の場合は i=2）
    """
    sys.path.insert(0, REPO_DIR)
    from videollama2.model import Videollama2Qwen2ForCausalLM

    print(f"\n{'='*60}")
    print(f"[MERA Step1] θ_{{i,vanilla}} 読み込み: {audio_vanilla_dir}")
    print(f"[MERA Step1] θ_{{i-1}}       読み込み: {prev_dir}")
    print(f"[MERA Step1] モダリティ番号 i={modality_index}")
    print(f"{'='*60}\n")

    model = Videollama2Qwen2ForCausalLM.from_pretrained(
        audio_vanilla_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # θ_{i,merged} = (i-1)/i × θ_{i-1} + 1/i × θ_{i,vanilla}（in place）
    mera_step1(model, prev_path=prev_dir, i=modality_index)

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print(f"[MERA Step1] マージ済みモデルを保存: {save_dir}")

    del model
    torch.cuda.empty_cache()


# ── Assemble 補助関数 ─────────────────────────────────────────────────────────

def _assemble_final_model() -> None:
    """
    CKPT_MERGED（フルモデル）に Step2a・Step2b で更新された両コネクタを注入して
    CKPT_FINAL に最終モデルを保存する。
    """
    sys.path.insert(0, REPO_DIR)

    img_bin = os.path.join(config.CKPT_IMG_REALIGNED, "mm_projector.bin")
    aud_bin = os.path.join(config.CKPT_AUD_REALIGNED, "mm_projector_a.bin")

    assemble_final_model(
        merged_path      = config.CKPT_MERGED,
        img_connector_bin= img_bin,
        aud_connector_bin= aud_bin,
        save_path        = config.CKPT_FINAL,
    )


# ── パイプライン ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: 画像モダリティで LoRA 学習 → LoRA マージ → θ_1
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Phase 1: 画像学習（MSCOCO + OK-VQA）→ θ_1")
    print("="*60)
    if not _already_done(config.CKPT_PHASE1, "Phase1（マージ済みモデル）"):
        if not _already_done(config.CKPT_PHASE1_LORA, "Phase1（LoRA 学習）"):
            train_image(
                model_path      = config.BASE_MODEL,
                data_json       = config.IMAGE_DATA_JSON,
                data_folder     = config.IMAGE_DATA_FOLDER,
                output_dir      = config.CKPT_PHASE1_LORA,
                vision_tower    = config.VISION_TOWER,
                lora_enable     = True,
                lora_r          = 128,
                lora_alpha      = 256,
                deepspeed_config= DEEPSPEED_CONFIG,
                num_gpus        = config.NUM_GPUS,
            )
        _merge_lora_and_save(
            base_path = config.BASE_MODEL,
            lora_path = config.CKPT_PHASE1_LORA,
            save_path = config.CKPT_PHASE1,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: θ_1 を出発点に音声モダリティを素直に LoRA FT → LoRA マージ → θ_{2,vanilla}
    #   CL 手法なし（これが MERA 論文の「vanilla model」）
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Phase 2: 音声学習（AudioCaps + Clotho-AQA）→ θ_{2,vanilla}")
    print("="*60)
    if not _already_done(config.CKPT_AUDIO_VANILLA, "Phase2（マージ済みモデル）"):
        if not _already_done(config.CKPT_AUDIO_VANILLA_LORA, "Phase2（LoRA 学習）"):
            train_audio(
                model_path      = config.CKPT_PHASE1,
                data_json       = config.AUDIO_DATA_JSON,
                output_dir      = config.CKPT_AUDIO_VANILLA_LORA,
                audio_tower     = config.BEATS_TOWER,
                lr              = 2e-4,
                mm_projector_lr = 2e-5,
                lora_enable     = True,
                lora_r          = 128,
                lora_alpha      = 256,
                deepspeed_config= DEEPSPEED_CONFIG,
                num_gpus        = config.NUM_GPUS,
            )
        _merge_lora_and_save(
            base_path = config.CKPT_PHASE1,
            lora_path = config.CKPT_AUDIO_VANILLA_LORA,
            save_path = config.CKPT_AUDIO_VANILLA,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: θ_1 と θ_{2,vanilla} の LLM 層をマージ → θ_{2,merged}
    #   θ_{2,merged} = 1/2 × θ_1 + 1/2 × θ_{2,vanilla}
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MERA Step1: LLM 重みマージ（θ_1 + θ_{2,vanilla}）→ θ_{2,merged}")
    print("="*60)
    if not _already_done(config.CKPT_MERGED, "Step1（LLM マージ）"):
        _apply_mera_step1_and_save(
            audio_vanilla_dir = config.CKPT_AUDIO_VANILLA,
            prev_dir          = config.CKPT_PHASE1,
            save_dir          = config.CKPT_MERGED,
            modality_index    = 2,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2a: 画像コネクタを画像 replay データで再整合
    #   LLM 凍結・LoRA 不要　出力: CKPT_IMG_REALIGNED/mm_projector.bin のみ
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MERA Step2a: 画像コネクタ ReAlign")
    print("="*60)
    if not _already_done(config.CKPT_IMG_REALIGNED, "Step2a（画像コネクタ ReAlign）"):
        step2 = get_step2_training_flags()
        train_image(
            model_path          = config.CKPT_MERGED,
            data_json           = config.IMAGE_REPLAY_JSON,
            data_folder         = config.IMAGE_REPLAY_FOLDER,
            output_dir          = config.CKPT_IMG_REALIGNED,
            vision_tower        = config.VISION_TOWER,
            tune_mm_mlp_adapter = True,
            lr                  = step2["lr"],
            batch_size          = step2["batch_size"],
            grad_accum          = step2["grad_accum"],
            num_gpus            = config.NUM_GPUS,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2b: 音声コネクタを音声 replay データで再整合
    #   LLM 凍結・LoRA 不要　出力: CKPT_AUD_REALIGNED/mm_projector_a.bin のみ
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MERA Step2b: 音声コネクタ ReAlign")
    print("="*60)
    if not _already_done(config.CKPT_AUD_REALIGNED, "Step2b（音声コネクタ ReAlign）"):
        step2 = get_step2_training_flags()
        train_audio(
            model_path            = config.CKPT_MERGED,
            data_json             = config.AUDIO_REPLAY_JSON,
            output_dir            = config.CKPT_AUD_REALIGNED,
            audio_tower           = config.BEATS_TOWER,
            tune_audio_tower      = False,
            tune_mm_mlp_adapter_a = True,
            lr                    = step2["lr"],
            batch_size            = step2["batch_size"],
            grad_accum            = step2["grad_accum"],
            num_gpus              = config.NUM_GPUS,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Assemble: CKPT_MERGED に両コネクタを注入して最終モデルを組み立てる
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MERA Assemble: 最終モデル組み立て")
    print("="*60)
    if not _already_done(config.CKPT_FINAL, "Assemble（最終モデル）"):
        _assemble_final_model()

    print("\n" + "="*60)
    print("MERA 学習完了")
    print(f"  Phase1 LoRA  （adapter）    : {config.CKPT_PHASE1_LORA}")
    print(f"  Phase1（θ_1）               : {config.CKPT_PHASE1}")
    print(f"  Phase2 LoRA  （adapter）    : {config.CKPT_AUDIO_VANILLA_LORA}")
    print(f"  Phase2（θ_{{2,vanilla}}）    : {config.CKPT_AUDIO_VANILLA}")
    print(f"  Step1（θ_{{2,merged}}）      : {config.CKPT_MERGED}")
    print(f"  Step2a（画像コネクタ）        : {config.CKPT_IMG_REALIGNED}/mm_projector.bin")
    print(f"  Step2b（音声コネクタ）        : {config.CKPT_AUD_REALIGNED}/mm_projector_a.bin")
    print(f"  最終モデル                    : {config.CKPT_FINAL}")
    print("="*60)
