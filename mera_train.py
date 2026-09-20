"""
MERA 順次学習パイプライン（RunPod 上で実行するスクリプト）

LLMバックボーンをVideoLLaMA2付属のQwen2-7Bから、マルチモーダル未対応の
Qwen2.5-1.5B（config.LLM_BACKBONE）に差し替えたため、Phase1の前に
画像・音声それぞれのコネクタをゼロから学習するPre-Training段階を追加している
（論文 Table 9 のPre-Training段階に対応。データ量はAppendix Bの通りCapのみ）。

論文 §4 に基づく 2 モダリティ（画像→音声）の MERA パイプライン:

  Pre-Training（画像）: LLM_BACKBONE + SigLIP のコネクタをランダム初期化から学習（MSCOCOのみ）
             → CKPT_IMG_PRETRAIN/（フルモデル）

  Phase1  : CKPT_IMG_PRETRAIN を出発点に画像モダリティで LoRA 学習（MSCOCO + OK-VQA）
             → CKPT_PHASE1_LORA/（LoRA adapter）
             → LoRA マージ → CKPT_PHASE1/（θ_1 完全モデル）
             ※ 評価①: eval_vqa.py --model CKPT_PHASE1 --eval-json image_eval.json
                → 画像スコアA（BRGの分母）

  Pre-Training（音声）: θ_1 + BEATs のコネクタをランダム初期化から学習（Clothoのみ、2026-09-18にAudioCapsから変更）
             → CKPT_AUD_PRETRAIN/（フルモデル）

  Phase2  : CKPT_AUD_PRETRAIN を出発点に音声モダリティを素直に LoRA FT（Clotho-AQA、confidence="yes"フィルタ済み）
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

import glob
import os
import sys
import torch
import transformers

import config
from mera import mera_step1, get_step2_training_flags, assemble_final_model
from train_runner import REPO_DIR, train_image, train_audio

# DeepSpeed 設定（Phase1/2 の LoRA 学習で使用）
DEEPSPEED_CONFIG = os.path.join(REPO_DIR, "scripts", "zero2_cpu_offload.json")


# ── 再開ロジック補助関数 ───────────────────────────────────────────────────────

def _already_done(path: str, name: str, marker: str = "config.json") -> bool:
    """
    path/marker が存在する場合はスキップし True を返す。
    ディレクトリ存在だけで判定すると OOM クラッシュ後の空ディレクトリを誤検知するため、
    そのステップが実際に書く決定的なファイルの存在で判定する。

    marker の目安:
      LoRA 系    : "adapter_config.json"
      フルモデル系: "config.json"（デフォルト）
      Step2a     : "mm_projector.bin"
      Step2b     : "mm_projector_a.bin"
    """
    marker_path = os.path.join(path, marker)
    if os.path.exists(marker_path):
        print(f"[skip] {name} → 完了マーカー検出のためスキップ: {marker_path}")
        return True
    return False


# ── LoRA マージ補助関数 ────────────────────────────────────────────────────────

def _merge_lora_and_save(base_path: str, lora_path: str, save_path: str) -> None:
    """
    LoRA チェックポイントを base モデルにマージして完全モデルとして保存する。

    videollama2/model/__init__.py の既存ロジック（PeftModel.merge_and_unload）を
    mera_train.py から直接呼ぶラッパー。

    引数:
        base_path : LoRA 学習の出発点（CKPT_IMG_PRETRAIN または CKPT_AUD_PRETRAIN）
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
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_path, use_fast=True)
    tokenizer.save_pretrained(save_path)
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
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.LLM_BACKBONE, use_fast=True)
    tokenizer.save_pretrained(save_dir)
    print(f"[MERA Step1] マージ済みモデルを保存: {save_dir}")

    del model
    torch.cuda.empty_cache()


# ── Pre-Training 後処理補助関数 ────────────────────────────────────────────────

def _has_full_model_weights(path: str) -> bool:
    """
    フルモデルの重みファイルが存在するかを判定する。

    config.jsonは、videollama2_trainer.safe_save_model_for_hf_trainerが
    tune_mm_mlp_adapter(_a)=True時にコネクタのみ保存するパスでも、
    早期returnの前に trainer.model.config.save_pretrained(output_dir) が
    無条件で書き込むため、フルモデル保存済みかどうかの判定には使えない
    （外部レビューで指摘・確認済みのバグ）。実際の重みファイルの有無で判定する。
    """
    candidates = [
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ]
    return any(os.path.exists(os.path.join(path, f)) for f in candidates)


def _ensure_pretrain_full_checkpoint(base_model_path: str, output_dir: str, modality: str) -> None:
    """
    Pre-Training学習の直後に呼び出し、output_dirにフルモデルが保存されているかを
    確認する。保存されていれば何もしない。コネクタの.binしか保存されていない場合
    （videollama2_trainer.safe_save_model_for_hf_trainerの早期returnにより
    フルモデルの自動保存が行われなかった場合）は、base_model_pathを読み込み直し、
    initialize_vision_modules/initialize_audio_modulesのpretrain_mm_mlp_adapter(_a)
    引数でコネクタの重みを注入し、フルモデルとして保存し直す。

    どちらの保存挙動になっても後続のPhase1/Phase2がそのまま model_path として
    使える状態を保証するための、条件分岐による両対応。呼び出し元では
    _already_done による学習自体のスキップ判定とは切り離し、毎回冪等に呼ぶこと
    （config.jsonの存在だけでは学習直後かどうか判定できないため）。

    引数:
        base_model_path : フルモデルが無かった場合に読み込み直すベースモデル
                          （画像=config.LLM_BACKBONE、音声=config.CKPT_PHASE1）
        output_dir      : Pre-Training学習の出力先（CKPT_IMG_PRETRAIN/CKPT_AUD_PRETRAIN）
        modality        : "image" または "audio"
    """
    if _has_full_model_weights(output_dir):
        print(f"[Pre-Training後処理] {output_dir} は既にフルモデルとして保存済み。組み立て不要")
        return

    print(f"[Pre-Training後処理] {output_dir} はコネクタのみ保存されていた。フルモデルを組み立てる")

    sys.path.insert(0, REPO_DIR)
    from videollama2.model import Videollama2Qwen2ForCausalLM

    model = Videollama2Qwen2ForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    if modality == "image":
        projector_bin = os.path.join(output_dir, "mm_projector.bin")

        class _VisionArgs:
            vision_tower             = config.VISION_TOWER
            mm_vision_select_layer   = -2
            mm_vision_select_feature = "patch"
            mm_projector_type        = "linear"
            pretrain_mm_mlp_adapter  = projector_bin

        model.get_model().initialize_vision_modules(model_args=_VisionArgs())

    elif modality == "audio":
        projector_bin = os.path.join(output_dir, "mm_projector_a.bin")

        class _AudioArgs:
            audio_tower               = config.BEATS_TOWER
            mm_projector_a_type       = "linear"
            pretrain_mm_mlp_adapter_a = projector_bin

        model.get_model().initialize_audio_modules(model_args=_AudioArgs())

    else:
        raise ValueError(f"未対応の modality: {modality}")

    model.save_pretrained(output_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    tokenizer.save_pretrained(output_dir)
    print(f"[Pre-Training後処理] フルモデルを組み立てて保存: {output_dir}")

    del model
    torch.cuda.empty_cache()


# ── Step2b 後処理補助関数 ──────────────────────────────────────────────────────

def _cleanup_step2b_output() -> None:
    """
    Step2b完了直後、mm_projector_a.bin以外の不要なファイルを削除する。

    videollama2_trainer.safe_save_model_for_hf_trainerの実装はStep2a
    （tune_mm_mlp_adapter=True）とStep2b（tune_mm_mlp_adapter_a=True）で非対称:
      Step2a: 早期returnがあり、mm_projector.binのみ保存される
      Step2b: 早期returnがなく、mm_projector_a.binに加えて
              使われないフルモデル一式（~16GB）まで保存されてしまう

    assemble_final_model()はmm_projector_a.binしか読まないため、
    このフルモデルは完全に無駄なディスク消費になる。
    """
    keep = {"mm_projector_a.bin"}
    for f in glob.glob(os.path.join(config.CKPT_AUD_REALIGNED, "*")):
        if os.path.isfile(f) and os.path.basename(f) not in keep:
            os.remove(f)
            print(f"[MERA Step2b後処理] 不要ファイル削除: {f}")


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

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.LLM_BACKBONE, use_fast=True)
    tokenizer.save_pretrained(config.CKPT_FINAL)


# ── パイプライン ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ─────────────────────────────────────────────────────────────────────────
    # Pre-Training（画像）: LLM_BACKBONE と SigLIP を繋ぐコネクタを
    #   ランダム初期化から学習する（論文 Table 9 の Pre-Training 段階に対応）
    #   LLM 完全凍結・LoRA 不要・Capのみ（MSCOCO）
    #   → CKPT_IMG_PRETRAIN（LLM_BACKBONE + SigLIP + 学習済み mm_projector のフルモデル）
    #
    #   videollama2_trainer.pyのtune_mm_mlp_adapter=True早期returnにより、
    #   ここではmm_projector.bin（+config.json）のみが保存される。フルモデルへの
    #   組み立ては_ensure_pretrain_full_checkpoint()が毎回冪等にチェック・対応する
    #   （config.jsonは両パターンで必ず存在するため完了判定には使えない。
    #   外部レビューで指摘・修正済み）。
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Pre-Training（画像）: LLM_BACKBONE ↔ SigLIP コネクタの初期学習")
    print("="*60)
    if not _already_done(config.CKPT_IMG_PRETRAIN, "Pre-Training（画像コネクタ学習）", "mm_projector.bin"):
        train_image(
            model_path          = config.LLM_BACKBONE,
            data_json           = config.IMAGE_PRETRAIN_JSON,
            data_folder         = config.IMAGE_DATA_FOLDER,
            output_dir          = config.CKPT_IMG_PRETRAIN,
            vision_tower        = config.VISION_TOWER,
            tune_mm_mlp_adapter = True,
            lora_enable         = False,
            lr                  = 1e-3,       # 論文 Table 9: Pre-Training のコネクタ学習率
            mm_projector_lr     = 1e-3,
            batch_size          = 16,
            grad_accum          = 8,          # 16 × 8 = 128（論文 Table 9: Pre-Training バッチ128。batch 32はRTX 4090 24GBでOOM）
            num_gpus            = config.NUM_GPUS,
        )
    # 学習をスキップした場合も含め毎回冪等にチェックする（_already_doneの外側）
    _ensure_pretrain_full_checkpoint(
        base_model_path = config.LLM_BACKBONE,
        output_dir      = config.CKPT_IMG_PRETRAIN,
        modality        = "image",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: 画像モダリティで LoRA 学習 → LoRA マージ → θ_1
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Phase 1: 画像学習（MSCOCO + OK-VQA）→ θ_1")
    print("="*60)
    if not _already_done(config.CKPT_PHASE1, "Phase1（マージ済みモデル）", "config.json"):
        if not _already_done(config.CKPT_PHASE1_LORA, "Phase1（LoRA 学習）", "adapter_config.json"):
            train_image(
                model_path      = config.CKPT_IMG_PRETRAIN,
                data_json       = config.IMAGE_DATA_JSON,
                data_folder     = config.IMAGE_DATA_FOLDER,
                output_dir      = config.CKPT_PHASE1_LORA,
                vision_tower    = config.VISION_TOWER,
                lora_enable     = True,
                lora_r          = 128,
                lora_alpha      = 128,
                deepspeed_config= DEEPSPEED_CONFIG,
                num_gpus        = config.NUM_GPUS,
            )
        _merge_lora_and_save(
            base_path = config.CKPT_IMG_PRETRAIN,
            lora_path = config.CKPT_PHASE1_LORA,
            save_path = config.CKPT_PHASE1,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Pre-Training（音声）: θ_1（CKPT_PHASE1）を出発点に、BEATs を繋ぐコネクタを
    #   ランダム初期化から学習する（論文 Table 9 の Pre-Training 段階に対応）
    #   LLM・音声エンコーダ完全凍結・LoRA不要・Capのみ（Clotho、2026-09-18にAudioCapsから変更）
    #   → CKPT_AUD_PRETRAIN（θ_1 + BEATs + 学習済み mm_projector_a のフルモデル）
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Pre-Training（音声）: θ_1 ↔ BEATs コネクタの初期学習")
    print("="*60)
    if not _already_done(config.CKPT_AUD_PRETRAIN, "Pre-Training（音声コネクタ学習）", "mm_projector_a.bin"):
        train_audio(
            model_path            = config.CKPT_PHASE1,
            data_json             = config.AUDIO_PRETRAIN_JSON,
            output_dir            = config.CKPT_AUD_PRETRAIN,
            audio_tower           = config.BEATS_TOWER,
            tune_audio_tower      = False,
            tune_mm_mlp_adapter_a = True,
            lora_enable           = False,
            lr                    = 1e-3,     # 論文 Table 9: Pre-Training のコネクタ学習率
            mm_projector_lr       = 1e-3,
            batch_size            = 16,
            grad_accum            = 8,        # 16 × 8 = 128（論文 Table 9: Pre-Training バッチ128。batch 32はRTX 4090 24GBでOOM）
            num_gpus              = config.NUM_GPUS,
        )
    # 学習をスキップした場合も含め毎回冪等にチェックする（_already_doneの外側）
    _ensure_pretrain_full_checkpoint(
        base_model_path = config.CKPT_PHASE1,
        output_dir      = config.CKPT_AUD_PRETRAIN,
        modality        = "audio",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: θ_1 を出発点に音声モダリティを素直に LoRA FT → LoRA マージ → θ_{2,vanilla}
    #   CL 手法なし（これが MERA 論文の「vanilla model」）
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Phase 2: 音声学習（Clotho-AQA）→ θ_{2,vanilla}")
    print("="*60)
    if not _already_done(config.CKPT_AUDIO_VANILLA, "Phase2（マージ済みモデル）", "config.json"):
        if not _already_done(config.CKPT_AUDIO_VANILLA_LORA, "Phase2（LoRA 学習）", "adapter_config.json"):
            train_audio(
                model_path      = config.CKPT_AUD_PRETRAIN,
                data_json       = config.AUDIO_DATA_JSON,
                output_dir      = config.CKPT_AUDIO_VANILLA_LORA,
                audio_tower     = config.BEATS_TOWER,
                lr              = 2e-4,
                mm_projector_lr = 2e-5,
                lora_enable     = True,
                lora_r          = 128,
                lora_alpha      = 128,
                deepspeed_config= DEEPSPEED_CONFIG,
                num_gpus        = config.NUM_GPUS,
            )
        _merge_lora_and_save(
            base_path = config.CKPT_AUD_PRETRAIN,
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
    if not _already_done(config.CKPT_MERGED, "Step1（LLM マージ）", "config.json"):
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
    if not _already_done(config.CKPT_IMG_REALIGNED, "Step2a（画像コネクタ ReAlign）", "mm_projector.bin"):
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
    if not _already_done(config.CKPT_AUD_REALIGNED, "Step2b（音声コネクタ ReAlign）", "mm_projector_a.bin"):
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
    # 学習をスキップした場合（クリーンアップ前にクラッシュして再開した場合）も
    # 含め毎回冪等に呼ぶ（外部レビューで指摘・修正）
    _cleanup_step2b_output()

    # ─────────────────────────────────────────────────────────────────────────
    # Assemble: CKPT_MERGED に両コネクタを注入して最終モデルを組み立てる
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MERA Assemble: 最終モデル組み立て")
    print("="*60)
    if not _already_done(config.CKPT_FINAL, "Assemble（最終モデル）", "config.json"):
        _assemble_final_model()

    print("\n" + "="*60)
    print("MERA 学習完了")
    print(f"  Pre-Training（画像）        : {config.CKPT_IMG_PRETRAIN}")
    print(f"  Phase1 LoRA  （adapter）    : {config.CKPT_PHASE1_LORA}")
    print(f"  Phase1（θ_1）               : {config.CKPT_PHASE1}")
    print(f"  Pre-Training（音声）        : {config.CKPT_AUD_PRETRAIN}")
    print(f"  Phase2 LoRA  （adapter）    : {config.CKPT_AUDIO_VANILLA_LORA}")
    print(f"  Phase2（θ_{{2,vanilla}}）    : {config.CKPT_AUDIO_VANILLA}")
    print(f"  Step1（θ_{{2,merged}}）      : {config.CKPT_MERGED}")
    print(f"  Step2a（画像コネクタ）        : {config.CKPT_IMG_REALIGNED}/mm_projector.bin")
    print(f"  Step2b（音声コネクタ）        : {config.CKPT_AUD_REALIGNED}/mm_projector_a.bin")
    print(f"  最終モデル                    : {config.CKPT_FINAL}")
    print("="*60)
