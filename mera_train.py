"""
MERA 順次学習ループ（RunPod 上で実行するスクリプト）

実行順序:
  Phase1  : 画像モダリティで全パラメータ学習（MSCOCO + OK-VQA）
  Step1   : LLM 重みを vanilla Mistral-7B とマージ（忘却抑制）
  Step2   : コネクタのみ replay データで再学習（ReAlign）
  Phase2  : 音声モダリティで全パラメータ学習（AudioCaps + Clotho-AQA）

前提:
  - このファイルは MERAモデル/ 直下（videollama2/ の親ディレクトリ）に置いて実行する
  - mera.py も同じディレクトリに置く
  - 実行コマンド例: python mera_train.py
"""

import os
import sys
import subprocess
import torch

# mera.py は同じディレクトリにある
from mera import mera_step1, get_step2_training_flags

# ── パス設定（RunPod 上の実際のパスに合わせて変更する）────────────────────────

# このファイルがあるディレクトリ = MERAモデル/ のルート
REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# ベースモデル（VideoLLaMA2.1-7B-AV の事前学習済みチェックポイント）
BASE_MODEL = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"

# vanilla Mistral（Step1 の θ_vanilla）
VANILLA_PATH = "mistralai/Mistral-7B-v0.1"

# ビジョンタワー（画像エンコーダ）
VISION_TOWER = "openai/clip-vit-large-patch14-336"

# 学習データ（RunPod Network Volume は /workspace/ にマウントされる）
IMAGE_DATA_JSON   = "/workspace/data/image_train.json"    # MSCOCO + OK-VQA を結合した JSON
IMAGE_DATA_FOLDER = "/workspace/data/coco/images"
AUDIO_DATA_JSON   = "/workspace/data/audio_train.json"    # AudioCaps + Clotho-AQA を結合した JSON
AUDIO_DATA_FOLDER = "/workspace/data/audio"
REPLAY_DATA_JSON  = "/workspace/data/replay_image.json"   # Step2 用の小さな replay セット（画像から抽出）
REPLAY_DATA_FOLDER = "/workspace/data/coco/images"

# 出力先
CKPT_DIR       = "/workspace/output/mera"
CKPT_PHASE1    = os.path.join(CKPT_DIR, "phase1_image")     # Phase1 完了チェックポイント
CKPT_MERGED    = os.path.join(CKPT_DIR, "phase1_merged")    # Step1 マージ後
CKPT_REALIGNED = os.path.join(CKPT_DIR, "phase1_realigned") # Step2 ReAlign 後
CKPT_PHASE2    = os.path.join(CKPT_DIR, "phase2_audio")     # Phase2 完了チェックポイント


# ── ヘルパー関数 ──────────────────────────────────────────────────────────────

def run_training(
    model_path: str,
    data_json: str,
    data_folder: str,
    output_dir: str,
    tune_mm_mlp_adapter: bool = False,
    num_epochs: int = 1,
    batch_size: int = 16,   # 論文 Table 9: Fine-Tuning / Realigning 共通
    grad_accum: int = 1,
    lr: float = 2e-4,       # 論文 Table 9: Fine-Tuning の LLM 学習率
    mm_projector_lr: float = 2e-5,  # 論文 Table 9: Fine-Tuning のコネクタ学習率
    num_gpus: int = 1,
):
    """
    videollama2/train.py を subprocess で呼び出して学習を実行する。

    tune_mm_mlp_adapter=True の場合、train.py 内の既存ブロック（514行付近）が
    コネクタのみを学習可能にするため、MERA Step2 の凍結設定はこれで対応する。
    """
    os.makedirs(output_dir, exist_ok=True)

    train_script = os.path.join(REPO_DIR, "videollama2", "train.py")

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--master_port=29500",
        train_script,
        "--model_type",                   "videollama2_mistral",
        "--model_path",                   model_path,
        "--version",                      "v1",
        "--vision_tower",                 VISION_TOWER,
        "--data_path",                    data_json,
        "--data_folder",                  data_folder,
        "--output_dir",                   output_dir,
        "--bf16",                         "True",
        "--num_train_epochs",             str(num_epochs),
        "--per_device_train_batch_size",  str(batch_size),
        "--gradient_accumulation_steps",  str(grad_accum),
        "--learning_rate",                str(lr),
        "--mm_projector_lr",              str(mm_projector_lr),
        "--model_max_length",             "2048",
        "--lazy_preprocess",              "True",
        "--group_by_modality_length",     "True",
        "--tune_mm_mlp_adapter",          str(tune_mm_mlp_adapter),
        "--save_strategy",                "epoch",
    ]

    print(f"\n{'='*60}")
    print(f"[run_training] 出力: {output_dir}")
    print(f"[run_training] モデル: {model_path}")
    print(f"[run_training] データ: {data_json}")
    print(f"[run_training] tune_mm_mlp_adapter={tune_mm_mlp_adapter}")
    print(f"{'='*60}\n")

    # cwd を REPO_DIR にすることで train.py 内の sys.path.append('./') が有効になる
    subprocess.run(cmd, cwd=REPO_DIR, check=True)


def apply_mera_step1_and_save(
    checkpoint_dir: str,
    vanilla_path: str,
    save_dir: str,
    modality_index: int,
):
    """
    checkpoint_dir のモデルを読み込み、MERA Step1（重みマージ）を適用して save_dir に保存する。

    LLM の重みのみマージし、エンコーダとコネクタは変更しない。
    マージ後のモデルを保存し、Step2 の学習開始点として使用する。
    """
    print(f"\n{'='*60}")
    print(f"[MERA Step1] チェックポイント: {checkpoint_dir}")
    print(f"[MERA Step1] vanilla: {vanilla_path}")
    print(f"[MERA Step1] モダリティ番号 i={modality_index}")
    print(f"{'='*60}\n")

    # videollama2 パッケージを参照できるようにする
    sys.path.insert(0, REPO_DIR)
    from videollama2.model import Videollama2MistralForCausalLM

    # Phase1 のチェックポイントを読み込む
    print(f"[MERA Step1] モデル読み込み中: {checkpoint_dir}")
    model = Videollama2MistralForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # MERA Step1: LLM 重みを vanilla Mistral-7B と平均マージ
    mera_step1(model, vanilla_path=vanilla_path, i=modality_index)

    # マージ済みモデルを保存（Step2 の学習開始点として使用）
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print(f"[MERA Step1] マージ済みモデルを保存: {save_dir}")

    del model
    torch.cuda.empty_cache()


# ── メイン: 順次学習ループ ────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Phase 1: 画像モダリティで全パラメータ学習 ──
    print("\n" + "="*60)
    print("Phase 1: 画像モダリティ学習（MSCOCO + OK-VQA）")
    print("="*60)
    run_training(
        model_path=BASE_MODEL,
        data_json=IMAGE_DATA_JSON,
        data_folder=IMAGE_DATA_FOLDER,
        output_dir=CKPT_PHASE1,
        tune_mm_mlp_adapter=False,  # 全パラメータ学習
        num_epochs=1,
    )

    # ── MERA Step1: LLM 重みを vanilla Mistral-7B とマージ ──
    # 画像学習で変化した LLM 重みを vanilla に寄せることで、
    # 音声学習後も画像の知識を部分的に保持できるようにする。
    apply_mera_step1_and_save(
        checkpoint_dir=CKPT_PHASE1,
        vanilla_path=VANILLA_PATH,
        save_dir=CKPT_MERGED,
        modality_index=2,  # 2番目のモダリティの学習前なので i=2
    )

    # ── MERA Step2: コネクタのみ replay データで再学習（ReAlign）──
    # Step1 で LLM 重みが変わったため、コネクタとの接続がずれる。
    # Step2 の設定（凍結フラグ・学習率）は mera.py の get_step2_training_flags() に集約。
    print("\n" + "="*60)
    print("MERA Step2: コネクタ ReAlign（画像 replay データ）")
    print("="*60)
    step2_flags = get_step2_training_flags()
    run_training(
        model_path=CKPT_MERGED,
        data_json=REPLAY_DATA_JSON,
        data_folder=REPLAY_DATA_FOLDER,
        output_dir=CKPT_REALIGNED,
        num_epochs=1,
        **step2_flags,
    )

    # ── Phase 2: 音声モダリティで全パラメータ学習 ──
    print("\n" + "="*60)
    print("Phase 2: 音声モダリティ学習（AudioCaps + Clotho-AQA）")
    print("="*60)
    run_training(
        model_path=CKPT_REALIGNED,
        data_json=AUDIO_DATA_JSON,
        data_folder=AUDIO_DATA_FOLDER,
        output_dir=CKPT_PHASE2,
        tune_mm_mlp_adapter=False,  # 全パラメータ学習
        num_epochs=1,
    )

    print("\n" + "="*60)
    print("学習完了")
    print(f"  Phase1 画像チェックポイント : {CKPT_PHASE1}")
    print(f"  Step1 マージ済み            : {CKPT_MERGED}")
    print(f"  Step2 ReAlign 済み          : {CKPT_REALIGNED}")
    print(f"  Phase2 音声チェックポイント : {CKPT_PHASE2}")
    print("="*60)
