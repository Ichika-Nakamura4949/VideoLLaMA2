"""
videollama2/train.py を torchrun 経由で呼び出すラッパー。

画像学習（Phase1・Step2a）と音声学習（Phase2・Step2b）で引数セットが大きく異なるため、
用途別の関数に分けて疎結合を保つ。

tune_mm_mlp_adapter / tune_mm_mlp_adapter_a フラグの使い分け:
  Phase1/Phase2（全パラメータ学習）: False（デフォルト）→ freeze ブロックをスキップ
  Step2a/Step2b（コネクタのみ再整合）: True → LLM+エンコーダを凍結しコネクタのみ学習

lora_enable フラグの使い分け:
  Phase1/Phase2（LLM を学習）: True → LoRA で LLM の trainable param を削減
  Step2a/Step2b（コネクタのみ再整合）: False（デフォルト）→ LLM 凍結なので LoRA 不要

途中保存と再開:
  save_strategy=steps で output_dir/checkpoint-N に save_steps ごとに保存する。
  videollama2/train.py は output_dir に checkpoint-* があれば resume_from_checkpoint=True で
  再開するため、中断後は同じ引数で呼び直すだけでよい。段階完了後の checkpoint-* の掃除は
  mera_train.py 側（_cleanup_checkpoints）で行う。
"""

import os
import subprocess

# このファイルが置かれているディレクトリ = MERAモデル/ のルート
REPO_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(REPO_DIR, "videollama2", "train.py")


def _run(cmd: list[str]) -> None:
    """コマンドを REPO_DIR から実行する。失敗時は CalledProcessError を送出する。"""
    print(f"\n{'='*60}")
    print(f"[torchrun] {' '.join(cmd[:6])} ...")
    print(f"{'='*60}\n")
    subprocess.run(cmd, cwd=REPO_DIR, check=True)


def train_image(
    model_path: str,
    data_json: str,
    data_folder: str,
    output_dir: str,
    vision_tower: str,
    *,
    tune_mm_mlp_adapter: bool = False,
    lora_enable: bool = False,
    lora_r: int = 128,
    lora_alpha: int = 128,
    deepspeed_config: str | None = None,
    lr: float = 2e-4,
    mm_projector_lr: float = 2e-5,
    batch_size: int = 4,
    grad_accum: int = 4,
    num_epochs: int = 1,
    num_gpus: int = 1,
    save_steps: int = 500,
    logging_steps: int = 500,
) -> None:
    """
    画像モダリティの学習を実行する（Phase1・Step2a で共用）。

    引数:
        model_path          : 学習開始点のモデルパス
        data_json           : 学習データの JSON ファイルパス
        data_folder         : 画像ファイルのルートディレクトリ
        output_dir          : チェックポイントの出力先
        vision_tower        : ビジョンエンコーダの ID またはパス
        tune_mm_mlp_adapter : True = コネクタのみ学習（Step2a ReAlign 用）
        lora_enable         : True = LoRA で LLM を学習（Phase1 用）
        lora_r              : LoRA rank（論文 Appendix C.1 = 128）
        lora_alpha          : LoRA alpha（論文 Appendix C.1 = 128、alpha/r=1.0）
        deepspeed_config    : DeepSpeed 設定 JSON のパス。None でオフ
        lr                  : LLM の学習率
        mm_projector_lr     : コネクタの学習率
        batch_size          : デバイスあたりのバッチサイズ
        grad_accum          : 勾配累積ステップ数
        num_epochs          : エポック数
        num_gpus            : GPU 数
        save_steps          : 途中保存の間隔（最適化ステップ数）。output_dir/checkpoint-N に
                              保存され、中断後に同じ output_dir で再実行すると自動で再開する
        logging_steps       : loss/学習率を trainer_state.json の log_history に記録する間隔
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--master_port=29500",
        TRAIN_SCRIPT,
        "--model_type",                   "videollama2_qwen2",
        "--model_path",                   model_path,
        "--vision_tower",                 vision_tower,
        "--mm_vision_select_layer",       "-2",
        "--image_aspect_ratio",           "pad",
        "--num_frames",                   "1",   # 画像のみ使用。16だと同一画像を16枚に複製してSigLIPに通す（結果は平均で同一・計算16倍）ためRTX 4090でOOM
        "--data_path",                    data_json,
        "--data_folder",                  data_folder,
        "--output_dir",                   output_dir,
        "--bf16",                         "True",
        "--tf32",                         "True",
        "--fp16",                         "False",
        "--num_train_epochs",             str(num_epochs),
        "--per_device_train_batch_size",  str(batch_size),
        "--gradient_accumulation_steps",  str(grad_accum),
        "--learning_rate",                str(lr),
        "--mm_projector_lr",              str(mm_projector_lr),
        "--weight_decay",                 "0.",
        "--warmup_ratio",                 "0.03",
        "--lr_scheduler_type",            "cosine",
        "--model_max_length",             "2048",
        "--lazy_preprocess",              "True",
        "--gradient_checkpointing",       "True",
        "--dataloader_num_workers",       "4",
        "--save_strategy",                "steps",
        "--save_steps",                   str(save_steps),
        "--save_total_limit",             "1",   # 最新の checkpoint-N だけ残す（旧版は自動削除）
        "--logging_steps",                str(logging_steps),
        "--tune_mm_mlp_adapter",          str(tune_mm_mlp_adapter),
        "--report_to",                    "tensorboard",
    ]

    if lora_enable:
        cmd += [
            "--lora_enable", "True",
            "--lora_r",      str(lora_r),
            "--lora_alpha",  str(lora_alpha),
        ]

    if deepspeed_config:
        cmd += ["--deepspeed", deepspeed_config]

    _run(cmd)


def train_audio(
    model_path: str,
    data_json: str,
    output_dir: str,
    audio_tower: str,
    *,
    tune_audio_tower: bool = True,
    tune_mm_mlp_adapter_a: bool = False,
    lora_enable: bool = False,
    lora_r: int = 128,
    lora_alpha: int = 128,
    deepspeed_config: str | None = None,
    lr: float = 2e-5,
    mm_projector_lr: float | None = None,
    batch_size: int = 4,
    grad_accum: int = 4,
    num_epochs: int = 1,
    num_gpus: int = 1,
    save_steps: int = 500,
    logging_steps: int = 500,
) -> None:
    """
    音声モダリティの学習を実行する（Phase2・Step2b で共用）。

    画像学習と異なり --data_path_a を使う。
    音声データ JSON には絶対パスが入っているため --data_folder は不要。

    引数:
        model_path            : 学習開始点のモデルパス
        data_json             : 音声学習データの JSON ファイルパス（絶対パス記載）
        output_dir            : チェックポイントの出力先
        audio_tower           : BEATs チェックポイントのパス
        tune_audio_tower      : True = 音声エンコーダを学習対象にする（Phase2 用）
                                False = エンコーダ凍結（Step2b ReAlign 用）
        tune_mm_mlp_adapter_a : False = 全パラメータ学習（Phase2 用・デフォルト）
                                True  = 音声コネクタのみ学習（Step2b ReAlign 用）
        lora_enable           : True = LoRA で LLM を学習（Phase2 用）
        lora_r                : LoRA rank（論文 Appendix C.1 = 128）
        lora_alpha            : LoRA alpha（論文 Appendix C.1 = 128、alpha/r=1.0）
        deepspeed_config      : DeepSpeed 設定 JSON のパス。None でオフ
        lr                    : LLM・音声エンコーダの学習率
        mm_projector_lr       : 音声コネクタ（mm_projector_a）の学習率。
                                None のとき lr と同じ値が使われる。
                                Phase2 では lr=2e-4 / mm_projector_lr=2e-5 を推奨（論文 Table 9）
        batch_size            : デバイスあたりのバッチサイズ
        grad_accum            : 勾配累積ステップ数
        num_epochs            : エポック数
        num_gpus              : GPU 数
        save_steps            : 途中保存の間隔（最適化ステップ数）。output_dir/checkpoint-N に
                                保存され、中断後に同じ output_dir で再実行すると自動で再開する
        logging_steps         : loss/学習率を trainer_state.json の log_history に記録する間隔
    """
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "--master_port=29500",
        TRAIN_SCRIPT,
        "--model_type",                   "videollama2_qwen2",
        "--model_path",                   model_path,
        "--data_path_a",                  data_json,
        "--audio_tower",                  audio_tower,
        "--tune_audio_tower",             str(tune_audio_tower),
        "--tune_mm_mlp_adapter_a",        str(tune_mm_mlp_adapter_a),
        "--output_dir",                   output_dir,
        "--bf16",                         "True",
        "--tf32",                         "True",
        "--fp16",                         "False",
        "--num_train_epochs",             str(num_epochs),
        "--per_device_train_batch_size",  str(batch_size),
        "--gradient_accumulation_steps",  str(grad_accum),
        "--learning_rate",                str(lr),
        "--weight_decay",                 "0.",
        "--warmup_ratio",                 "0.03",
        "--lr_scheduler_type",            "cosine",
        "--model_max_length",             "2048",
        "--lazy_preprocess",              "True",
        "--gradient_checkpointing",       "True",
        "--dataloader_num_workers",       "4",
        "--save_strategy",                "steps",
        "--save_steps",                   str(save_steps),
        "--save_total_limit",             "1",   # 最新の checkpoint-N だけ残す（旧版は自動削除）
        "--logging_steps",                str(logging_steps),
        "--report_to",                    "tensorboard",
    ]

    if mm_projector_lr is not None:
        cmd += ["--mm_projector_lr", str(mm_projector_lr)]

    if lora_enable:
        cmd += [
            "--lora_enable", "True",
            "--lora_r",      str(lora_r),
            "--lora_alpha",  str(lora_alpha),
        ]

    if deepspeed_config:
        cmd += ["--deepspeed", deepspeed_config]

    _run(cmd)
