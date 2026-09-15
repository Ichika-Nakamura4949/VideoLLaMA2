"""
VideoLLaMA2（Qwen2 バックボーン）向け MERA（Merge then ReAlign）実装。

Step1: θ_{i,vanilla}（新モダリティで素直にFTしたモデル）と
       θ_{i-1}（前段ステージの最終モデル）の LLM 層をマージ
Step2: 画像・音声コネクタを個別に replay データで再整合した後、
       assemble_final_model() で CKPT_MERGED に両コネクタを注入して最終モデルを組み立てる

"vanilla" はオリジナルQwen2 事前学習モデルではなく、
論文 §5.1 定義の「CL 手法なしに新モダリティを素直に FT したモデル」を指す。
"""

import os
import torch
from transformers import AutoModelForCausalLM


def mera_step1(model, prev_path: str, i: int) -> None:
    """
    θ_merged = (i-1)/i × θ_{i-1} + 1/i × θ_{i,vanilla}

    引数:
        model     : θ_{i,vanilla}（新モダリティで素直にFTしたモデル、in memory で変更される）
        prev_path : θ_{i-1} のチェックポイントパス（前段ステージの最終モデル）
                    ─ 呼び出し元が sys.path に REPO_DIR を追加していること
        i         : 現在のモダリティ番号（2番目のモダリティなら i=2）

    マージ対象は Qwen2 の LLM バックボーン全体:
      model.embed_tokens.weight, model.layers.*, model.norm.weight, lm_head.*
    エンコーダ（vision_tower, audio_tower）とコネクタ（mm_projector*）は変更しない。
    """
    assert i >= 2, "MERA Step1 は2番目以降のモダリティから実行する（i >= 2）"

    # 呼び出し元が sys.path を設定済みであることを前提に遅延インポート
    from videollama2.model import Videollama2Qwen2ForCausalLM

    print(f"\n[MERA Step1] θ_{{i-1}} を読み込み中: {prev_path} ...")
    prev_model = Videollama2Qwen2ForCausalLM.from_pretrained(
        prev_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    prev_sd = prev_model.state_dict()

    alpha = (i - 1) / i  # θ_{i-1} の重み
    beta  = 1.0 / i      # θ_{i,vanilla} の重み

    # Qwen2 の LLM バックボーンを構成するパラメータ名の判定条件。
    # prefix マッチに加え、embed_tokens と最終 norm を exact match で捕捉する。
    _LLM_PREFIXES = ("model.layers.", "lm_head.")
    _LLM_EXACT    = {"model.embed_tokens.weight", "model.norm.weight"}

    merged_count = 0
    skipped = []
    for name, param in model.named_parameters():
        # LLM バックボーン全体をマージ対象にする。
        # エンコーダ (vision_tower, audio_tower) とコネクタ (mm_projector*) はスキップ。
        is_llm = any(name.startswith(p) for p in _LLM_PREFIXES) or name in _LLM_EXACT
        if not is_llm:
            continue

        if name not in prev_sd:
            skipped.append(name)
            continue

        prev_param = prev_sd[name].to(param.device, dtype=param.dtype)
        # θ_merged = α × θ_{i-1} + β × θ_{i,vanilla}（in place）
        param.data = alpha * prev_param + beta * param.data
        merged_count += 1

    del prev_model, prev_sd
    torch.cuda.empty_cache()

    print(f"[MERA Step1] {merged_count} テンソルをマージ完了 (i={i}, α={alpha:.3f}, β={beta:.3f})")
    if skipped:
        print(f"[MERA Step1] スキップ（前段モデルに該当キーなし）: {skipped}")


def get_step2_training_flags() -> dict:
    """
    MERA Step2（ReAlign）共通のハイパーパラメータを返す。

    画像コネクタ（Step2a）・音声コネクタ（Step2b）の両学習で使用する。
    tune_mm_mlp_adapter / tune_mm_mlp_adapter_a フラグは
    mera_train.py が各学習呼び出し時に明示的に渡す。
    """
    return {
        "lr":         2e-5,  # 論文 Table 9: Realigning フェーズのコネクタ学習率
        "batch_size": 4,     # 論文 Table 9: 実効バッチ 16 を grad_accum=4 で実現
        "grad_accum": 4,     # 4 × 4 = 16
    }


def assemble_final_model(
    merged_path: str,
    img_connector_bin: str,
    aud_connector_bin: str,
    save_path: str,
) -> None:
    """
    CKPT_MERGED をベースに Step2 で更新された両コネクタを注入し、最終モデルを保存する。

    videollama2_trainer.safe_save_model_for_hf_trainer の挙動:
    - tune_mm_mlp_adapter=True  → mm_projector.bin を保存して return（フルモデルは保存されない）
    - tune_mm_mlp_adapter_a=True → mm_projector_a.bin を保存するが return がないため
                                    フルモデルも CKPT_AUD_REALIGNED に書き出される。
      ただし mm_projector_a.bin のみを使い、フルモデルは無視する。

    注意: get_mm_adapter_state_maybe_zero_3 は部分文字列一致でキーを抽出するため、
    mm_projector.bin には mm_projector_a のキーも混入する（値は CKPT_MERGED 時点の未更新値）。
    img_weights の load_state_dict 前に明示フィルタで除去する。

    引数:
        merged_path      : Step1 マージ済みフルモデルのパス（CKPT_MERGED）
        img_connector_bin: Step2a の出力 mm_projector.bin へのフルパス
        aud_connector_bin: Step2b の出力 mm_projector_a.bin へのフルパス
        save_path        : 最終モデルの保存先（CKPT_FINAL）

    呼び出し元が sys.path に REPO_DIR を追加していること。
    """
    from videollama2.model import Videollama2Qwen2ForCausalLM

    print(f"\n{'='*60}")
    print(f"[MERA Assemble] ベースモデル: {merged_path}")
    print(f"[MERA Assemble] 画像コネクタ: {img_connector_bin}")
    print(f"[MERA Assemble] 音声コネクタ: {aud_connector_bin}")
    print(f"{'='*60}\n")

    model = Videollama2Qwen2ForCausalLM.from_pretrained(
        merged_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )

    # 画像コネクタを差し替える
    # mm_projector.bin には mm_projector_a キーが混入するため明示フィルタで除去する
    img_weights = torch.load(img_connector_bin, map_location="cpu")
    img_weights = {k: v for k, v in img_weights.items() if "mm_projector_a" not in k}
    missing, unexpected = model.load_state_dict(img_weights, strict=False)
    print(f"[MERA Assemble] 画像コネクタ注入完了 ({len(img_weights)} テンソル)")
    if unexpected:
        print(f"[MERA Assemble] 警告 unexpected_keys: {unexpected}")

    # 音声コネクタを差し替える
    aud_weights = torch.load(aud_connector_bin, map_location="cpu")
    missing, unexpected = model.load_state_dict(aud_weights, strict=False)
    print(f"[MERA Assemble] 音声コネクタ注入完了 ({len(aud_weights)} テンソル)")
    if unexpected:
        print(f"[MERA Assemble] 警告 unexpected_keys: {unexpected}")

    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    print(f"[MERA Assemble] 最終モデル保存完了: {save_path}")

    del model
    torch.cuda.empty_cache()
