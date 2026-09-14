"""
VideoLLaMA2 向け MERA（Merge then ReAlign）実装。

Step1: 学習済み LLM の重みと vanilla Mistral-7B を平均マージ
Step2: エンコーダ+LLM を凍結し、mm_projector（コネクタ）のみ学習可能にする
"""

import torch
from transformers import AutoModelForCausalLM


def mera_step1(model, vanilla_path: str, i: int):
    """
    θ_merged = (i-1)/i × θ_prev + 1/i × θ_vanilla

    LLM バックボーンの重みを vanilla Mistral-7B とマージする。
    対象は model.layers.* (Mistral のトランスフォーマーブロック) と lm_head.* のみ。
    エンコーダ (vision_tower) とコネクタ (mm_projector) は変更しない。

    重みキーについて:
      VideoLLaMA2 の named_parameters は model.layers.0.self_attn.q_proj.weight のような形式。
      vanilla MistralForCausalLM も同じキー構造なので、そのまま対応付けられる。

    引数:
        model      : Videollama2MistralForCausalLM のインスタンス（モダリティ i-1 の学習後）
        vanilla_path: Mistral-7B-v0.1 のパスまたは HuggingFace ID
        i          : 現在のモダリティ番号（2番目のモダリティなら i=2）
    """
    assert i >= 2, "MERA Step1 は2番目以降のモダリティから実行する（i >= 2）"

    print(f"[MERA Step1] vanilla モデルを読み込み中: {vanilla_path} ...")
    vanilla = AutoModelForCausalLM.from_pretrained(
        vanilla_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    vanilla_sd = vanilla.state_dict()

    alpha = (i - 1) / i  # θ_prev の重み
    beta  = 1.0 / i      # θ_vanilla の重み

    merged_count = 0
    skipped = []
    for name, param in model.named_parameters():
        # LLM のトランスフォーマー層と LM ヘッドのみを対象にする。
        # エンコーダ (model.vision_tower.*) とコネクタ (model.mm_projector.*) はスキップ。
        if not (name.startswith("model.layers.") or name.startswith("lm_head.")):
            continue

        # VideoLLaMA2 と vanilla Mistral は同じキー構造を持つため、キーはそのまま使用する。
        vanilla_key = name

        if vanilla_key not in vanilla_sd:
            skipped.append(name)
            continue

        v_param = vanilla_sd[vanilla_key].to(param.device, dtype=param.dtype)
        param.data = alpha * param.data + beta * v_param
        merged_count += 1

    del vanilla, vanilla_sd
    torch.cuda.empty_cache()

    print(f"[MERA Step1] {merged_count} テンソルをマージ完了 (i={i}, α={alpha:.3f}, β={beta:.3f})")
    if skipped:
        print(f"[MERA Step1] スキップ（vanilla に該当キーなし）: {skipped}")


def get_step2_training_flags() -> dict:
    """
    MERA Step2（ReAlign）用の train.py フラグと学習率を返す。

    Step1 で LLM の重みがマージされたため、コネクタが新しい LLM との接続を
    再学習する必要がある（ReAlign）。
    train.py は tune_mm_mlp_adapter=True を受け取ると、エンコーダ+LLM を凍結し
    コネクタのみ学習可能にする（train.py 514行付近の既存ブロック）。

    Step2 に関する設定値をここに集約することで、train.py のフラグ名が変わった
    場合もこの関数だけ修正すれば済む。
    """
    return {
        "tune_mm_mlp_adapter": True,
        "lr": 2e-5,       # 論文 Table 9: Realigning フェーズのコネクタ学習率
        "batch_size": 16, # 論文 Table 9: Realigning フェーズのバッチサイズ
    }
