"""
チェックポイントを image_eval.json または audio_eval.json で評価し Accuracy を返す。

モダリティは JSON の最初のエントリに "image" キーがあれば画像、
"audio" キーがあれば音声として自動判定する。

使い方:
  # 画像評価（BRGの分子・分母に使う）
  python eval_vqa.py \\
      --model       /workspace/output/mera/phase1_image \\
      --eval-json   /workspace/data/image_eval.json \\
      --data-folder /workspace/data

  # 音声評価（FRGの分子・分母に使う）
  python eval_vqa.py \\
      --model     /workspace/output/mera/phase2_audio_vanilla \\
      --eval-json /workspace/data/audio_eval.json

オプション:
  --output    予測結果を JSONL で保存するパス（省略可）
  --limit     評価サンプル数の上限（デバッグ用。省略時は全件）
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def normalize(text: str) -> str:
    """大文字小文字・末尾句読点を統一して比較用に正規化する。"""
    return re.sub(r"[^\w\s]", "", text.strip().lower())


def run(args) -> float:
    from videollama2 import model_init, mm_infer
    from videollama2.utils import disable_torch_init

    disable_torch_init()

    data = json.load(open(args.eval_json))
    if args.limit:
        data = data[: args.limit]

    # モダリティを最初のエントリから自動判定
    first = data[0]
    if "image" in first:
        modal = "image"
    elif "audio" in first:
        modal = "audio"
    else:
        raise ValueError("JSON エントリに 'image' も 'audio' キーもありません")

    print(f"[eval] モデル  : {args.model}")
    print(f"[eval] eval JSON: {args.eval_json}  ({len(data)} サンプル)")
    print(f"[eval] モダリティ: {modal}")

    model, processor, tokenizer = model_init(args.model)

    # 画像専用評価のとき vision_tower 以外を無効化しない（両方必要）
    # 音声専用評価のとき vision_tower はメモリ節約のため None にする
    if modal == "audio":
        model.model.vision_tower = None

    correct = 0
    results = []

    for sample in tqdm(data):
        # 問い文から <image>\n / <audio>\n プレフィックスを除去
        raw_q = sample["conversations"][0]["value"]
        question = re.sub(r"^<(image|audio)>\n", "", raw_q).strip()
        question = question + "\nAnswer with a single word or short phrase."

        ground_truth = sample["conversations"][1]["value"]

        try:
            if modal == "image":
                image_path = os.path.join(args.data_folder, sample["image"])
                tensor = processor["image"](image_path)
            else:
                tensor = processor["audio"](sample["audio"])

            pred = mm_infer(
                tensor,
                question,
                model=model,
                tokenizer=tokenizer,
                modal=modal,
                do_sample=False,
            )
        except Exception as e:
            print(f"[eval] 推論エラー (id={sample['id']}): {e}")
            pred = ""

        is_correct = normalize(pred) == normalize(ground_truth)
        if is_correct:
            correct += 1

        results.append({
            "id":       sample["id"],
            "gt":       ground_truth,
            "pred":     pred,
            "correct":  is_correct,
        })

    accuracy = correct / len(data)
    print(f"\n[eval] Accuracy: {accuracy:.4f}  ({correct}/{len(data)})")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[eval] 予測結果を保存: {args.output}")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       required=True,  help="評価するチェックポイントのパス")
    parser.add_argument("--eval-json",   required=True,  help="image_eval.json または audio_eval.json のパス")
    parser.add_argument("--data-folder", default="/workspace/data",
                        help="画像パスのルートディレクトリ（音声評価では不使用）")
    parser.add_argument("--output",      default=None,   help="予測結果 JSONL の保存先（省略可）")
    parser.add_argument("--limit",       type=int, default=None, help="評価サンプル数の上限（デバッグ用）")
    args = parser.parse_args()

    run(args)
