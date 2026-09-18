"""
Clotho-AQA を VideoLLaMA2 学習/評価用 JSON に変換する。

Clotho-AQA のディレクトリ構造（zip 展開後、2026-09-18に実機で確認済み。
train/val/test 別フォルダには分かれておらず、音声ファイルは1フォルダにまとまっている点に注意）:
  /workspace/data/audio/clotho_aqa/
  ├── audio_files/    ←音声ファイル（.wav）。splitの区別はCSV側にしかない
  ├── clotho_aqa_train.csv
  ├── clotho_aqa_val.csv
  └── clotho_aqa_test.csv

CSV 列: file_name, QuestionText, answer, confidence

出力:
  /workspace/data/intermediate/clotho_aqa_train.json
  /workspace/data/intermediate/clotho_aqa_test.json   ← 評価用（FRG）

Clotho-AQA は 1 音声あたり複数の Q&A が付いている。全ペアを個別エントリとして展開する。
"""

import csv
import json
from pathlib import Path


CLOTHO_ROOT  = Path("/workspace/data/audio/clotho_aqa")
AUDIO_DIR    = CLOTHO_ROOT / "audio_files"
OUT_DIR      = Path("/workspace/data/intermediate")

SPLITS = {
    "train": "clotho_aqa_train.csv",
    "val":   "clotho_aqa_val.csv",
    "test":  "clotho_aqa_test.csv",
}


def convert(csv_filename: str, out_path: Path) -> None:
    csv_path = CLOTHO_ROOT / csv_filename

    if not csv_path.exists():
        print(f"[Clotho-AQA] {csv_path} が見つかりません。スキップ")
        return

    print(f"[Clotho-AQA] 変換開始: {csv_path}")
    entries = []
    entry_id = 0
    skipped  = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav_file = AUDIO_DIR / row["file_name"]
            if not wav_file.exists():
                skipped += 1
                continue
            entries.append({
                "id": str(entry_id),
                "audio": str(wav_file),
                "conversations": [
                    {"from": "human", "value": f"<audio>\n{row['QuestionText']}"},
                    {"from": "gpt",   "value": row["answer"]},
                ],
            })
            entry_id += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"[Clotho-AQA] {csv_filename}: {len(entries)} エントリ（{skipped} スキップ）→ {out_path}")


if __name__ == "__main__":
    for split_name, csv_file in SPLITS.items():
        convert(
            csv_filename = csv_file,
            out_path     = OUT_DIR / f"clotho_aqa_{split_name}.json",
        )
