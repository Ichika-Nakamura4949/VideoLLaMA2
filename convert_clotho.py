"""
Clotho（v2、development分割）を VideoLLaMA2 Pre-Training用 JSON に変換する。

AudioCapsの代替として採用（2026-09-18、YouTube由来データのダウンロードが
実運用上困難と判明したため）。MERA論文自体はAudioCapsを採用しているが、
論文中に選定理由の明記はなく、Clotho-AQAとの素性の近さ（同じClotho音源）
の観点ではむしろClothoの方が一貫性が高いと判断した。

入力:
  /workspace/data/audio/clotho/clotho_captions_development.csv
    （列: file_name, caption_1, caption_2, caption_3, caption_4, caption_5。実データで確認済み）
  /workspace/data/audio/clotho/development/{file_name}
    （7z展開済みの音声ファイル）

1音声につき5キャプションが付与されているため、Clotho-AQAと同様に
全ペアを個別エントリとして展開する（3,839クリップ × 5 ≒ 19,195エントリ想定）。

出力:
  /workspace/data/intermediate/clotho_train.json
"""

import csv
import json
from pathlib import Path


CLOTHO_ROOT = Path("/workspace/data/audio/clotho")
AUDIO_DIR   = CLOTHO_ROOT / "development"
CSV_PATH    = CLOTHO_ROOT / "clotho_captions_development.csv"
OUT_PATH    = Path("/workspace/data/intermediate/clotho_train.json")

CAPTION_COLUMNS = ["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"]


def convert() -> None:
    if not CSV_PATH.exists():
        print(f"[Clotho] {CSV_PATH} が見つかりません。スキップ")
        return

    print(f"[Clotho] 変換開始: {CSV_PATH}")
    entries  = []
    entry_id = 0
    skipped  = 0

    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav_file = AUDIO_DIR / row["file_name"]
            if not wav_file.exists():
                skipped += len(CAPTION_COLUMNS)
                continue
            for col in CAPTION_COLUMNS:
                caption = row[col]
                entries.append({
                    "id": str(entry_id),
                    "audio": str(wav_file),
                    "conversations": [
                        {"from": "human", "value": "<audio>\nDescribe this audio."},
                        {"from": "gpt",   "value": caption},
                    ],
                })
                entry_id += 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"[Clotho] {len(entries)} エントリ（音声ファイル欠損によるスキップ {skipped} 件）→ {OUT_PATH}")


if __name__ == "__main__":
    convert()
