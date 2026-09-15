"""
AudioCaps を VideoLLaMA2 学習用 JSON に変換する。

入力:
  /workspace/data/audio/audiocaps/train.csv  （列: audiocap_id, youtube_id, start_time, caption）
  /workspace/data/audio/audiocaps/val.csv    （評価では使わないが一応処理する）

音声ファイルの命名規則: yt-dlp でダウンロード後、ffmpeg でクリップしたファイルが
  /workspace/data/audio/audiocaps/train/{youtube_id}.wav
として存在する前提。yt-dlp 失敗・削除済み動画によるファイル欠損は自動スキップする。

出力:
  /workspace/data/intermediate/audiocaps_train.json

評価には Clotho-AQA を使うため、audiocaps_val.json は不要。
ただし val.csv が存在する場合は念のため変換する。
"""

import csv
import json
from pathlib import Path


AUDIO_ROOT   = Path("/workspace/data/audio/audiocaps")
OUT_DIR      = Path("/workspace/data/intermediate")


def convert(csv_path: Path, audio_dir: Path, out_path: Path) -> None:
    print(f"[AudioCaps] 変換開始: {csv_path}")
    entries  = []
    skipped  = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            youtube_id = row["youtube_id"]
            wav_file   = audio_dir / f"{youtube_id}.wav"
            if not wav_file.exists():
                skipped += 1
                continue
            abs_path = str(audio_dir / f"{youtube_id}.wav")
            entries.append({
                "id": row["audiocap_id"],
                "audio": abs_path,
                "conversations": [
                    {"from": "human", "value": "<audio>\nDescribe this audio."},
                    {"from": "gpt",   "value": row["caption"]},
                ],
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"[AudioCaps] {csv_path.stem}: {len(entries)} エントリ（{skipped} スキップ）→ {out_path}")


if __name__ == "__main__":
    convert(
        csv_path  = AUDIO_ROOT / "train.csv",
        audio_dir = AUDIO_ROOT / "train",
        out_path  = OUT_DIR / "audiocaps_train.json",
    )
    val_csv = AUDIO_ROOT / "val.csv"
    if val_csv.exists():
        convert(
            csv_path  = val_csv,
            audio_dir = AUDIO_ROOT / "val",
            out_path  = OUT_DIR / "audiocaps_val.json",
        )
