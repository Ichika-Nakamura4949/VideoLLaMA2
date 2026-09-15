"""
AudioCaps の音声ファイルを yt-dlp + ffmpeg でダウンロードする。

AudioCaps は YouTube の 10 秒クリップで構成されており、CSV に
  audiocap_id, youtube_id, start_time, caption
の形式でメタデータが配布されている。yt-dlp で全体をダウンロードして
ffmpeg で start_time から 10 秒を切り出すことで wav を作る。

必要ツール（RunPod 環境）:
  pip install yt-dlp
  apt-get install -y ffmpeg

使い方:
  python download_audiocaps.py

出力先:
  /workspace/data/audio/audiocaps/train/{youtube_id}.wav

命名規則は convert_audiocaps.py の wav_file チェックと一致させている。
ダウンロード済みファイルはスキップするため、中断しても再実行で続きから再開できる。
失敗した youtube_id は failed_audiocaps.txt に記録する。
"""

import csv
import subprocess
import time
from pathlib import Path


CSV_PATH      = Path("/workspace/data/audio/audiocaps/train.csv")
OUT_DIR       = Path("/workspace/data/audio/audiocaps/train")
FAILED_LOG    = Path("/workspace/data/audio/audiocaps/failed_audiocaps.txt")
CLIP_DURATION = 10     # AudioCaps のクリップ長は常に 10 秒
SLEEP_SEC     = 1.0    # ダウンロード間隔（レートリミット対策）
SAMPLE_RATE   = 16000  # BEATs の標準入力サンプリングレート


def download_clip(youtube_id: str, start_time: float, out_path: Path) -> bool:
    url          = f"https://www.youtube.com/watch?v={youtube_id}"
    tmp_template = str(out_path.parent / f"{youtube_id}.tmp.%(ext)s")
    tmp_wav      = out_path.parent / f"{youtube_id}.tmp.wav"

    # yt-dlp: 最良音質を wav に変換してダウンロード
    dl = subprocess.run(
        [
            "yt-dlp",
            "-x", "--audio-format", "wav",
            "--audio-quality", "0",
            "--no-playlist",
            "-o", tmp_template,
            url,
        ],
        capture_output=True,
    )
    if dl.returncode != 0 or not tmp_wav.exists():
        return False

    # ffmpeg: start_time から 10 秒を切り出して 16 kHz モノラルに変換
    trim = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(tmp_wav),
            "-ss", str(start_time),
            "-t", str(CLIP_DURATION),
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            str(out_path),
        ],
        capture_output=True,
    )
    tmp_wav.unlink(missing_ok=True)
    return trim.returncode == 0


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    failed: list[str] = []
    total = len(rows)

    for i, row in enumerate(rows, 1):
        youtube_id = row["youtube_id"]
        start_time = float(row["start_time"])
        out_path   = OUT_DIR / f"{youtube_id}.wav"

        if out_path.exists():
            print(f"[{i}/{total}] skip  {youtube_id}")
            continue

        print(f"[{i}/{total}] dl    {youtube_id}  (start={start_time}s)", flush=True)
        ok = download_clip(youtube_id, start_time, out_path)

        if ok:
            print(f"[{i}/{total}] done  {youtube_id}")
        else:
            print(f"[{i}/{total}] FAIL  {youtube_id}")
            failed.append(youtube_id)

        time.sleep(SLEEP_SEC)

    if failed:
        with open(FAILED_LOG, "w") as f:
            f.write("\n".join(failed))
        print(f"\n失敗: {len(failed)} 件 → {FAILED_LOG}")

    succeeded = total - len(failed)
    print(f"完了: {succeeded}/{total} 件")
