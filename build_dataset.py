"""
データセット変換のマスタースクリプト。RunPod 上で各変換スクリプトを順に実行し、
最終的に mera_train.py が必要とする JSON ファイルを /workspace/data/ に揃える。

実行コマンド（MERAモデル/ 直下から）:
  python build_dataset.py

前提（このスクリプト実行前に手動でダウンロードしておくもの）:
  - /workspace/data/coco/images/train2014/    （MSCOCO train 画像）
  - /workspace/data/coco/images/val2014/      （MSCOCO val 画像、BRG 評価用）
  - /workspace/data/coco/annotations/         （captions_train2014.json 等）
  - /workspace/data/okvqa/                    （OK-VQA の questions/annotations JSON）
  - /workspace/data/audio/clotho/development/       （Clotho v2、7z 展開済み。Pre-Training用）
  - /workspace/data/audio/clotho/clotho_captions_development.csv
  - /workspace/data/audio/clotho_aqa/         （zip 展開済み、全 split 含む）

AudioCapsは2026-09-18にClothoへ差し替え（YouTube由来データの取得が実運用上
困難だったため。詳細はフェーズ3/進捗メモ参照）。

生成物:
  /workspace/data/intermediate/   ← 各データセット単体の変換結果（中間ファイル）
    mscoco_train.json
    mscoco_val.json
    okvqa_train.json
    okvqa_val.json
    clotho_train.json
    clotho_aqa_train.json
    clotho_aqa_test.json

  /workspace/data/               ← mera_train.py / 評価スクリプトが直接参照する JSON
    image_train.json             ← Phase 1 学習用（mscoco_train + okvqa_train）
    audio_train.json             ← Phase 2 学習用（clotho_aqa_train）
    image_eval.json              ← BRG 評価用（okvqa_val）
    audio_eval.json              ← FRG 評価用（clotho_aqa_test）
    replay_image.json            ← Step 2a 用（image_train のサブセット）
    replay_audio.json            ← Step 2b 用（audio_train のサブセット）
"""

import json
import subprocess
import sys
from pathlib import Path

DATA_DIR  = Path("/workspace/data")
INTER_DIR = DATA_DIR / "intermediate"

SCRIPTS = [
    "convert_mscoco.py",
    "convert_okvqa.py",
    "convert_clotho.py",
    "convert_clotho_aqa.py",
]


def run_script(script: str) -> None:
    print(f"\n{'='*60}")
    print(f"実行: {script}")
    print("="*60)
    result = subprocess.run([sys.executable, script], check=True)


def merge(paths: list[Path], out_path: Path) -> None:
    merged = []
    for p in paths:
        with open(p) as f:
            merged.extend(json.load(f))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    total = len(merged)
    srcs  = " + ".join(p.name for p in paths)
    print(f"[merge] {srcs} → {out_path.name}: {total} エントリ")


if __name__ == "__main__":
    # ── ステップ 1: 各データセットを中間 JSON に変換 ──────────────────────────
    for script in SCRIPTS:
        run_script(script)

    # ── ステップ 2: 中間 JSON をマージして最終 JSON を生成 ────────────────────
    print(f"\n{'='*60}")
    print("マージ処理")
    print("="*60)

    # Phase 1 学習用（画像）
    merge(
        paths    = [INTER_DIR / "mscoco_train.json", INTER_DIR / "okvqa_train.json"],
        out_path = DATA_DIR / "image_train.json",
    )

    # Phase 2 学習用（音声）
    # AudioCapsは2026-09-18にClothoへ差し替え。Pre-Training専用（AUDIO_PRETRAIN_JSON）に
    # 回ったため、Phase 2本番学習はClotho-AQA単体になった（論文のAudioCaps+Clotho-AQA構成から変更）
    merge(
        paths    = [INTER_DIR / "clotho_aqa_train.json"],
        out_path = DATA_DIR / "audio_train.json",
    )

    # BRG 評価用（OK-VQA val）
    merge(
        paths    = [INTER_DIR / "okvqa_val.json"],
        out_path = DATA_DIR / "image_eval.json",
    )

    # FRG 評価用（Clotho-AQA test）
    merge(
        paths    = [INTER_DIR / "clotho_aqa_test.json"],
        out_path = DATA_DIR / "audio_eval.json",
    )

    # ── ステップ 3: replay サブセットを生成 ───────────────────────────────────
    print(f"\n{'='*60}")
    print("replay セット生成")
    print("="*60)
    run_script("make_replay.py")

    # ── 完了サマリ ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("データセット準備完了")
    targets = [
        "image_train.json", "audio_train.json",
        "image_eval.json",  "audio_eval.json",
        "replay_image.json", "replay_audio.json",
    ]
    for name in targets:
        p = DATA_DIR / name
        if p.exists():
            with open(p) as f:
                n = len(json.load(f))
            print(f"  {name:30s}: {n} エントリ")
        else:
            print(f"  {name:30s}: ファイルなし（要確認）")
    print("="*60)
