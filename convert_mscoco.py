"""
MSCOCO-2014 キャプションデータを VideoLLaMA2 学習用 JSON に変換する。

入力:
  /workspace/data/coco/annotations/captions_train2014.json
  /workspace/data/coco/annotations/captions_val2014.json  ← 評価用（BRG）

出力:
  /workspace/data/intermediate/mscoco_train.json
  /workspace/data/intermediate/mscoco_val.json

MSCOCO キャプションは 1 画像あたり 5 件あるため、学習データは全て使用する（約 41 万エントリ）。
"""

import json
import os
from pathlib import Path


COCO_ROOT    = Path("/workspace/data/coco")
ANNOT_DIR    = COCO_ROOT / "annotations"
OUT_DIR      = Path("/workspace/data/intermediate")


def build_image_id_to_filename(images: list) -> dict:
    return {img["id"]: img["file_name"] for img in images}


def convert(annot_path: Path, split: str, out_path: Path) -> None:
    print(f"[MSCOCO] {split} 変換開始: {annot_path}")
    with open(annot_path) as f:
        data = json.load(f)

    id2file = build_image_id_to_filename(data["images"])
    entries = []
    for ann in data["annotations"]:
        image_id  = ann["image_id"]
        filename  = id2file[image_id]
        rel_path  = f"coco/images/{split}/{filename}"
        entries.append({
            "id": str(ann["id"]),
            "image": rel_path,
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe this image."},
                {"from": "gpt",   "value": ann["caption"]},
            ],
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"[MSCOCO] {split}: {len(entries)} エントリ → {out_path}")


if __name__ == "__main__":
    convert(
        ANNOT_DIR / "captions_train2014.json",
        split    = "train2014",
        out_path = OUT_DIR / "mscoco_train.json",
    )
    convert(
        ANNOT_DIR / "captions_val2014.json",
        split    = "val2014",
        out_path = OUT_DIR / "mscoco_val.json",
    )
