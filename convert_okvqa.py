"""
OK-VQA を VideoLLaMA2 学習/評価用 JSON に変換する。

入力（train）:
  /workspace/data/okvqa/OpenEnded_mscoco_train2014_questions.json
  /workspace/data/okvqa/mscoco_train2014_annotations.json

入力（val）:
  /workspace/data/okvqa/OpenEnded_mscoco_val2014_questions.json
  /workspace/data/okvqa/mscoco_val2014_annotations.json

出力:
  /workspace/data/intermediate/okvqa_train.json
  /workspace/data/intermediate/okvqa_val.json     ← 評価用（BRG）

画像は MSCOCO と共有。画像パスは MSCOCO と同じ形式で記述する。

回答の選び方: 複数の回答が付いているが、最頻回答を1つだけ使う。
同率の場合は最初に現れたものを採用する。
"""

import json
from collections import Counter
from pathlib import Path


OKVQA_ROOT = Path("/workspace/data/okvqa")
OUT_DIR    = Path("/workspace/data/intermediate")


def most_common_answer(answers: list) -> str:
    counts = Counter(a["answer"].strip().lower() for a in answers)
    return counts.most_common(1)[0][0]


def convert(q_path: Path, a_path: Path, img_split: str, out_path: Path) -> None:
    print(f"[OK-VQA] 変換開始: {q_path.name}")
    with open(q_path) as f:
        qs = json.load(f)["questions"]
    with open(a_path) as f:
        anns = json.load(f)["annotations"]

    qid2ans = {a["question_id"]: a["answers"] for a in anns}

    entries = []
    for q in qs:
        qid      = q["question_id"]
        image_id = q["image_id"]
        filename = f"COCO_{img_split}_{str(image_id).zfill(12)}.jpg"
        rel_path = f"coco/images/{img_split}/{filename}"
        answer   = most_common_answer(qid2ans[qid])
        entries.append({
            "id": str(qid),
            "image": rel_path,
            "conversations": [
                {"from": "human", "value": f"<image>\n{q['question']}"},
                {"from": "gpt",   "value": answer},
            ],
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"[OK-VQA] {img_split}: {len(entries)} エントリ → {out_path}")


if __name__ == "__main__":
    convert(
        q_path    = OKVQA_ROOT / "OpenEnded_mscoco_train2014_questions.json",
        a_path    = OKVQA_ROOT / "mscoco_train2014_annotations.json",
        img_split = "train2014",
        out_path  = OUT_DIR / "okvqa_train.json",
    )
    convert(
        q_path    = OKVQA_ROOT / "OpenEnded_mscoco_val2014_questions.json",
        a_path    = OKVQA_ROOT / "mscoco_val2014_annotations.json",
        img_split = "val2014",
        out_path  = OUT_DIR / "okvqa_val.json",
    )
