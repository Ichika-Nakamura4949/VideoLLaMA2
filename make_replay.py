"""
replay_image.json / replay_audio.json を image_train.json / audio_train.json から作成する。

Step 2a（画像コネクタ ReAlign）と Step 2b（音声コネクタ ReAlign）に使う少量のデータセット。

MERA 論文（§4）に従い、各学習データの r% をランダムサンプリングする。
論文では r=1 と r=10 の両方で実験しており、デフォルトは r=1。
再現性のために seed を固定している。
"""

import json
import random
from pathlib import Path


IMAGE_TRAIN  = Path("/workspace/data/image_train.json")
AUDIO_TRAIN  = Path("/workspace/data/audio_train.json")
IMAGE_REPLAY = Path("/workspace/data/replay_image.json")
AUDIO_REPLAY = Path("/workspace/data/replay_audio.json")

REPLAY_RATE = 0.01  # 論文の r=1%。r=10% にしたい場合は 0.10 に変更
SEED        = 42


def make_replay(src: Path, dst: Path, rate: float, seed: int) -> None:
    with open(src) as f:
        data = json.load(f)

    n = max(1, int(len(data) * rate))
    random.seed(seed)
    sampled = random.sample(data, n)

    with open(dst, "w") as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    print(f"[replay] {dst.name}: {len(sampled)} エントリ（元: {len(data)} エントリ、{rate*100:.0f}%）")


if __name__ == "__main__":
    make_replay(IMAGE_TRAIN, IMAGE_REPLAY, REPLAY_RATE, SEED)
    make_replay(AUDIO_TRAIN, AUDIO_REPLAY, REPLAY_RATE, SEED)
