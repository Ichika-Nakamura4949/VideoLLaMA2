"""
MERA 実験の設定ファイル。RunPod 実行前にここだけ変更すればよい。

チェックポイントの対応:
  CKPT_PHASE1_LORA       = θ_1 (LoRA adapter): Phase1 訓練後の LoRA チェックポイント（マージ前）
  CKPT_PHASE1            = θ_1 (full model)  : LoRA をマージした完全モデル（Step1 の入力）
  CKPT_AUDIO_VANILLA_LORA= θ_{2,v} (LoRA)   : Phase2 訓練後の LoRA チェックポイント（マージ前）
  CKPT_AUDIO_VANILLA     = θ_{2,v} (full)    : LoRA をマージした完全モデル（Step1 の入力）
  CKPT_MERGED            = θ_{2,m}           : θ_1 と θ_{2,v} を LLM 層でマージした結果
  CKPT_IMG_REALIGNED                         : Step2a で画像コネクタのみ再整合（mm_projector.bin のみ保存）
  CKPT_AUD_REALIGNED                         : Step2b で音声コネクタのみ再整合（mm_projector_a.bin のみ保存）
  CKPT_FINAL                                 : CKPT_MERGED に両コネクタを注入した最終モデル
"""

# ── モデル ──────────────────────────────────────────────────────────────────

# VideoLLaMA2.1-7B-AV: Qwen2-7B + SigLIP + BEATs の事前学習済みモデル
BASE_MODEL   = "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"

# ビジョンエンコーダ（VideoLLaMA2.1 系列で使われる SigLIP）
VISION_TOWER = "google/siglip-so400m-patch14-384"

# 音声エンコーダ（BEATs チェックポイント）
# 入手先: VideoLLaMA2 README の OneDrive リンク → BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
# RunPod に手動で配置しておく（自動ダウンロード不可）
BEATS_TOWER  = "/workspace/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"

# ── データ（RunPod Network Volume = /workspace/） ────────────────────────────

IMAGE_DATA_JSON    = "/workspace/data/image_train.json"   # MSCOCO + OK-VQA
IMAGE_DATA_FOLDER  = "/workspace/data"
AUDIO_DATA_JSON    = "/workspace/data/audio_train.json"   # AudioCaps + Clotho-AQA

# Step2 ReAlign 用 replay セット（各モダリティのサブセット）
IMAGE_REPLAY_JSON   = "/workspace/data/replay_image.json"
IMAGE_REPLAY_FOLDER = "/workspace/data"
AUDIO_REPLAY_JSON   = "/workspace/data/replay_audio.json"

# ── チェックポイント出力先 ─────────────────────────────────────────────────

CKPT_DIR                = "/workspace/output/mera"

# LoRA チェックポイント（マージ前）
CKPT_PHASE1_LORA        = f"{CKPT_DIR}/phase1_image_lora"         # θ_1 LoRA adapter
CKPT_AUDIO_VANILLA_LORA = f"{CKPT_DIR}/phase2_audio_vanilla_lora" # θ_{2,v} LoRA adapter

# マージ済み完全モデル（Step1 以降の入力）
CKPT_PHASE1        = f"{CKPT_DIR}/phase1_image"          # θ_1（画像学習済みフルモデル）
CKPT_AUDIO_VANILLA = f"{CKPT_DIR}/phase2_audio_vanilla"  # θ_{2,vanilla}（音声 FT 結果）
CKPT_MERGED        = f"{CKPT_DIR}/step1_merged"          # θ_{2,merged}（LLM マージ後）
CKPT_IMG_REALIGNED = f"{CKPT_DIR}/step2a_img_realigned"  # 画像コネクタ再整合（mm_projector.bin のみ）
CKPT_AUD_REALIGNED = f"{CKPT_DIR}/step2b_aud_realigned"  # 音声コネクタ再整合（mm_projector_a.bin のみ）
CKPT_FINAL         = f"{CKPT_DIR}/final"                 # 最終モデル（組み立て後）

# ── ハードウェア ──────────────────────────────────────────────────────────

# RunPod インスタンスの GPU 数に合わせて変更する
NUM_GPUS = 1
