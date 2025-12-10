import torch
import os

# === 경로 설정 ===
VIDEO_PATH = "/mnt/Ko-AD/walter_white_mike_segment_0_20s.mp4"
MODEL_ID = "unsloth/Qwen3-VL-8B-Instruct"

# [수정] 출력 폴더 지정
OUTPUT_DIR = "output"

# [수정] 파일명은 그대로, 경로만 output 폴더 안으로 연결
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "final_ad_creation_output.json")

# === 폴더 설정 ===
SAMPLED_FRAMES_DIR = "sampled_frames"
CHUNK_FRAMES_DIR = "chunk_frames"
EXTRACTED_AUDIO_PATH = "temp_audio.wav"

# === 모델 설정 ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === 파라미터 ===
SIGLIP_MODEL = "google/siglip-large-patch16-384"
SCENE_THRESHOLD = 0.28
MIN_CHUNK_LEN = 2.0
MAX_CHUNK_LEN = 8.0

LLM_MODEL_ID = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"

# 파일 경로들
STAGE1_OUTPUT_JSON = os.path.join(OUTPUT_DIR, "stage1_narrative_output.json")
STAGE2_OUTPUT_JSON = os.path.join(OUTPUT_DIR, "stage2_final_output.json")
STAGE2_LOG_FILE = os.path.join(OUTPUT_DIR, "stage2_final_log.jsonl")
STAGE3_OUTPUT_JSON = os.path.join(OUTPUT_DIR, "stage3_final_narrative_refined.json")

# ========================================================
# 🌟 [실행 옵션 설정]
# ========================================================
ONLY_RUN_PART_B = True  # True: Qwen(Part A) 건너뜀

# 🌟 Part B 시작 단계 설정 (1, 2, 3 중 선택)
# 1: 서사 생성 (Narrative) 부터 시작 (기본값)
# 2: 기억 추적 (Memory Tracker) 부터 시작 -> [지금 원하시는 설정!]
# 3: 서사 다듬기 (Refinement) 만 실행
PART_B_START_STAGE = 2 

def init_folders():
    import shutil
    # Part A부터 돌릴 때만 폴더 초기화
    if not ONLY_RUN_PART_B:
        for folder in [SAMPLED_FRAMES_DIR, CHUNK_FRAMES_DIR]:
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder, exist_ok=True)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)