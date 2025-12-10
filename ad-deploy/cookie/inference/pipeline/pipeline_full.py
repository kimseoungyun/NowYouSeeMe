import os
import json
import subprocess
import uuid
import gc

import pandas as pd
import numpy as np
from PIL import Image
import ffmpeg
import torch
import shutil

# ========================
# CONFIG
# ========================
from config import (
    VIDEO_PATH, EXTRACTED_AUDIO_PATH, WHISPER_DEVICE, WHISPER_MODEL,
    MAX_FRAMES_PER_CHUNK, QWEN_MODEL_ID, LLAMA_MODEL_NAME,
    WHISPER_OUTPUT, QWEN_OUTPUT_ONLY, STAGE1_OUTPUT, STAGE2_OUTPUT,
    STAGE3_OUTPUT, FINAL_OUTPUT, OUTPUT_DIR
)

# ---------------- AUDIO ----------------
from audio.extract_audio import extract_audio
from audio.whisper_transcribe import transcribe_audio

# ---------------- VIDEO ----------------
from video.scene_detection_siglip import detect_scenes_siglip

# ---------------- CHUNKING ----------------
from chunking.chunk_preprocess import preprocess_dialogues
from chunking.chunk_split import (
    expand_scene_to_dialogues, split_scene_by_dialogues, chunk_subscenes
)
from chunking.chunk_merge import merge_last_chunk, merge_too_short_chunks
from chunking.chunk_refine import refine_long_chunks

# ---------------- FRAME SAMPLING ----------------
from video.frame_sampling import get_shot_based_frames

# ---------------- QWEN (VISION) ----------------
from vision.qwen_loader import load_qwen_model
from vision.qwen_stage1_identification import qwen_stage1_identify
from vision.qwen_stage2_analysis import qwen_stage2_analysis

# ---------------- LLaMA (LANGUAGE) ----------------
from language.llama_loader import load_llama_model
from language.stage1_visual_narrative import run_stage1_visual_narrative
from language.stage2_character_tracking import run_stage2_character_tracking
from language.stage3_narrative_refine import run_stage3_narrative_refine

def run_full_pipeline():
    print("\n=========================================")
    print("🚀  STARTING FULL Ko-AD INFERENCE PIPELINE")
    print("===========================================\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 폴더 초기화 (원본과 동일)
    for folder in ["sampled_frames", "chunk_frames"]:
        shutil.rmtree(folder, ignore_errors=True)
        os.makedirs(folder, exist_ok=True)

# =====================================================================
#           VIDEO CODEC NORMALIZATION (H.264 Conversion)
# =====================================================================

def prepare_video_input(video_path: str) -> str:
    """
    AV1 / VP9 / HEVC → 자동으로 H.264 변환.
    (원본 코드 그대로 유지)
    """
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", video_path
        ]
        probe_output = subprocess.check_output(cmd).decode("utf-8")
        info = json.loads(probe_output)

        v_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
        codec = v_stream.get("codec_name", "").lower()

        print(f">> Detected codec: {codec}")

    except Exception as e:
        print(f"⚠️ ffprobe failed: {e}")
        codec = "unknown"

    if codec in ["h264", "avc1"]:
        print(">> Video already H.264 — no conversion needed.")
        return video_path

    output_path = f"{video_path}.converted_{uuid.uuid4().hex}.mp4"
    print(f">> Converting to H.264: {output_path}")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "19",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(">> Conversion completed.")
        return output_path
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return video_path  # fallback


# =====================================================================
#                      QWEN INFERENCE FACTORY
# =====================================================================
def build_qwen_inference(model, tokenizer):

    def run_qwen_inference(images, prompt, max_tokens=256):

        messages = [{
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": prompt}
            ]
        }]

        # Apply chat template
        input_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True
        )

        # ORIGINAL QWEN3-VL TOKENIZER → TEXT + IMAGES IN ONE CALL
        inputs = tokenizer(
            images,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to("cuda")

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.2,
            do_sample=True,
            repetition_penalty=1.1
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]

        if "<|im_start|>assistant" in decoded:
            decoded = decoded.split("<|im_start|>assistant")[-1]

        decoded = decoded.replace("<|im_end|>", "").strip()

        return decoded

    return run_qwen_inference
# =====================================================================
#                      FULL PIPELINE
# =====================================================================

def run_full_pipeline():

    print("\n=========================================")
    print("🚀  STARTING FULL Ko-AD INFERENCE PIPELINE")
    print("===========================================\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 0) Video codec 정리 (H.264 변환)
    processed_video_path = prepare_video_input(VIDEO_PATH)

    # 1) Extract Audio
    extract_audio(processed_video_path, EXTRACTED_AUDIO_PATH)

    # 2) Whisper
    df_whisper = transcribe_audio(EXTRACTED_AUDIO_PATH, DEVICE=WHISPER_DEVICE)

    # Whisper 결과 저장
    df_whisper.to_json(WHISPER_OUTPUT, indent=2, force_ascii=False)
    print(f"📄 Whisper JSON saved → {WHISPER_OUTPUT}")

    # 3) Scene Detection
    scenes = detect_scenes_siglip(processed_video_path)

    # 4) Preprocess Whisper → Dialogue List
    dialogues = preprocess_dialogues(df_whisper)

    # 5) CHUNK GENERATION
    print("\n>> Generating chunks...")

    final_chunks = []
    scene_revised = []

    for sc in scenes:
        new_end = expand_scene_to_dialogues(sc["start"], sc["end"], dialogues)
        scene_revised.append({
            "id": sc["id"],
            "start": sc["start"],
            "end": new_end
        })

    for i in range(1, len(scene_revised)):
        scene_revised[i]["start"] = scene_revised[i - 1]["end"]

    for sc in scene_revised:
        subs = split_scene_by_dialogues(sc, dialogues)
        chunks = chunk_subscenes(subs)
        chunks = merge_last_chunk(chunks)
        final_chunks.extend(chunks)

    final_chunks = merge_too_short_chunks(final_chunks)
    refined_chunks = refine_long_chunks(final_chunks, dialogues)

    final_chunks = merge_too_short_chunks(refined_chunks, min_len=2.0, max_len=8.0)


    print(f"✓ Total chunks: {len(final_chunks)}")

    # 6) Qwen Model Load
    print("\n>> Loading Qwen-VL Model...")
    qwen_model, qwen_tokenizer = load_qwen_model(QWEN_MODEL_ID)
    run_qwen_inference = build_qwen_inference(qwen_model, qwen_tokenizer)

    # ---------------------------------------------------
    # 7) Visual Processing per Chunk  (원본 동일 구조로 저장)
    # ---------------------------------------------------

    print("\n>> Running Qwen Visual Stages...")

    qwen_results = []   # <-- 원본 JSON 구조 유지하는 리스트
    run_qwen_inference = build_qwen_inference(qwen_model, qwen_tokenizer)

    for idx, ch in enumerate(final_chunks):
        print(f"Chunk {idx+1}/{len(final_chunks)}")

        c_start = ch["start"]
        c_end = ch["end"]

        # -----------------------------
        # Whisper → dialogue map
        # -----------------------------
        dialogue_texts = df_whisper[
            (df_whisper["start"] >= c_start) & (df_whisper["start"] < c_end)
        ]["text"].tolist()

        dialogue_text = " ".join(dialogue_texts).strip()
        if not dialogue_text:
            dialogue_text = ""

        # -----------------------------
        # Frame sampling → PIL images
        # -----------------------------
        timestamps = get_shot_based_frames(
            processed_video_path,
            c_start,
            c_end,
            max_total_frames=MAX_FRAMES_PER_CHUNK
        )

        chunk_dir = f"chunk_frames/chunk_{idx:03d}"
        os.makedirs(chunk_dir, exist_ok=True)

        pil_images = []
        for i, t in enumerate(timestamps):
            out_path = os.path.join(chunk_dir, f"frame_{i:03d}.jpg")

            try:
                (ffmpeg.input(processed_video_path, ss=float(t))
                    .filter("scale", 768, -1)
                    .output(out_path, vframes=1)
                    .overwrite_output()
                    .run(quiet=True))

                if os.path.exists(out_path):
                    pil_images.append(Image.open(out_path).convert("RGB"))

            except Exception:
                continue
            
        if not pil_images:
            character_info = "Error: No valid frames"
            visual_analysis = "Error: No valid frames"
        else:
            # Stage 1
            character_info = qwen_stage1_identify(
                pil_images, run_qwen_inference
            )

            # Stage 2
            _, visual_analysis = qwen_stage2_analysis(
                pil_images, character_info, run_qwen_inference
            )

        # -------------------------
        # 원본 JSON 구조에 맞게 저장
        # -------------------------
        qwen_results.append({
            "start_time": f"{c_start:.2f}",
            "end_time": f"{c_end:.2f}",
            "dialogue": dialogue_text,
            "characters": character_info,
            "visual_analysis": visual_analysis
        })

    # ---------------------------------------------------
    # 8) Save QWEN-only JSON (원본과 동일한 형식)
    # ---------------------------------------------------
    with open(QWEN_OUTPUT_ONLY, "w", encoding="utf-8") as f:
        json.dump(qwen_results, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Qwen-only JSON saved to: {QWEN_OUTPUT_ONLY}\n")
    df_qwen = pd.DataFrame(qwen_results)


    # 9) QWEN 모델 메모리 해제
    del qwen_model
    del qwen_tokenizer
    del run_qwen_inference
    gc.collect()
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------
    # ------------------------- LANGUAGE STAGES -------------------------
    # -------------------------------------------------------------------

    print("\n>> Loading LLaMA Model...")
    llama_model, llama_tokenizer = load_llama_model(LLAMA_MODEL_NAME)

    print("\n>> Running Stage1: Visual Narrative Synthesis...")
    df_stage1 = run_stage1_visual_narrative(df_qwen, llama_model, llama_tokenizer)
    df_stage1.to_json(STAGE1_OUTPUT, indent=2, force_ascii=False)
    print(f"📄 Stage1 JSON saved → {STAGE1_OUTPUT}")

    print("\n>> Running Stage2: Character Tracking...")
    df_stage2 = run_stage2_character_tracking(df_stage1, llama_model, llama_tokenizer)
    df_stage2.to_json(STAGE2_OUTPUT, indent=2, force_ascii=False)
    print(f"📄 Stage2 JSON saved → {STAGE2_OUTPUT}")

    print("\n>> Running Stage3: Narrative Refinement...")
    df_stage3 = run_stage3_narrative_refine(df_stage2, llama_model, llama_tokenizer)
    df_stage3.to_json(STAGE3_OUTPUT, indent=2, force_ascii=False)
    print(f"📄 Stage3 JSON saved → {STAGE3_OUTPUT}")

    # 최종 저장
    df_stage3.to_json(FINAL_OUTPUT, indent=2, force_ascii=False)
    print(f"\n🎉 FINAL OUTPUT SAVED → {FINAL_OUTPUT}")

    return df_stage3


if __name__ == "__main__":
    run_full_pipeline()
