import sys
import os
import json
import numpy as np
from tqdm import tqdm

# Import Modules
import config
from audio.processing import extract_and_transcribe
from vision.detection import detect_scenes_siglip
from chunking.logic import create_final_chunks
from vision.qwen_model import load_model, run_inference
from vision.frame_extractor import get_shot_based_frames, extract_frames_for_chunk

from utils.gpu import clear_gpu_memory
from utils.video_utils import prepare_video_input
from language.loader import load_llama_model
from language.narrative import generate_narratives
from language.memory import run_memory_tracker
from language.refinement import run_narrative_refinement

def main():
    # 0. 초기화
    config.init_folders()
    print(f"Using device: {config.DEVICE}")

    # ============================================================
    # Part A: 시각/청각 정보 추출 (Qwen-VL & Whisper)
    # config.ONLY_RUN_PART_B가 False일 때만 실행
    # ============================================================
    
    if not config.ONLY_RUN_PART_B:
        # [추가된 로직] 비디오 코덱 확인 및 변환
        print(">> Checking video codec...")
        config.VIDEO_PATH = prepare_video_input(config.VIDEO_PATH)

        # 1. 오디오 처리
        df_whisper = extract_and_transcribe()
        if not df_whisper.empty:
            dialogues = df_whisper.sort_values(by="start")[['start','end','text']].to_dict('records')
        else:
            dialogues = []

        # 2. 비디오 씬 감지
        scene_fixed = detect_scenes_siglip()

        # 3. 청크 생성
        final_chunks = create_final_chunks(scene_fixed, dialogues)
        print(f"\n>> Total Chunks created: {len(final_chunks)}")

        # 4. 모델 로딩 (Qwen)
        model, tokenizer = load_model()
        if model is None:
            print("Exiting due to model load failure.")
            return

        # 5. 메인 추론 루프
        print(">> Starting 2-Stage Qwen-VL Inference...")
        final_results = []
        
        for idx, ch in enumerate(tqdm(final_chunks, desc="Processing Chunks")):
            c_start, c_end = ch["start"], ch["end"]
            folder = os.path.join(config.CHUNK_FRAMES_DIR, f"chunk_{idx:03d}")
            os.makedirs(folder, exist_ok=True)
            
            chunk_dialogues = ch.get("text", "")

            # 5-1. 프레임 추출
            ts_list = get_shot_based_frames(config.VIDEO_PATH, c_start, c_end)
            
            # Fallback logic
            if not ts_list:
                ts_list = np.arange(c_start, c_end, 0.5).tolist()
                if not ts_list: ts_list = [c_start]
                if len(ts_list) > 12:
                    indices = np.linspace(0, len(ts_list) - 1, 12, dtype=int)
                    ts_list = [ts_list[i] for i in indices]

            pil_images = extract_frames_for_chunk(config.VIDEO_PATH, ts_list, folder)

            if not pil_images:
                final_results.append({
                    "start_time": f"{round(c_start, 2):.2f}",
                    "end_time": f"{round(c_end, 2):.2f}",
                    "dialogue": chunk_dialogues,
                    "characters": "Error: No frames",
                    "visual_analysis": "Error: No frames"
                })
                continue

            try:
                # ==============================================================
                # Stage 1: 인물 식별
                # ==============================================================
                prompt_stage_1 = """
                Analyze these sequential images from a video clip.
                Your task is to catalogue distinct characters for a continuity tracking database.

                [INSTRUCTION]
                Identify up to 5 main characters. If there are more, group the rest as "Crowd".
                For each person, strictly separate **Invariant Traits** (Body/Face) from **Variant Traits** (Clothing).

                [OUTPUT FORMAT - STRICT]
                - Person 1:
                  - Visible: Image [start] to [end]
                  - Body/Face: [Gender, Age range, Hair color/style, Facial hair, Glasses, Height/Build]
                  - Attire: [Clothing type, Colors, Accessories]
                - Person 2:
                  ...

                [UNCERTAINTY RULES]
                1. If details are blurry, state "Unclear" or "Silhouette". Do NOT guess.
                2. If no distinct people are present, output "No distinct people found."
                """

                # 토큰 수를 조금 늘려서(300) 상세 묘사를 확보
                character_info = run_inference(model, tokenizer, pil_images, prompt_stage_1, max_tokens=300)

                keywords = ["Person", "Man", "Woman", "Boy", "Girl", "Crowd"]
                has_people = any(k in character_info for k in keywords) and "No distinct people" not in character_info

                final_characters = ""
                final_visual_analysis = ""

                if has_people:
                    final_characters = character_info
                    
                    # ==============================================================
                    # Stage 2-A: 행동 및 서사 분석
                    # ==============================================================
                    prompt_action = f"""
                    Based on the character list below:
                    ---
                    {character_info}
                    ---
                    Describe the VISUAL ACTION sequence chronologically.

                    [CRITICAL REQUIREMENT: IMAGE RANGES]
                    You MUST start every sentence with the specific **Image Range** where the action happens.
                    
                    [STRICT OUTPUT FORMAT]
                    - [Image Start-End]: [Who] [Action description].
                    
                    [Example]
                    - [Image 1-4]: Person 1 sits on the sofa and reads a book.
                    - [Image 5-8]: Person 1 puts the book down and looks at Person 2.
                    - [Image 9-12]: Person 2 enters the room and waves hand.

                    [RULES]
                    1. **Chronological**: Describe from the first image to the last.
                    2. **No Skipping**: Cover the entire sequence.
                    3. **Interaction**: Mention clearly if characters interact.
                    4. **Visuals Only**: Describe observable movements, not hidden thoughts.
                    """
                    
                    action_result = run_inference(model, tokenizer, pil_images, prompt_action, max_tokens=400)
                    final_visual_analysis = f"[Action Summary]:\n{action_result}"

                else:
                    final_characters = "None (Background Scene)"
                    
                    # ==============================================================
                    # Stage 2-B: 배경 분석
                    # ==============================================================
                    prompt_scenery = """
                    Since no main characters are focused, describe the SCENE details.
                    
                    [Output Format]
                    1. **Location/Setting**: (e.g., "Dark alleyway", "Bright office")
                    2. **Key Objects**: List major visible items.
                    3. **Cinematography**: Camera angle or lighting (e.g., "Close-up on a clock", "Wide shot of a city").
                    4. **Atmosphere**: One sentence on the mood (e.g., "Chaotic", "Serene").
                    """
                    
                    scenery_result = run_inference(model, tokenizer, pil_images, prompt_scenery, max_tokens=250)
                    final_visual_analysis = f"[Background Scene]:\n{scenery_result}"

            except Exception as e:
                final_characters = "Analysis Failed"
                final_visual_analysis = f"Inference Error: {e}"
                print(f"⚠️ Error in chunk {idx}: {e}")

            # 5-3. 결과 저장
            chunk_dialogues_text = " ".join([
                d['text'] for d in dialogues
                if max(ch['start'], d['start']) < min(ch['end'], d['end'])
            ])

            final_results.append({
                "start_time": f"{round(ch['start'], 2):.2f}",
                "end_time": f"{round(ch['end'], 2):.2f}",
                "dialogue": chunk_dialogues_text,
                "characters": final_characters,
                "visual_analysis": final_visual_analysis
            })

            # 6. JSON 실시간 저장 (안전장치: 루프 안에서 저장)
            with open(config.OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)
            
        print(f"\n🎉 Saving final JSON to: {config.OUTPUT_JSON_PATH}")

        # Part A 완료 후 Qwen 메모리 해제 (⚠️ 중요: for 루프 밖으로 뺐습니다)
        import gc
        import torch
        
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect() 
        torch.cuda.empty_cache()

        print(f"✅ GPU Memory Cleared after Part A. Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    else:
        # Part A가 스킵되었을 경우
        print("\n⏩ Skipped Part A (Qwen). Starting directly from Part B (Llama).")
        # Part B 시작 전 입력 파일 확인
        if not os.path.exists(config.OUTPUT_JSON_PATH):
            print(f"❌ Error: Input file '{config.OUTPUT_JSON_PATH}' not found!")
            print("💡 Hint: You must run Part A at least once (set ONLY_RUN_PART_B = False in config.py).")
            return

    # ============================================================
    # Part B: 서사 및 연속성 추적 (Llama-3.1-70B)
    # ============================================================
    print("\n" + "="*50)
    print(">> Transitioning to Part B: Language Processing")
    print("="*50)
    
    # 설정된 시작 단계 확인
    start_stage = getattr(config, 'PART_B_START_STAGE', 1)
    print(f"ℹ️ Start Configuration: Stage {start_stage} ~ Stage 3")

    # 8. Llama 모델 로드 (Part B 전체에서 공유)
    llama_model, llama_tokenizer = load_llama_model()

    # ------------------------------------------------------------
    # [Stage 1] 서사(Narrative) 생성
    # ------------------------------------------------------------
    if start_stage <= 1:
        if not os.path.exists(config.OUTPUT_JSON_PATH):
            print(f"❌ Error: Input file for Stage 1 not found: {config.OUTPUT_JSON_PATH}")
            return
        print("\n>> [Stage 1] Running Narrative Generation...")
        generate_narratives(llama_model, llama_tokenizer, config.OUTPUT_JSON_PATH)
    else:
        print("\n⏩ [Stage 1] Skipped (User Request).")

    # ------------------------------------------------------------
    # [Stage 2] 기억 추적 (Memory Tracker)
    # ------------------------------------------------------------
    if start_stage <= 2:
        if not os.path.exists(config.STAGE1_OUTPUT_JSON):
            print(f"❌ Error: Input file for Stage 2 not found: {config.STAGE1_OUTPUT_JSON}")
            print("💡 Hint: You skipped Stage 1, but its output file does not exist.")
            return
        print("\n>> [Stage 2] Running Memory Tracker...")
        run_memory_tracker(llama_model, llama_tokenizer, config.STAGE1_OUTPUT_JSON)
    else:
        print("\n⏩ [Stage 2] Skipped (User Request).")

    # ------------------------------------------------------------
    # [Stage 3] 서사 다듬기 (Refinement)
    # ------------------------------------------------------------
    if start_stage <= 3:
        if not os.path.exists(config.STAGE2_OUTPUT_JSON):
            print(f"❌ Error: Input file for Stage 3 not found: {config.STAGE2_OUTPUT_JSON}")
            return
        print("\n>> [Stage 3] Running Narrative Refinement...")
        run_narrative_refinement(llama_model, llama_tokenizer, config.STAGE2_OUTPUT_JSON)

    # 12. Llama 메모리 해제 (최종 완료)
    clear_gpu_memory(llama_model, llama_tokenizer)
    
    print("\n✅ All Stages Completed Successfully! Final output is in:", config.STAGE3_OUTPUT_JSON)

if __name__ == "__main__":
    main()