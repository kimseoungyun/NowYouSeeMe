#!/usr/bin/env python3
"""
inference_server.py
===================
Inference 파이프라인을 FastAPI 서버로 제공합니다.
Docker 백엔드에서 HTTP로 호출할 수 있습니다.

Usage:
    cd /mnt/Ko-AD/inference
    source venv/bin/activate
    uvicorn inference_server:app --host 0.0.0.0 --port 5000

또는:
    python inference_server.py
"""

import os
import sys
import json
import asyncio
import shutil
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

# FastAPI imports
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    print("FastAPI not installed. Installing...")
    os.system(f"{sys.executable} -m pip install fastapi uvicorn")
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

# 현재 디렉토리를 sys.path에 추가
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# 번역용
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False
    print("⚠️ deep_translator not installed. Translation will be disabled.")


# =============================================================================
# Pydantic Models
# =============================================================================

class InferenceRequest(BaseModel):
    video_path: str
    output_dir: Optional[str] = None
    lang: str = "en"  # 'en' or 'ko'
    video_id: Optional[str] = None


class InferenceResponse(BaseModel):
    success: bool
    output_path: Optional[str] = None
    segments: Optional[list] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None


# =============================================================================
# Helper Functions
# =============================================================================

def check_gpu():
    """GPU 상태 확인"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3)
            }
    except:
        pass
    return {"available": False, "name": None, "memory_gb": None}


def translate_to_korean(text: str) -> str:
    """영어 텍스트를 한국어로 번역"""
    if not TRANSLATOR_AVAILABLE or not text:
        return text
    try:
        translator = GoogleTranslator(source='en', target='ko')
        return translator.translate(text)
    except Exception as e:
        print(f"⚠️ Translation error: {e}")
        return text


def convert_to_tts_format(stage3_data: list, video_path: str, lang: str = 'en') -> dict:
    """stage3 결과를 TTS 호환 형식으로 변환"""
    audio_descriptions = []
    
    for idx, item in enumerate(stage3_data):
        start_time = float(item.get('start_time', 0))
        end_time = float(item.get('end_time', 0))
        duration = end_time - start_time
        
        description = item.get('final_narrative', '') or item.get('stage1_result', '')
        
        if not description:
            continue
        
        # 한국어 번역
        if lang == 'ko':
            description = translate_to_korean(description)
            
        audio_descriptions.append({
            "id": idx + 1,
            "original_id": idx + 1,
            "start_time": start_time,
            "end_time": end_time,
            "duration_sec": round(duration, 3),
            "description": description,
            "dialogue": item.get('dialogue', ''),
            "is_split": False,
            "split_info": None
        })
    
    return {
        "video_info": {
            "source_file": os.path.basename(video_path),
            "language": lang,
            "total_segments": len(audio_descriptions)
        },
        "audio_descriptions": audio_descriptions
    }


def run_inference_pipeline(video_path: str, output_dir: str, lang: str = 'en', video_id: str = None) -> dict:
    """
    Inference 파이프라인 실행
    """
    import subprocess
    import uuid
    import time
    
    # 비디오 경로 검증
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # output_dir 설정 - video_id별로 서브디렉토리 생성
    if not output_dir:
        output_dir = os.path.join(SCRIPT_DIR, "output")
    
    # video_id가 없으면 UUID 생성
    if not video_id:
        video_id = str(uuid.uuid4())[:8]
    
    # video_id별 고유 폴더 생성 (덮어쓰기 방지)
    job_output_dir = os.path.join(output_dir, f"inference_{video_id}")
    
    # 기존 결과가 있으면 삭제 (새로운 요청이므로)
    if os.path.exists(job_output_dir):
        print(f"🗑️ Removing existing output: {job_output_dir}")
        shutil.rmtree(job_output_dir)
    
    os.makedirs(job_output_dir, exist_ok=True)
    print(f"📁 Output directory: {job_output_dir}")
    
    # 비디오 파일을 inference 폴더로 심볼릭 링크 생성
    video_basename = os.path.basename(video_path)
    inference_video_path = os.path.join(SCRIPT_DIR, video_basename)
    
    link_created = False
    # 기존 링크가 있으면 삭제
    if os.path.exists(inference_video_path) or os.path.islink(inference_video_path):
        if os.path.islink(inference_video_path):
            os.unlink(inference_video_path)
        elif video_path != inference_video_path:
            os.remove(inference_video_path)
    
    if video_path != inference_video_path:
        try:
            os.symlink(video_path, inference_video_path)
            link_created = True
        except OSError:
            shutil.copy2(video_path, inference_video_path)
            link_created = True
    
    try:
        # config를 동적으로 설정하는 wrapper 스크립트
        wrapper_script = f'''
import sys
import os
sys.path.insert(0, "{SCRIPT_DIR}")
os.chdir("{SCRIPT_DIR}")

import config
config.VIDEO_PATH = "{inference_video_path}"
config.OUTPUT_DIR = "{job_output_dir}"
config.OUTPUT_JSON_PATH = os.path.join(config.OUTPUT_DIR, "final_ad_creation_output.json")
config.STAGE1_OUTPUT_JSON = os.path.join(config.OUTPUT_DIR, "stage1_narrative_output.json")
config.STAGE2_OUTPUT_JSON = os.path.join(config.OUTPUT_DIR, "stage2_final_output.json")
config.STAGE2_LOG_FILE = os.path.join(config.OUTPUT_DIR, "stage2_final_log.jsonl")
config.STAGE3_OUTPUT_JSON = os.path.join(config.OUTPUT_DIR, "stage3_final_narrative_refined.json")

from main import main
main()
'''
        
        # venv Python 경로
        venv_python = os.path.join(SCRIPT_DIR, "venv", "bin", "python")
        if not os.path.exists(venv_python):
            venv_python = sys.executable
        
        # 환경 변수 설정
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '0')
        env['PYTHONUNBUFFERED'] = '1'
        
        # subprocess로 실행
        process = subprocess.run(
            [venv_python, '-c', wrapper_script],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            env=env
        )
        
        if process.returncode != 0:
            raise RuntimeError(f"Inference failed: {process.stderr}")
        
        # 결과 파일 로드 (job_output_dir에서)
        stage3_path = os.path.join(job_output_dir, "stage3_final_narrative_refined.json")
        
        # Fallback 파일들
        fallback_files = [
            "stage2_final_output.json",
            "stage1_narrative_output.json", 
            "final_ad_creation_output.json"
        ]
        
        if not os.path.exists(stage3_path):
            for fallback in fallback_files:
                fallback_path = os.path.join(job_output_dir, fallback)
                if os.path.exists(fallback_path):
                    stage3_path = fallback_path
                    break
        
        if not os.path.exists(stage3_path):
            raise FileNotFoundError(f"No output JSON found in {job_output_dir}")
        
        with open(stage3_path, 'r', encoding='utf-8') as f:
            stage3_data = json.load(f)
        
        # TTS 호환 형식으로 변환
        tts_result = convert_to_tts_format(stage3_data, video_path, lang)
        
        # 최종 JSON 저장 (상위 output_dir에 저장 - 기존 시스템 호환)
        output_filename = f"{video_id}_{lang}.ad.json"
        
        tts_output_path = os.path.join(output_dir, output_filename)
        
        with open(tts_output_path, 'w', encoding='utf-8') as f:
            json.dump(tts_result, f, ensure_ascii=False, indent=2)
        
        # 세그먼트 형식 변환
        segments = [
            {
                'id': seg['id'],
                'start': seg['start_time'],
                'end': seg['end_time'],
                'text': seg['description']
            }
            for seg in tts_result['audio_descriptions']
        ]
        
        return {
            "success": True,
            "output_path": tts_output_path,
            "segments": segments
        }
        
    finally:
        # 심볼릭 링크 정리
        if link_created and os.path.islink(inference_video_path):
            os.unlink(inference_video_path)
        elif link_created and os.path.isfile(inference_video_path):
            os.remove(inference_video_path)


# =============================================================================
# FastAPI App
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 시작/종료 시 실행"""
    print("🚀 Inference Server Starting...")
    gpu_info = check_gpu()
    if gpu_info["available"]:
        print(f"✅ GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.1f} GB)")
    else:
        print("⚠️ GPU not available, using CPU")
    yield
    print("🛑 Inference Server Shutting Down...")


app = FastAPI(
    title="Ko-AD Inference API",
    description="Qwen-VL + Llama-70B 기반 화면해설 생성 API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """서버 상태 확인"""
    gpu_info = check_gpu()
    return HealthResponse(
        status="ok",
        gpu_available=gpu_info["available"],
        gpu_name=gpu_info["name"],
        gpu_memory_gb=gpu_info["memory_gb"]
    )


@app.post("/generate-ad", response_model=InferenceResponse)
async def generate_ad(request: InferenceRequest):
    """
    화면해설 생성 API
    
    - video_path: 비디오 파일 경로 (절대 경로)
    - output_dir: 출력 디렉토리 (옵션)
    - lang: 출력 언어 ('en' 또는 'ko')
    - video_id: 비디오 ID (파일명에 사용)
    """
    try:
        result = await asyncio.to_thread(
            run_inference_pipeline,
            request.video_path,
            request.output_dir,
            request.lang,
            request.video_id
        )
        return InferenceResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"❌ Error: {error_detail}")
        return InferenceResponse(
            success=False,
            error=str(e)
        )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("🔥 Ko-AD Inference Server")
    print("=" * 50)
    print(f"📁 Script Dir: {SCRIPT_DIR}")
    print(f"🐍 Python: {sys.executable}")
    
    gpu_info = check_gpu()
    if gpu_info["available"]:
        print(f"🎮 GPU: {gpu_info['name']} ({gpu_info['memory_gb']:.1f} GB)")
    else:
        print("⚠️ GPU: Not available")
    
    print("=" * 50)
    print("Starting server on http://0.0.0.0:8001")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8001)

