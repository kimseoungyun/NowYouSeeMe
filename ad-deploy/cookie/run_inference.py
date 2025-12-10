#!/usr/bin/env python3
"""
run_inference.py
================
inference 파이프라인 wrapper 스크립트.
inference 폴더의 기존 main.py를 subprocess로 호출하고,
결과를 TTS 호환 형식으로 변환합니다.

inference 폴더의 기존 파일들은 전혀 수정하지 않습니다.

Usage:
    python run_inference.py --video_path /path/to/video.mp4 --output_dir /path/to/output --lang ko
    
Output:
    TTS 호환 JSON 파일 (audio_descriptions 형식)
"""

import sys
import os
import json
import argparse
import subprocess
import shutil
from pathlib import Path

# inference 폴더 경로
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
INFERENCE_DIR = os.path.join(PROJECT_ROOT, "inference")
INFERENCE_VENV_PYTHON = os.path.join(INFERENCE_DIR, "venv", "bin", "python")
INFERENCE_MAIN = os.path.join(INFERENCE_DIR, "main.py")


def translate_to_korean_simple(text):
    """
    간단한 번역 함수 (Google Translate API 사용).
    실제 프로덕션에서는 더 나은 번역 API를 사용하세요.
    """
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='en', target='ko')
        return translator.translate(text)
    except ImportError:
        print("⚠️ deep_translator not installed. Install with: pip install deep-translator")
        return text
    except Exception as e:
        print(f"⚠️ Translation error: {e}")
        return text


def convert_to_tts_format(stage3_data, video_path, lang='en'):
    """
    stage3 결과를 TTS 호환 형식 (audio_descriptions)으로 변환합니다.
    
    Input format (stage3):
    [
        {
            "start_time": 0.0,
            "end_time": 3.94,
            "dialogue": "Hello, Walter.",
            "final_narrative": "Man A stands near the front car..."
        }
    ]
    
    Output format (TTS compatible):
    {
        "video_info": {...},
        "audio_descriptions": [
            {
                "id": 1,
                "start_time": 0.0,
                "end_time": 3.94,
                "duration_sec": 3.94,
                "description": "Man A stands near the front car..."
            }
        ]
    }
    """
    audio_descriptions = []
    
    for idx, item in enumerate(stage3_data):
        start_time = float(item.get('start_time', 0))
        end_time = float(item.get('end_time', 0))
        duration = end_time - start_time
        
        # final_narrative 또는 stage1_result 사용
        description = item.get('final_narrative', '') or item.get('stage1_result', '')
        
        if not description:
            continue
        
        # 한국어 번역 (옵션)
        if lang == 'ko':
            description = translate_to_korean_simple(description)
            
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


def run_inference_pipeline(video_path, output_dir=None, lang='en', video_id=None):
    """
    inference 파이프라인을 실행합니다.
    
    Args:
        video_path: 입력 비디오 경로 (절대 경로)
        output_dir: 출력 디렉토리
        lang: 출력 언어 ('en' 또는 'ko')
        video_id: 비디오 ID (파일명 생성에 사용)
    
    Returns:
        dict: TTS 호환 형식의 결과
    """
    
    # 비디오 경로를 절대 경로로 변환
    video_path = os.path.abspath(video_path)
    
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # output_dir이 지정되지 않으면 inference/output 사용
    if output_dir is None:
        output_dir = os.path.join(INFERENCE_DIR, "output")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f">> Inference Pipeline Starting...")
    print(f"   Video: {video_path}")
    print(f"   Output: {output_dir}")
    print(f"   Language: {lang}")
    
    # =====================================================
    # Step 1: inference/main.py 실행
    # =====================================================
    # inference 폴더의 config.py를 임시로 수정하지 않고,
    # 환경 변수로 비디오 경로를 전달하는 방식은 지원되지 않으므로,
    # 비디오 파일을 inference 폴더로 복사하거나 심볼릭 링크를 생성합니다.
    
    # 비디오 파일이 inference 폴더에 없으면 심볼릭 링크 생성
    video_basename = os.path.basename(video_path)
    inference_video_path = os.path.join(INFERENCE_DIR, video_basename)
    
    link_created = False
    if video_path != inference_video_path and not os.path.exists(inference_video_path):
        try:
            os.symlink(video_path, inference_video_path)
            link_created = True
            print(f"   Created symlink: {inference_video_path} -> {video_path}")
        except OSError as e:
            # 심볼릭 링크 실패 시 복사
            print(f"   Symlink failed, copying video file...")
            shutil.copy2(video_path, inference_video_path)
            link_created = True
    
    # config.py의 VIDEO_PATH를 임시로 수정하는 대신,
    # 비디오 파일 이름이 일치하도록 처리
    # (기존 config.py의 VIDEO_PATH와 다른 경우)
    
    # Python 스크립트 실행 (inference 폴더에서)
    # main.py는 config.py의 VIDEO_PATH를 사용하므로,
    # 별도 래퍼 스크립트를 만들어 config를 동적으로 설정
    
    wrapper_script = f'''
import sys
import os
sys.path.insert(0, "{INFERENCE_DIR}")
os.chdir("{INFERENCE_DIR}")

# config 모듈의 VIDEO_PATH를 동적으로 변경
import config
config.VIDEO_PATH = "{inference_video_path}"
config.OUTPUT_DIR = "{output_dir}"
config.OUTPUT_JSON_PATH = os.path.join(config.OUTPUT_DIR, "final_ad_creation_output.json")
config.STAGE1_OUTPUT_JSON = os.path.join(config.OUTPUT_DIR, "stage1_narrative_output.json")
config.STAGE2_OUTPUT_JSON = os.path.join(config.OUTPUT_DIR, "stage2_final_output.json")
config.STAGE2_LOG_FILE = os.path.join(config.OUTPUT_DIR, "stage2_final_log.jsonl")
config.STAGE3_OUTPUT_JSON = os.path.join(config.OUTPUT_DIR, "stage3_final_narrative_refined.json")

# main 모듈 실행
from main import main
main()
'''
    
    print(f"\n>> Running inference pipeline...")
    
    # CUDA 환경 변수를 명시적으로 전달
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = env.get('CUDA_VISIBLE_DEVICES', '0')
    env['PYTHONUNBUFFERED'] = '1'
    
    # subprocess로 실행
    process = subprocess.Popen(
        [INFERENCE_VENV_PYTHON, '-c', wrapper_script],
        cwd=INFERENCE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env
    )
    
    # 실시간으로 출력 표시
    stdout_lines = []
    stderr_lines = []
    
    # stderr를 실시간으로 출력 (진행 상황)
    for line in process.stderr:
        print(line, end='')
        stderr_lines.append(line)
    
    # stdout 읽기
    stdout, _ = process.communicate()
    stdout_lines.append(stdout)
    
    return_code = process.returncode
    
    # 심볼릭 링크 정리
    if link_created and os.path.islink(inference_video_path):
        os.unlink(inference_video_path)
        print(f"   Removed symlink: {inference_video_path}")
    
    if return_code != 0:
        error_msg = ''.join(stderr_lines)
        raise RuntimeError(f"Inference pipeline failed (exit code {return_code}):\n{error_msg}")
    
    # =====================================================
    # Step 2: 결과 JSON 로드 및 변환
    # =====================================================
    stage3_path = os.path.join(output_dir, "stage3_final_narrative_refined.json")
    
    if not os.path.exists(stage3_path):
        # stage3가 없으면 stage2나 stage1 결과 사용
        for fallback in ["stage2_final_output.json", "stage1_narrative_output.json", "final_ad_creation_output.json"]:
            fallback_path = os.path.join(output_dir, fallback)
            if os.path.exists(fallback_path):
                stage3_path = fallback_path
                print(f"   Using fallback: {fallback}")
                break
    
    if not os.path.exists(stage3_path):
        raise FileNotFoundError(f"No output JSON found in {output_dir}")
    
    with open(stage3_path, 'r', encoding='utf-8') as f:
        stage3_data = json.load(f)
    
    print(f"\n>> Converting to TTS format...")
    tts_result = convert_to_tts_format(stage3_data, video_path, lang)
    
    # =====================================================
    # Step 3: 최종 JSON 저장
    # =====================================================
    if video_id:
        output_filename = f"{video_id}_{lang}.ad.json"
    else:
        video_basename_no_ext = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{video_basename_no_ext}_{lang}.ad.json"
    
    tts_output_path = os.path.join(output_dir, output_filename)
    
    with open(tts_output_path, 'w', encoding='utf-8') as f:
        json.dump(tts_result, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Final TTS-compatible JSON saved to: {tts_output_path}")
    print(f"✅ Total audio descriptions: {len(tts_result['audio_descriptions'])}")
    
    return tts_result, tts_output_path


def main():
    parser = argparse.ArgumentParser(description='Run inference pipeline for audio description generation')
    parser.add_argument('--video_path', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'ko'], help='Output language (en or ko)')
    parser.add_argument('--video_id', type=str, default=None, help='Video ID for output filename')
    parser.add_argument('--json_only', action='store_true', help='Output JSON result to stdout (for API integration)')
    
    args = parser.parse_args()
    
    try:
        result, output_path = run_inference_pipeline(
            video_path=args.video_path,
            output_dir=args.output_dir,
            lang=args.lang,
            video_id=args.video_id
        )
        
        if args.json_only:
            # API 통합을 위해 JSON만 stdout으로 출력
            print(json.dumps({
                'success': True,
                'output_path': output_path,
                'segments': [
                    {
                        'id': seg['id'],
                        'start': seg['start_time'],
                        'end': seg['end_time'],
                        'text': seg['description']
                    }
                    for seg in result['audio_descriptions']
                ]
            }))
        else:
            print("\n" + "="*50)
            print("🎉 Pipeline completed successfully!")
            print("="*50)
            
    except Exception as e:
        import traceback
        error_info = {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        if args.json_only:
            print(json.dumps(error_info))
        else:
            print(f"\n❌ Pipeline failed: {e}")
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

