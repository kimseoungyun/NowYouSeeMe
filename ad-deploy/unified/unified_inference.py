#!/usr/bin/env python3
"""
통합 인퍼런스 엔트리포인트
===========================
단일 CLI로 각 AD 생성 모델(Gemini, Jack, GPT/cookie)을 실행하고,
선택한 TTS 엔진으로 최종 오디오/영상까지 생성합니다.

입력:
    - --model         : gemini | jack | gpt | gpi (gpt alias)
    - --video_path    : 대상 영상 경로
    - --lang          : ko | en
    - --voice_profile : gtts | kor_male | kor_female | eng_male | eng_female |
                        gemini_kor_female | gemini_kor_male |
                        gemini_eng_female | gemini_eng_male
    - --gemini_api_key / --openai_api_key : 필요시 키 주입

출력:
    - AD JSON 파일 (audio_descriptions 형식)
    - --skip_tts 를 지정하지 않으면 TTS가 입혀진 mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from extract_for_gpt import process_video_for_gpt
from gemini_json_tts import synthesize_tts_for_video
from get_AD_gemini import generate_ad_for_video as gemini_generate_ad
from get_AD_gemini import save_ad_json as gemini_save_json
from get_AD_gpt import generate_ad_from_extracted_data as gpt_generate_ad
from get_AD_jack import generate_ad_for_video as jack_generate_ad
from get_AD_jack import save_ad_json as jack_save_json

LOGGER = logging.getLogger("unified_inference")

VOICE_PROFILES = [
    "gtts",
    "kor_male",
    "kor_female",
    "eng_male",
    "eng_female",
    "gemini_kor_female",
    "gemini_kor_male",
    "gemini_eng_female",
    "gemini_eng_male",
]

MODEL_CHOICES = ["gemini", "jack", "gpt", "cookie"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_gemini(video_path: Path, lang: str, api_key: str, output_dir: Path) -> Tuple[Path, List[Dict]]:
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    full, segments = gemini_generate_ad(str(video_path), api_key=api_key, lang=lang)
    video_id = video_path.stem
    ad_path = Path(gemini_save_json(f"{video_id}_{lang}", full, str(output_dir)))
    return ad_path, segments


def run_jack(video_path: Path, lang: str, api_key: str, output_dir: Path) -> Tuple[Path, List[Dict]]:
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    full, segments = jack_generate_ad(str(video_path), lang=lang)
    video_id = video_path.stem
    ad_path = Path(jack_save_json(f"{video_id}_{lang}", full, str(output_dir)))
    return ad_path, segments


def run_gpt(
    video_path: Path,
    lang: str,
    api_key: str,
    output_dir: Path,
    fps: float,
    whisper_model: str,
    min_silence: float,
) -> Tuple[Path, List[Dict]]:
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    base_dir = ensure_dir(output_dir / "gpt_data")
    # 1) 전처리 (VAD + Whisper + 프레임 추출)
    process_video_for_gpt(
        input_video=str(video_path),
        output_dir=str(base_dir),
        fps=fps,
        language=lang,
        whisper_model=whisper_model,
        min_silence_duration=min_silence,
    )
    data_dir = base_dir / video_path.stem
    # 2) GPT 모델로 AD 생성
    result = gpt_generate_ad(
        data_dir=str(data_dir),
        api_key=api_key,
        language=lang,
        min_duration=min_silence,
    )
    ad_path = data_dir / f"{data_dir.name}.ad.json"
    segments = result.get("audio_descriptions", [])
    return ad_path, segments


def run_cookie(video_path: Path, lang: str, output_dir: Path) -> Tuple[Path, List[Dict]]:
    """
    cookie 모델: 기존 run_inference.py + inference 파이프라인 실행
    """
    root_dir = Path(__file__).parent.parent  # .../github
    cookie_dir = root_dir / "cookie"
    cookie_run = cookie_dir / "run_inference.py"
    cookie_inference = cookie_dir / "inference"
    shared_inference = root_dir / "inference"

    if not cookie_run.exists():
        raise FileNotFoundError(f"cookie run_inference.py not found: {cookie_run}")
    if not cookie_inference.exists():
        raise FileNotFoundError(f"cookie inference dir not found: {cookie_inference}")

    # run_inference.py는 PROJECT_ROOT/INFERENCE_DIR를 고정 경로로 계산하므로
    # github/inference 위치에 실제 파이프라인을 만들어줌 (없을 때만)
    if not shared_inference.exists():
        try:
            os.symlink(cookie_inference, shared_inference, target_is_directory=True)
        except OSError:
            shutil.copytree(cookie_inference, shared_inference, dirs_exist_ok=True)

    cmd = [
        sys.executable or "python",
        str(cookie_run),
        "--video_path",
        str(video_path),
        "--output_dir",
        str(output_dir),
        "--lang",
        lang,
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(cookie_dir),
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )

    if proc.returncode != 0:
        raise RuntimeError(f"cookie model failed: {proc.stderr or proc.stdout}")

    ad_filename = f"{video_path.stem}_{lang}.ad.json"
    ad_path = output_dir / ad_filename
    if not ad_path.exists():
        raise FileNotFoundError(f"AD JSON not found at {ad_path}")

    with ad_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("audio_descriptions", [])
    return ad_path, segments


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="통합 AD 인퍼런스 + TTS 실행기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, choices=MODEL_CHOICES, help="AD 생성 모델 선택")
    parser.add_argument("--video_path", required=True, help="입력 비디오 경로")
    parser.add_argument("--lang", default="ko", choices=["ko", "en"], help="AD/TTS 언어")
    parser.add_argument("--output_dir", default="./outputs", help="결과 저장 디렉토리")
    parser.add_argument("--voice_profile", default="gtts", choices=VOICE_PROFILES, help="TTS 음성 프로필")
    parser.add_argument("--gemini_api_key", default=None, help="Gemini API 키 (Gemini/Jack/성우 Gemini TTS)")
    parser.add_argument("--openai_api_key", default=None, help="OpenAI API 키 (GPT/gpi 모델)")
    parser.add_argument("--skip_tts", action="store_true", help="TTS/영상 합성 건너뛰고 AD JSON만 생성")
    parser.add_argument("--disable_ducking", action="store_true", help="VAD 기반 더킹 비활성화")
    parser.add_argument("--fps", type=float, default=2.0, help="[gpt] 프레임 추출 FPS")
    parser.add_argument("--whisper_model", default="small", choices=["tiny", "base", "small", "medium", "large"], help="[gpt] Whisper 모델")
    parser.add_argument("--min_silence", type=float, default=2.5, help="[gpt] 무음 구간 최소 길이(초)")
    parser.add_argument("--verbose", action="store_true", help="디버그 로그 출력")
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    video_path = Path(args.video_path).expanduser().resolve()
    output_dir = ensure_dir(Path(args.output_dir).expanduser().resolve())

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    LOGGER.info("=== 통합 인퍼런스 시작 ===")
    LOGGER.info("모델=%s | 언어=%s | 영상=%s", args.model, args.lang, video_path)

    model = args.model.lower()
    ad_json_path: Path
    segments: List[Dict]

    if model == "cookie":
        ad_json_path, segments = run_cookie(
            video_path=video_path,
            lang=args.lang,
            output_dir=output_dir,
        )
    elif model == "gpt":
        if not (args.openai_api_key or os.environ.get("OPENAI_API_KEY")):
            raise ValueError("GPT 모델 사용 시 OPENAI_API_KEY 또는 --openai_api_key 가 필요합니다.")
        ad_json_path, segments = run_gpt(
            video_path=video_path,
            lang=args.lang,
            api_key=args.openai_api_key,
            output_dir=output_dir,
            fps=args.fps,
            whisper_model=args.whisper_model,
            min_silence=args.min_silence,
        )
    elif model == "gemini":
        if not (args.gemini_api_key or os.environ.get("GEMINI_API_KEY")):
            raise ValueError("Gemini 모델 사용 시 GEMINI_API_KEY 또는 --gemini_api_key 가 필요합니다.")
        ad_json_path, segments = run_gemini(
            video_path=video_path,
            lang=args.lang,
            api_key=args.gemini_api_key,
            output_dir=output_dir,
        )
    elif model == "jack":
        if not (args.gemini_api_key or os.environ.get("GEMINI_API_KEY")):
            raise ValueError("Jack 모델 사용 시 GEMINI_API_KEY 또는 --gemini_api_key 가 필요합니다.")
        ad_json_path, segments = run_jack(
            video_path=video_path,
            lang=args.lang,
            api_key=args.gemini_api_key,
            output_dir=output_dir,
        )
    else:
        raise ValueError(f"지원하지 않는 모델: {args.model}")

    summary: Dict[str, object] = {
        "success": True,
        "model": model,
        "lang": args.lang,
        "video_path": str(video_path),
        "ad_json_path": str(ad_json_path),
        "segments": segments,
    }

    LOGGER.info("AD JSON 생성 완료: %s", ad_json_path)

    if args.skip_tts:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    LOGGER.info("TTS/영상 합성 시작 (voice=%s)...", args.voice_profile)
    tts_result = synthesize_tts_for_video(
        video_id=video_path.stem,
        video_path=str(video_path),
        ad_json_path=str(ad_json_path),
        output_dir=str(output_dir),
        lang=args.lang,
        voice_profile=args.voice_profile,
        enable_ducking=not args.disable_ducking,
        gemini_api_key=args.gemini_api_key,
    )

    summary.update(
        {
            "tts": tts_result,
            "finalAudioPath": tts_result.get("finalAudioPath"),
            "finalVideoPath": tts_result.get("finalVideoPath"),
        }
    )

    LOGGER.info("✅ 완료! 최종 영상: %s", tts_result.get("finalVideoPath"))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

