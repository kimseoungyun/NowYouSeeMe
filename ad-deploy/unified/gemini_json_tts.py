# -*- coding: utf-8 -*-
"""
TTS Synthesis Module for Audio Description

This module synthesizes TTS audio from AD JSON segments and mixes them with the original video.
It supports gTTS (basic), Coqui XTTS v2 (advanced 4 voice profiles), and Gemini TTS (Google AI).
Includes VAD-based ducking to reduce AD volume when overlapping with dialogue.

It can be used as a CLI tool or imported as a module.
"""

import os
import io
import json
import logging
import wave
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from pydub import AudioSegment
from gtts import gTTS
import ffmpeg
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

MIN_DESC_LEN = 2.5  # Minimum description length in seconds

# === Ducking Hyperparameters ===
DIALOGUE_LUFS_MARGIN = 2.0  # AD를 배경 발화보다 몇 LUFS 작게 할지 (2~4 LUFS 권장)
AD_OVERLAP_EXTRA_ATTENUATION_DB = -3.0  # 대사와 겹치는 구간일 때 AD를 추가로 더 줄이는 양 (dB, 음수)
AD_OVERFLOW_ATTENUATION_DB = -6.0  # AD가 end_time을 넘어갈 때 overflow 부분의 감쇄량 (dB, 음수)

# === TTS Voice Profiles ===
# voice_profile: 'gtts' | 'kor_male' | 'kor_female' | 'eng_male' | 'eng_female'
#                | 'gemini_kor_male' | 'gemini_kor_female' | 'gemini_eng_male' | 'gemini_eng_female'
TTS_VOICE_PROFILES = {
    "kor_male": {
        "language": "ko",
        "speaker_wav": "tts_voice/cut_5min_kor_male.mp3",
        "speed": 1.05,
    },
    "kor_female": {
        "language": "ko",
        "speaker_wav": "tts_voice/cut_5min_kor_female.mp3",
        "speed": 1.05,
    },
    "eng_male": {
        "language": "en",
        "speaker_wav": "tts_voice/cut_5min_eng_male.mp3",
        "speed": 1.05,
    },
    "eng_female": {
        "language": "en",
        "speaker_wav": "tts_voice/cut_5min_eng_female.mp3",
        "speed": 1.05,
    },
}

# === Gemini TTS Voice Profiles ===
# Gemini TTS uses Google's AI voices via the genai SDK
# Available voices: Puck (male), Charon (male), Kore (female), Aoede (female), Fenrir (male), etc.
# Reference: https://docs.cloud.google.com/text-to-speech/docs/gemini-tts?hl=ko#voice_options
GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"

# 화면해설(Audio Description)에 최적화된 스타일 프롬프트
# - 차분하고 명확한 톤: 시청자가 영상에 집중할 수 있도록
# - 중립적이고 객관적인 어조: 감정을 과하게 표현하지 않음
# - 전문적인 내레이션 스타일: 다큐멘터리 나레이터처럼
# - 속도 1.5배: [extremely fast] 태그 사용
GEMINI_TTS_AD_PROMPT_KO = """다음 화면해설을 차분하고 명확한 톤으로 빠르게 읽어주세요.
- 전문적인 다큐멘터리 나레이터처럼 중립적이고 객관적인 어조로 말해주세요.
- 감정을 과하게 표현하지 말고, 담담하게 사실을 전달하세요.
[extremely fast] """

GEMINI_TTS_AD_PROMPT_EN = """Read the following audio description in a calm, clear tone at a fast pace.
- Speak like a professional documentary narrator with a neutral and objective voice.
- Do not express emotions excessively; deliver the facts in a composed manner.
[extremely fast] """

GEMINI_TTS_VOICE_PROFILES = {
    "gemini_kor_female": {
        "language": "ko",
        "voice_name": "Kore",  # Female voice - warm and clear
        "description": "한국어 여성 음성 (Kore)",
        "ad_prompt": GEMINI_TTS_AD_PROMPT_KO,
    },
    "gemini_kor_male": {
        "language": "ko",
        "voice_name": "Puck",  # Male voice - friendly and versatile
        "description": "한국어 남성 음성 (Puck)",
        "ad_prompt": GEMINI_TTS_AD_PROMPT_KO,
    },
    "gemini_eng_female": {
        "language": "en",
        "voice_name": "Aoede",  # Female voice - clear and professional
        "description": "영어 여성 음성 (Aoede)",
        "ad_prompt": GEMINI_TTS_AD_PROMPT_EN,
    },
    "gemini_eng_male": {
        "language": "en",
        "voice_name": "Charon",  # Male voice - deep and authoritative
        "description": "영어 남성 음성 (Charon)",
        "ad_prompt": GEMINI_TTS_AD_PROMPT_EN,
    },
}

# Global Gemini client (loaded lazily)
_gemini_client = None

# Global Coqui TTS model (loaded lazily)
_coqui_tts_model = None


def get_coqui_tts_model():
    """
    Lazily load Coqui TTS model (XTTS v2).
    Returns None if Coqui TTS is not available.
    """
    global _coqui_tts_model
    if _coqui_tts_model is not None:
        return _coqui_tts_model
    
    try:
        # Coqui TTS 라이센스 동의 자동 처리 (비대화형 환경용)
        import os
        os.environ["COQUI_TOS_AGREED"] = "1"
        
        logger.info("[TTS] Coqui TTS 모듈 임포트 시도...")
        from TTS.api import TTS
        logger.info("[TTS] Coqui TTS 모듈 임포트 성공")
        
        TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
        logger.info(f"[TTS] Coqui TTS 모델 로드 시도: {TTS_MODEL_NAME}")
        
        # Try GPU first, fallback to CPU
        try:
            logger.info("[TTS] GPU 모드로 모델 로드 시도...")
            _coqui_tts_model = TTS(TTS_MODEL_NAME, gpu=True)
            logger.info("[TTS] ✅ Coqui XTTS v2 모델 GPU 모드로 로드 완료")
        except Exception as gpu_err:
            logger.warning(f"[TTS] ⚠️ GPU 로드 실패: {type(gpu_err).__name__}: {gpu_err}")
            logger.warning(f"[TTS] GPU 에러 상세: {str(gpu_err)}")
            import traceback
            logger.debug(f"[TTS] GPU 에러 스택:\n{traceback.format_exc()}")
            
            logger.info("[TTS] CPU 모드로 모델 로드 재시도...")
            try:
                _coqui_tts_model = TTS(TTS_MODEL_NAME, gpu=False)
                logger.info("[TTS] ✅ Coqui XTTS v2 모델 CPU 모드로 로드 완료")
            except Exception as cpu_err:
                logger.error(f"[TTS] ❌ CPU 로드도 실패: {type(cpu_err).__name__}: {cpu_err}")
                logger.error(f"[TTS] CPU 에러 상세: {str(cpu_err)}")
                import traceback
                logger.error(f"[TTS] CPU 에러 스택:\n{traceback.format_exc()}")
                raise cpu_err
        
        return _coqui_tts_model
    except ImportError as import_err:
        logger.error(f"[TTS] ❌ Coqui TTS 모듈 임포트 실패: {import_err}")
        logger.error(f"[TTS] ImportError 상세: {str(import_err)}")
        import traceback
        logger.error(f"[TTS] ImportError 스택:\n{traceback.format_exc()}")
        logger.warning("[TTS] gTTS로 폴백합니다.")
        return None
    except Exception as e:
        logger.error(f"[TTS] ❌ Coqui TTS 로드 실패: {type(e).__name__}: {e}")
        logger.error(f"[TTS] 에러 상세: {str(e)}")
        import traceback
        logger.error(f"[TTS] 에러 스택:\n{traceback.format_exc()}")
        logger.warning("[TTS] gTTS로 폴백합니다.")
        return None


def get_gemini_client(api_key: str = None):
    """
    Lazily initialize Gemini client for TTS.
    
    Args:
        api_key: Optional API key. If not provided, looks for GEMINI_API_KEY env var.
        
    Returns:
        Gemini client or None if not available.
    """
    global _gemini_client
    if _gemini_client is not None:
        return _gemini_client
    
    try:
        from google import genai
        
        # Get API key from parameter or environment
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("API_KEY_GEMINI")
        
        if not key:
            logger.error("[TTS] ❌ Gemini API 키가 설정되지 않았습니다.")
            logger.error("[TTS] GEMINI_API_KEY 환경 변수를 설정하거나 api_key 파라미터를 전달하세요.")
            return None
        
        _gemini_client = genai.Client(api_key=key)
        logger.info("[TTS] ✅ Gemini 클라이언트 초기화 완료")
        return _gemini_client
        
    except ImportError as e:
        logger.error(f"[TTS] ❌ google-genai 모듈 임포트 실패: {e}")
        logger.error("[TTS] 설치 명령: pip install google-genai")
        return None
    except Exception as e:
        logger.error(f"[TTS] ❌ Gemini 클라이언트 초기화 실패: {e}")
        return None


def synthesize_tts_segment_gemini(
    text: str,
    output_path: str,
    voice_profile: str = "gemini_kor_female",
    api_key: str = None,
) -> AudioSegment:
    """
    Gemini TTS를 사용하여 TTS 오디오 생성
    
    Args:
        text: 합성할 텍스트
        output_path: 출력 파일 경로
        voice_profile: Gemini TTS 음성 프로필 (gemini_kor_female, gemini_kor_male, gemini_eng_female, gemini_eng_male)
        api_key: Gemini API 키 (선택사항, 환경변수에서도 읽음)
        
    Returns:
        AudioSegment 객체
    """
    from google.genai import types
    
    client = get_gemini_client(api_key)
    if client is None:
        raise RuntimeError("Gemini 클라이언트를 초기화할 수 없습니다.")
    
    if voice_profile not in GEMINI_TTS_VOICE_PROFILES:
        raise ValueError(f"알 수 없는 Gemini 음성 프로필: {voice_profile}. 가능한 값: {list(GEMINI_TTS_VOICE_PROFILES.keys())}")
    
    cfg = GEMINI_TTS_VOICE_PROFILES[voice_profile]
    voice_name = cfg["voice_name"]
    lang = cfg["language"]
    ad_prompt = cfg.get("ad_prompt", "")
    
    # 프롬프트와 텍스트를 결합하여 화면해설에 최적화된 톤과 속도 적용
    # [fast] 태그로 약 1.25배 속도 증가
    full_text = f"{ad_prompt}{text}"
    
    logger.debug(f"[TTS] Gemini TTS 합성 시작: voice={voice_name}, lang={lang}, text={text[:50]}...")
    logger.debug(f"[TTS] 적용된 프롬프트: {ad_prompt[:50]}...")
    
    try:
        # Gemini TTS API 호출
        response = client.models.generate_content(
            model=GEMINI_TTS_MODEL,
            contents=full_text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name,
                        )
                    )
                ),
            ),
        )
        
        # 오디오 데이터 추출
        audio_data = None
        mime_type = None
        
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        audio_data = part.inline_data.data
                        mime_type = part.inline_data.mime_type
                        break
        
        if audio_data is None:
            raise RuntimeError("Gemini TTS 응답에서 오디오 데이터를 찾을 수 없습니다.")
        
        logger.debug(f"[TTS] Gemini TTS 응답 수신: mime_type={mime_type}, data_size={len(audio_data)} bytes")
        
        # 오디오 데이터를 AudioSegment로 변환
        # Gemini TTS는 기본적으로 audio/L16;rate=24000 (raw PCM) 또는 다른 형식 반환
        if mime_type and "L16" in mime_type:
            # Raw PCM 16-bit signed integer, little-endian
            # 샘플 레이트 추출 (예: audio/L16;rate=24000)
            sample_rate = 24000  # 기본값
            if "rate=" in mime_type:
                try:
                    rate_str = mime_type.split("rate=")[1].split(";")[0]
                    sample_rate = int(rate_str)
                except (IndexError, ValueError):
                    pass
            
            # Raw PCM 데이터를 WAV로 변환
            wav_path = Path(output_path).with_suffix(".wav")
            with wave.open(str(wav_path), 'wb') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)
            
            audio = AudioSegment.from_wav(str(wav_path))
        elif mime_type and ("mp3" in mime_type.lower() or "mpeg" in mime_type.lower()):
            # MP3 형식
            mp3_path = Path(output_path).with_suffix(".mp3")
            with open(str(mp3_path), 'wb') as f:
                f.write(audio_data)
            audio = AudioSegment.from_mp3(str(mp3_path))
        elif mime_type and "wav" in mime_type.lower():
            # WAV 형식
            wav_path = Path(output_path).with_suffix(".wav")
            with open(str(wav_path), 'wb') as f:
                f.write(audio_data)
            audio = AudioSegment.from_wav(str(wav_path))
        else:
            # 알 수 없는 형식 - raw PCM으로 가정
            logger.warning(f"[TTS] 알 수 없는 오디오 형식: {mime_type}, raw PCM으로 처리")
            sample_rate = 24000
            wav_path = Path(output_path).with_suffix(".wav")
            with wave.open(str(wav_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_data)
            audio = AudioSegment.from_wav(str(wav_path))
        
        # 모노/16kHz로 변환 (일관성 유지)
        audio = audio.set_channels(1).set_frame_rate(16000)
        
        logger.info(f"[TTS] ✅ Gemini TTS 합성 완료: {len(audio)}ms")
        return audio
        
    except Exception as e:
        logger.error(f"[TTS] ❌ Gemini TTS 합성 실패: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"[TTS] 에러 스택:\n{traceback.format_exc()}")
        raise


def audiosegment_to_lufs(audio: AudioSegment) -> float:
    """
    pydub.AudioSegment -> integrated LUFS (EBU R128) 측정
    """
    try:
        import pyloudnorm as pyln
    except ImportError:
        logger.warning("[TTS] pyloudnorm 모듈을 찾을 수 없습니다. 기본 LUFS 값 사용.")
        return -23.0
    
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    # 스테레오일 경우 모노로 변환
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels))
        samples = samples.mean(axis=1)

    # 16비트 오디오의 최대 피크 값
    peak = float(1 << (8 * audio.sample_width - 1))
    samples = samples / peak

    try:
        meter = pyln.Meter(audio.frame_rate)
        loudness = meter.integrated_loudness(samples)
        return loudness if loudness > -70 else -30.0  # 측정 불가 시 기본값
    except Exception as e:
        logger.warning(f"[TTS] LUFS 측정 실패: {e}")
        return -30.0


def load_mono_16k(audio_path: str) -> Tuple[np.ndarray, int]:
    """
    WAV 파일을 mono / 16k로 맞춰서 numpy array로 반환 (VAD용)
    soundfile + librosa를 사용하여 torchcodec 의존성 문제 회피
    """
    try:
        import soundfile as sf
        import librosa
        
        # soundfile로 오디오 로드 (torchcodec 문제 회피)
        wav, sr = sf.read(audio_path, dtype='float32')
        
        # 스테레오일 경우 모노로 변환
        if len(wav.shape) > 1 and wav.shape[1] > 1:
            wav = wav.mean(axis=1)
        
        # 16kHz로 리샘플링 (librosa 사용)
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            sr = 16000
        
        return wav, sr
    except ImportError as e:
        logger.warning(f"[TTS] soundfile 또는 librosa를 찾을 수 없습니다: {e}. VAD 기능 비활성화.")
        return None, 16000
    except Exception as e:
        logger.warning(f"[TTS] 오디오 로드 실패: {e}. VAD 기능 비활성화.")
        return None, 16000


def get_silero_vad_model():
    """
    Silero VAD 모델 로드
    """
    try:
        import torch
        import sys
        import os
        from contextlib import redirect_stdout, redirect_stderr
        
        # stdout/stderr를 완전히 억제 (torch.hub 다운로드 메시지 차단)
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    trust_repo=True,
                    verbose=False,
                )
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
        
        return model, utils
    except Exception as e:
        logger.warning(f"[TTS] Silero VAD 모델 로드 실패: {e}")
        return None, None


def get_speech_timestamps_from_audio(audio_path: str) -> List[Dict]:
    """
    오디오 파일에서 VAD를 사용하여 대사 구간 검출
    Returns: [{"start": sample_idx, "end": sample_idx}, ...]
    """
    wav, sr = load_mono_16k(audio_path)
    if wav is None:
        return []
    
    model, utils = get_silero_vad_model()
    if model is None:
        return []
    
    try:
        import torch
        
        (get_speech_timestamps, _, _, _, _) = utils
        wav_tensor = torch.from_numpy(wav).float()
        
        speech_ts = get_speech_timestamps(
            wav_tensor,
            model,
            sampling_rate=sr,
        )
        logger.info(f"[TTS] VAD 분석 완료. 검출된 speech 구간 수: {len(speech_ts)}")
        return speech_ts
    except Exception as e:
        logger.warning(f"[TTS] VAD 분석 실패: {e}")
        return []


def get_overlap_intervals_for_ad(
    ad_start_sec: float, 
    ad_end_sec: float, 
    speech_ts: List[Dict], 
    sr: int = 16000
) -> List[Tuple[float, float]]:
    """
    AD 구간과 VAD speech_ts 사이의 겹치는 구간들을 (초 단위)로 계산
    Returns: [(ov_start_sec, ov_end_sec), ...]
    """
    overlaps = []

    for seg in speech_ts:
        sp_start = seg["start"] / sr
        sp_end = seg["end"] / sr

        ov_start = max(ad_start_sec, sp_start)
        ov_end = min(ad_end_sec, sp_end)

        if ov_start < ov_end:
            overlaps.append((ov_start, ov_end))

    if not overlaps:
        return []

    # 시작 시간 기준 정렬 및 병합
    overlaps.sort(key=lambda x: x[0])
    merged = [overlaps[0]]

    for cur_start, cur_end in overlaps[1:]:
        last_start, last_end = merged[-1]
        if cur_start <= last_end:
            merged[-1] = (last_start, max(last_end, cur_end))
        else:
            merged.append((cur_start, cur_end))

    return merged


def change_speed(audio: AudioSegment, speed: float) -> AudioSegment:
    """
    pydub을 사용하여 오디오 속도 조절 (피치 함께 변함)
    """
    if speed == 1.0:
        return audio

    new_frame_rate = int(audio.frame_rate * speed)
    audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})
    audio = audio.set_frame_rate(16000)
    return audio


def resolve_speaker_wav_path(speaker_wav_relative: str) -> str:
    """
    TTS voice profile의 상대 경로를 절대 경로로 변환
    """
    # 현재 스크립트 디렉토리 기준
    script_dir = Path(__file__).parent
    speaker_wav_path = script_dir / speaker_wav_relative
    
    if speaker_wav_path.exists():
        return str(speaker_wav_path)
    
    # 환경변수로 지정된 경로에서도 찾아봄
    python_dir = os.environ.get("PYTHONPATH", "")
    if python_dir:
        alt_path = Path(python_dir) / speaker_wav_relative
        if alt_path.exists():
            return str(alt_path)
    
    # 못 찾으면 원래 경로 반환 (에러가 날 것)
    logger.warning(f"[TTS] Speaker WAV 파일을 찾을 수 없습니다: {speaker_wav_relative}")
    return str(speaker_wav_path)


def parse_time_str(t: str) -> float:
    """
    Parse time string in format "M:SS.s" / "MM:SS.s" / "H:MM:SS.s" to seconds (float).
    
    Args:
        t: Time string (e.g., "0:17.9", "1:23.5", "0:00:12.3")
        
    Returns:
        Time in seconds as float
    """
    try:
        # Handle numeric string (old format)
        if ':' not in str(t):
            return float(t)
        
        parts = str(t).split(":")
        parts = [float(p) for p in parts]
        if len(parts) == 2:
            m, s = parts
            return m * 60 + s
        elif len(parts) == 3:
            h, m, s = parts
            return h * 3600 + m * 60 + s
        else:
            return float(t)
    except (ValueError, AttributeError):
        logger.warning(f"Failed to parse time string: {t}, using 0.0")
        return 0.0


def load_ad_segments(json_path: str) -> pd.DataFrame:
    """
    Load AD segments from JSON file.
    
    Supports both formats:
    - New format: [{"id": 1, "start": 4.0, "end": 6.6, "text": "..."}]
    - Old format: [{"start_time": "0:17.9", "end_time": "0:21.4", "duration_sec": 3.5, "description": "..."}]
    
    Args:
        json_path: Path to AD JSON file
        
    Returns:
        DataFrame with columns: start_time, end_time, duration_sec, description
    """
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"AD JSON file not found: {json_path}")
    
    with p.open("r", encoding="utf-8") as f:
        items = json.load(f)
    
    if isinstance(items, dict):
        # New format: {"audio_descriptions": [...], "full_transcript": [...]}
        if "audio_descriptions" in items:
            items = items["audio_descriptions"]
        else:
            # Fallback or error? 
            # If it's a dict but no audio_descriptions, check if it's a single segment?
            # Unlikely given the prompt.
            raise ValueError("AD JSON is a dict but missing 'audio_descriptions' key")
            
    if not isinstance(items, list):
        raise ValueError(f"AD JSON must be a list or dict with 'audio_descriptions', got {type(items)}")
    
    records = []
    for item in items:
        # Handle new format: {"id": 1, "start": 4.0, "end": 6.6, "text": "..."}
        if "start" in item and "end" in item:
            start = float(item["start"])
            end = float(item["end"])
            text = str(item.get("text", item.get("description", "")))
        # Handle old format: {"start_time": "0:17.9", "end_time": "0:21.4", "description": "..."}
        elif "start_time" in item and "end_time" in item:
            start = parse_time_str(item["start_time"])
            end = parse_time_str(item["end_time"])
            text = str(item.get("description", item.get("text", "")))
        else:
            logger.warning(f"Skipping invalid AD segment: {item}")
            continue
        
        dur = max(0.0, end - start)
        records.append({
            "start_time": start,
            "end_time": end,
            "duration_sec": dur,
            "description": text,
        })
    
    df = pd.DataFrame(records)
    df = df[df["duration_sec"] >= MIN_DESC_LEN].reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} AD segments (>= {MIN_DESC_LEN}s) from {json_path}")
    return df


def check_ffmpeg_available():
    """
    Check if ffmpeg is available on the system.
    
    Raises:
        RuntimeError: If ffmpeg is not found or not accessible
    """
    import shutil
    import os
    
    # Try to find ffmpeg in PATH
    ffmpeg_path = shutil.which('ffmpeg')
    
    # On Windows, also check common installation locations
    if not ffmpeg_path and os.name == 'nt':
        common_paths = [
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Microsoft', 'WinGet', 'Packages', 'Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe', 'ffmpeg-8.0-essentials_build', 'bin', 'ffmpeg.exe'),
            os.path.join('C:', 'ffmpeg', 'bin', 'ffmpeg.exe'),
            os.path.join(os.environ.get('ProgramFiles', ''), 'ffmpeg', 'bin', 'ffmpeg.exe'),
        ]
        for common_path in common_paths:
            if os.path.exists(common_path):
                ffmpeg_path = common_path
                logger.info(f"FFmpeg found at common location: {ffmpeg_path}")
                # Add to PATH for this process
                bin_dir = os.path.dirname(ffmpeg_path)
                current_path = os.environ.get('PATH', '')
                if bin_dir not in current_path:
                    os.environ['PATH'] = f"{bin_dir};{current_path}" if current_path else bin_dir
                    logger.info(f"Added FFmpeg bin directory to PATH: {bin_dir}")
                break
    
    if not ffmpeg_path:
        # Log current PATH for debugging
        current_path = os.environ.get('PATH', '')
        logger.error(f"FFmpeg not found. Current PATH: {current_path[:500]}")
        raise RuntimeError(
            "ffmpeg not found in system PATH. "
            "Please install ffmpeg and ensure it's accessible from the command line. "
            "Download from: https://ffmpeg.org/download.html"
        )
    
    logger.info(f"FFmpeg found at: {ffmpeg_path}")
    return ffmpeg_path


def ensure_wav_audio(input_video_path: str, output_wav_path: str):
    """
    Extract audio from video as WAV file if it doesn't exist.
    
    Args:
        input_video_path: Path to input video file
        output_wav_path: Path to output WAV file
    """
    # Convert to absolute paths
    input_video_path = os.path.abspath(input_video_path)
    output_wav_path = os.path.abspath(output_wav_path)
    
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")
    
    if os.path.exists(output_wav_path):
        logger.debug(f"Base audio WAV already exists: {output_wav_path}")
        return
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_wav_path)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Extracting audio from video: {input_video_path}")
    logger.info(f"Output WAV path: {output_wav_path}")
    
    try:
        (
            ffmpeg
            .input(input_video_path)
            .output(
                output_wav_path,
                acodec="pcm_s16le",
                ac=1,  # mono
                ar="16000",  # 16kHz
            )
            .overwrite_output()
            .run(quiet=True)
        )
        logger.info(f"Extracted audio to: {output_wav_path}")
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"FFmpeg error: {error_msg}")
        raise RuntimeError(f"FFmpeg failed to extract audio: {error_msg}") from e
    except FileNotFoundError as e:
        logger.error(f"FFmpeg executable not found: {e}")
        raise RuntimeError(
            "ffmpeg executable not found. Please install ffmpeg and ensure it's in your system PATH."
        ) from e


def check_python_audioop_compatibility():
    """
    Check if audioop or its replacement is available for pydub.
    
    Raises:
        RuntimeError: If audioop is not available and cannot be imported
    """
    try:
        import audioop
        logger.debug("audioop module found (built-in)")
        return True
    except ImportError:
        try:
            import pyaudioop as audioop
            logger.debug("pyaudioop module found (replacement for audioop)")
            return True
        except ImportError:
            try:
                import audioop_lts as audioop
                logger.debug("audioop-lts module found (replacement for audioop)")
                return True
            except ImportError:
                raise RuntimeError(
                    "audioop module not available. "
                    "On Python 3.13+, audioop was removed. "
                    "Please install audioop-lts: pip install audioop-lts"
                )


def synthesize_tts_segment_gtts(
    text: str, 
    output_path: str, 
    lang: str = "ko"
) -> AudioSegment:
    """
    gTTS를 사용하여 TTS 오디오 생성 (기본 방식)
    """
    mp3_path = Path(output_path).with_suffix(".mp3")
    
    tts = gTTS(text=text, lang=lang)
    tts.save(str(mp3_path))
    
    audio = AudioSegment.from_mp3(str(mp3_path))
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    # Clean up MP3
    mp3_path.unlink(missing_ok=True)
    
    return audio


def synthesize_tts_segment_coqui(
    text: str, 
    output_path: str, 
    voice_profile: str = "kor_female"
) -> AudioSegment:
    """
    Coqui XTTS v2를 사용하여 TTS 오디오 생성 (고급 방식)
    """
    tts_model = get_coqui_tts_model()
    if tts_model is None:
        raise RuntimeError("Coqui TTS 모델을 로드할 수 없습니다.")
    
    if voice_profile not in TTS_VOICE_PROFILES:
        raise ValueError(f"알 수 없는 음성 프로필: {voice_profile}. 가능한 값: {list(TTS_VOICE_PROFILES.keys())}")
    
    cfg = TTS_VOICE_PROFILES[voice_profile]
    lang = cfg["language"]
    speaker_wav = resolve_speaker_wav_path(cfg["speaker_wav"])
    speed = cfg["speed"]
    
    tmp_wav = Path(output_path).with_suffix(".tmp.wav")
    
    # Coqui TTS로 생성
    tts_model.tts_to_file(
        text=text,
        file_path=str(tmp_wav),
        speaker_wav=speaker_wav,
        language=lang,
    )
    
    # pydub으로 로드 & 모노/16k & 속도 조절
    audio = AudioSegment.from_wav(str(tmp_wav))
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio = change_speed(audio, speed)
    
    # Clean up temp file
    if tmp_wav.exists():
        tmp_wav.unlink()
    
    return audio


def synthesize_tts_for_video(
    video_id: str,
    video_path: str,
    ad_json_path: str,
    output_dir: str,
    lang: str = "ko",  # or "en" depending on AD language
    voice_profile: str = "gtts",  # 'gtts' | 'kor_male' | 'kor_female' | 'eng_male' | 'eng_female' | 'gemini_kor_female' | 'gemini_kor_male' | 'gemini_eng_female' | 'gemini_eng_male'
    enable_ducking: bool = True,  # VAD 기반 더킹 활성화 여부
    gemini_api_key: str = None,  # Gemini TTS용 API 키 (선택사항)
) -> Dict:
    """
    Synthesize TTS audio from AD JSON and mix with video.
    
    Args:
        video_id: Unique identifier for the video
        video_path: Path to input video file
        ad_json_path: Path to AD JSON file
        output_dir: Directory to save output files
        lang: Language code for TTS (default: "ko" for Korean)
        voice_profile: TTS voice profile:
            - 'gtts': Google Text-to-Speech (기본, 무료)
            - 'kor_male', 'kor_female', 'eng_male', 'eng_female': Coqui XTTS v2 (고급)
            - 'gemini_kor_female', 'gemini_kor_male', 'gemini_eng_female', 'gemini_eng_male': Gemini TTS (Google AI)
        enable_ducking: Enable VAD-based ducking (reduce AD volume when overlapping with dialogue)
        gemini_api_key: Optional Gemini API key for Gemini TTS (can also use GEMINI_API_KEY env var)
        
    Returns:
        Dictionary with paths and metadata:
        {
            "videoId": str,
            "finalAudioPath": str,
            "finalVideoPath": str,
            "usedSegments": int,
            "cutSegments": int,
        }
    """
    logger.info(f"[TTS] Starting TTS synthesis for video: {video_id}")
    logger.info(f"[TTS] Voice profile: {voice_profile}, Ducking: {enable_ducking}")
    
    # Check Python version and audioop compatibility
    import sys
    logger.info(f"[TTS] Python version: {sys.version}")
    try:
        check_python_audioop_compatibility()
    except RuntimeError as e:
        logger.error(f"[TTS] Audio compatibility check failed: {e}")
        raise
    
    # Diagnose package availability
    logger.info("[TTS] ========== 패키지 진단 ==========")
    
    # Check Coqui TTS
    try:
        import TTS
        logger.info(f"[TTS] ✅ Coqui TTS 설치됨: {TTS.__version__ if hasattr(TTS, '__version__') else '버전 정보 없음'}")
    except ImportError as e:
        logger.error(f"[TTS] ❌ Coqui TTS 미설치: {e}")
        logger.error("[TTS] 설치 명령: pip install coqui-tts")
    
    # Check soundfile
    try:
        import soundfile as sf
        logger.info(f"[TTS] ✅ soundfile 설치됨: {sf.__version__ if hasattr(sf, '__version__') else '버전 정보 없음'}")
    except ImportError as e:
        logger.error(f"[TTS] ❌ soundfile 미설치: {e}")
        logger.error("[TTS] 설치 명령: pip install soundfile")
    
    # Check librosa
    try:
        import librosa
        logger.info(f"[TTS] ✅ librosa 설치됨: {librosa.__version__ if hasattr(librosa, '__version__') else '버전 정보 없음'}")
    except ImportError as e:
        logger.error(f"[TTS] ❌ librosa 미설치: {e}")
        logger.error("[TTS] 설치 명령: pip install librosa")
    
    # Check pyloudnorm
    try:
        import pyloudnorm
        logger.info(f"[TTS] ✅ pyloudnorm 설치됨: {pyloudnorm.__version__ if hasattr(pyloudnorm, '__version__') else '버전 정보 없음'}")
    except ImportError as e:
        logger.warning(f"[TTS] ⚠️ pyloudnorm 미설치: {e} (더킹 기능 제한)")
        logger.warning("[TTS] 설치 명령: pip install pyloudnorm")
    
    logger.info("[TTS] ==================================")
    
    # Check ffmpeg availability
    try:
        check_ffmpeg_available()
    except RuntimeError as e:
        logger.error(f"[TTS] FFmpeg check failed: {e}")
        raise
    
    # Convert all paths to absolute
    video_path = os.path.abspath(video_path)
    ad_json_path = os.path.abspath(ad_json_path)
    output_dir = os.path.abspath(output_dir)
    
    logger.info(f"[TTS] Absolute paths:")
    logger.info(f"[TTS]   Video: {video_path}")
    logger.info(f"[TTS]   AD JSON: {ad_json_path}")
    logger.info(f"[TTS]   Output dir: {output_dir}")
    
    # Verify input files exist
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(ad_json_path):
        raise FileNotFoundError(f"AD JSON file not found: {ad_json_path}")
    
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # 1) Load AD segments
    df = load_ad_segments(ad_json_path)
    
    if len(df) == 0:
        raise ValueError(f"No valid AD segments found in {ad_json_path}")
    
    # 2) Extract base audio WAV (if not exists)
    base_wav = out_root / f"{video_id}_base.wav"
    ensure_wav_audio(video_path, str(base_wav))
    
    base_audio = AudioSegment.from_wav(str(base_wav))
    total_ms = len(base_audio)
    
    logger.info(f"[TTS] Base audio length: {total_ms}ms ({total_ms/1000:.1f}s)")
    
    # 3) VAD 분석 (더킹용)
    speech_ts = []
    dialogue_lufs = -23.0  # 기본값
    target_ad_lufs = -25.0  # 기본값
    
    if enable_ducking:
        logger.info("[TTS] VAD 분석 시작...")
        speech_ts = get_speech_timestamps_from_audio(str(base_wav))
        
        if not speech_ts:
            # Check if it's because packages are missing
            try:
                import soundfile
                import librosa
            except ImportError:
                warning_msg = (
                    f"[TTS] ⚠️ 더킹 기능을 사용할 수 없습니다!\n"
                    f"VAD 분석에 필요한 패키지가 설치되지 않았습니다.\n"
                    f"\n필요한 패키지:\n"
                    f"  - soundfile>=0.12.0\n"
                    f"  - librosa>=0.11.0\n"
                    f"\n설치 방법:\n"
                    f"  pip install soundfile librosa\n"
                    f"\n또는 설치 스크립트 실행:\n"
                    f"  Windows: install_tts_dependencies.bat\n"
                    f"  Linux/Mac: bash install_tts_dependencies.sh\n"
                    f"\n더킹 없이 진행합니다."
                )
                logger.warning(warning_msg)
        
        if speech_ts:
            # 대사 LUFS 계산
            dialogue_track = AudioSegment.silent(duration=total_ms)
            for seg in speech_ts:
                start_ms = int(seg['start'] * 1000 / 16000)
                end_ms = int(seg['end'] * 1000 / 16000)
                
                if start_ms < 0:
                    start_ms = 0
                if end_ms > total_ms:
                    end_ms = total_ms
                if start_ms >= end_ms:
                    continue
                
                speech_seg = base_audio[start_ms:end_ms]
                dialogue_track = dialogue_track.overlay(speech_seg, position=start_ms)
            
            try:
                if len(dialogue_track) >= 1000:
                    dialogue_lufs = audiosegment_to_lufs(dialogue_track)
                    logger.info(f"[TTS] 배경 발화 (Dialogue) LUFS: {dialogue_lufs:.2f} LUFS")
            except Exception as e:
                logger.warning(f"[TTS] 배경 발화 LUFS 측정 실패, 기본값 사용: {e}")
            
            target_ad_lufs = dialogue_lufs - DIALOGUE_LUFS_MARGIN
            logger.info(f"[TTS] AD 목표 LUFS (배경 발화 - {DIALOGUE_LUFS_MARGIN} LUFS): {target_ad_lufs:.2f} LUFS")
        else:
            logger.info("[TTS] VAD 구간 없음, 더킹 비활성화")
    
    # 4) Create silent track for TTS overlay
    tts_track = AudioSegment.silent(duration=total_ms)
    
    tts_segments_dir = out_root / f"{video_id}_tts_segments"
    tts_segments_dir.mkdir(exist_ok=True)
    generated_segments: List[Dict] = []
    
    used_cnt = 0
    overflow_cnt = 0  # end_time을 넘어간 세그먼트 수
    
    # Determine TTS engine to use
    use_gemini = voice_profile.startswith("gemini_") and voice_profile in GEMINI_TTS_VOICE_PROFILES
    use_coqui = not use_gemini and voice_profile != "gtts" and voice_profile in TTS_VOICE_PROFILES
    
    if use_gemini:
        # Pre-check Gemini client
        gemini_client = get_gemini_client(gemini_api_key)
        if gemini_client is None:
            error_msg = (
                f"[TTS] ❌ Gemini TTS를 사용할 수 없습니다!\n"
                f"음성 프로필 '{voice_profile}'을 사용하려면 google-genai 패키지와 API 키가 필요합니다.\n"
                f"\n설치 방법:\n"
                f"  pip install google-genai\n"
                f"\nAPI 키 설정:\n"
                f"  환경변수 GEMINI_API_KEY를 설정하거나 gemini_api_key 파라미터를 전달하세요.\n"
                f"\ngTTS로 폴백합니다."
            )
            logger.error(error_msg)
            logger.warning("[TTS] gTTS로 폴백합니다.")
            use_gemini = False
        else:
            logger.info(f"[TTS] ✅ Gemini TTS 사용: {voice_profile} ({GEMINI_TTS_VOICE_PROFILES[voice_profile]['description']})")
    
    if use_coqui:
        # Pre-load Coqui model
        coqui_model = get_coqui_tts_model()
        if coqui_model is None:
            error_msg = (
                f"[TTS] ❌ Coqui TTS를 사용할 수 없습니다!\n"
                f"음성 프로필 '{voice_profile}'을 사용하려면 coqui-tts 패키지가 필요합니다.\n"
                f"\n설치 방법:\n"
                f"  pip install coqui-tts>=0.27.0\n"
                f"\n또는 설치 스크립트 실행:\n"
                f"  Windows: install_tts_dependencies.bat\n"
                f"  Linux/Mac: bash install_tts_dependencies.sh\n"
                f"\ngTTS로 폴백합니다."
            )
            logger.error(error_msg)
            logger.warning("[TTS] gTTS로 폴백합니다.")
            use_coqui = False
    
    # 5) Generate TTS for each segment
    for idx, row in df.iterrows():
        text = str(row["description"]).strip()
        if not text:
            continue
        
        start_s = float(row["start_time"])
        end_s = float(row["end_time"])
        slot_start_ms = int(start_s * 1000)
        slot_end_ms = int(end_s * 1000)
        slot_len_ms = max(1, slot_end_ms - slot_start_ms)
        
        # 다음 AD 세그먼트의 시작 시간 확인 (나레이션 겹침 방지)
        next_ad_start_ms = None
        if idx + 1 < len(df):
            next_start_s = float(df.iloc[idx + 1]["start_time"])
            next_ad_start_ms = int(next_start_s * 1000)
        
        _wav_path = tts_segments_dir / f"seg_{idx:03d}.wav"
        
        try:
            logger.debug(f"[TTS] Generating TTS for segment {idx}: {text[:50]}...")
            
            # Generate TTS
            if use_gemini:
                tts_audio = synthesize_tts_segment_gemini(text, str(_wav_path), voice_profile, gemini_api_key)
            elif use_coqui:
                tts_audio = synthesize_tts_segment_coqui(text, str(_wav_path), voice_profile)
            else:
                tts_audio = synthesize_tts_segment_gtts(text, str(_wav_path), lang)
            
            ad_duration_ms = len(tts_audio)
            ad_end_s = start_s + (ad_duration_ms / 1000.0)
            
            # Check if TTS overflows beyond end_time
            has_overflow = ad_duration_ms > slot_len_ms
            overflow_start_ms = slot_len_ms if has_overflow else ad_duration_ms
            
            if has_overflow:
                logger.debug(f"[TTS] Segment {idx} overflow: TTS={ad_duration_ms}ms, slot={slot_len_ms}ms, overflow={ad_duration_ms - slot_len_ms}ms")
            
            # 6) 더킹 처리: LUFS 정규화 + overflow 구간 감쇄 + 대사 겹침 구간 감쇄
            if enable_ducking:
                # (1) AD 전체를 목표 LUFS로 맞춤
                try:
                    ad_lufs = audiosegment_to_lufs(tts_audio)
                    gain_db = target_ad_lufs - ad_lufs
                    tts_audio = tts_audio.apply_gain(gain_db)
                except Exception as e:
                    logger.warning(f"[TTS] LUFS 정규화 실패: {e}")
                
                # (2) Overflow 구간 감쇄 (end_time 이후 부분)
                # TTS가 slot보다 길면, end_time 이후 부분의 볼륨을 줄임
                if has_overflow:
                    # slot 내 구간 (정상 볼륨)
                    in_slot_audio = tts_audio[:overflow_start_ms]
                    # overflow 구간 (볼륨 감쇄)
                    overflow_audio = tts_audio[overflow_start_ms:]
                    overflow_audio = overflow_audio.apply_gain(AD_OVERFLOW_ATTENUATION_DB)
                    
                    # 합치기
                    tts_audio = in_slot_audio + overflow_audio
                    overflow_cnt += 1  # overflow 발생 카운트
                    logger.info(f"[TTS] Segment {idx}: overflow {ad_duration_ms - slot_len_ms}ms에 {AD_OVERFLOW_ATTENUATION_DB}dB 감쇄 적용")
                
                # (3) 대사와 겹치는 구간 계산 (VAD 기반, 선택적)
                if speech_ts:
                    overlap_intervals_sec = get_overlap_intervals_for_ad(
                        start_s, ad_end_s, speech_ts, 16000
                    )
                    
                    if overlap_intervals_sec:
                        # (4) 겹치는 구간을 ms 단위로 변환 (AD 내부 상대 위치)
                        overlap_intervals_ms = []
                        for ov_start_sec, ov_end_sec in overlap_intervals_sec:
                            rel_start_ms = int((ov_start_sec - start_s) * 1000)
                            rel_end_ms = int((ov_end_sec - start_s) * 1000)
                            rel_start_ms = max(0, min(rel_start_ms, ad_duration_ms))
                            rel_end_ms = max(0, min(rel_end_ms, ad_duration_ms))
                            if rel_start_ms < rel_end_ms:
                                overlap_intervals_ms.append((rel_start_ms, rel_end_ms))
                        
                        if overlap_intervals_ms:
                            # (5) 겹치지 않는 구간과 겹치는 구간 분리
                            overlap_intervals_ms.sort(key=lambda x: x[0])
                            
                            non_overlap_intervals_ms = []
                            cur = 0
                            for ov_start_ms, ov_end_ms in overlap_intervals_ms:
                                if cur < ov_start_ms:
                                    non_overlap_intervals_ms.append((cur, ov_start_ms))
                                cur = max(cur, ov_end_ms)
                            if cur < ad_duration_ms:
                                non_overlap_intervals_ms.append((cur, ad_duration_ms))
                            
                            # (6) 새 AudioSegment 조합
                            adjusted_ad = AudioSegment.silent(duration=ad_duration_ms)
                            
                            # 겹치지 않는 구간 (기본 볼륨)
                            for ns, ne in non_overlap_intervals_ms:
                                if ne <= ns:
                                    continue
                                seg = tts_audio[ns:ne]
                                adjusted_ad = adjusted_ad.overlay(seg, position=ns)
                            
                            # 겹치는 구간 (추가 감쇄)
                            for os_ms, oe_ms in overlap_intervals_ms:
                                if oe_ms <= os_ms:
                                    continue
                                seg = tts_audio[os_ms:oe_ms]
                                seg = seg.apply_gain(AD_OVERLAP_EXTRA_ATTENUATION_DB)
                                adjusted_ad = adjusted_ad.overlay(seg, position=os_ms)
                            
                            tts_audio = adjusted_ad
                            logger.debug(f"[TTS] Segment {idx}: 대사 겹침 {len(overlap_intervals_ms)}개 구간 감쇄 적용")
            
            # 다음 AD 구간 시작 전에 현재 TTS 강제 종료 (나레이션 겹침 방지)
            if next_ad_start_ms is not None:
                max_tts_duration_ms = next_ad_start_ms - slot_start_ms
                if max_tts_duration_ms > 0 and len(tts_audio) > max_tts_duration_ms:
                    logger.info(f"[TTS] Segment {idx}: 다음 AD 시작({next_ad_start_ms}ms) 전에 강제 종료 - 원래 길이 {len(tts_audio)}ms → {max_tts_duration_ms}ms로 잘림")
                    tts_audio = tts_audio[:max_tts_duration_ms]
            
            # Overlay in TTS track
            tts_track = tts_track.overlay(tts_audio, position=slot_start_ms)
            used_cnt += 1
            
            # Export WAV for debugging (optional)
            tts_audio.export(str(_wav_path), format="wav")
            
            generated_segments.append({
                "index": int(idx),
                "start_time": float(start_s),
                "end_time": float(end_s),
                "description": text,
                "duration_ms": int(len(tts_audio)),
                "wav_path": str(_wav_path),
                "has_overflow": has_overflow,  # TTS가 end_time을 넘어갔는지 여부
            })
            
        except Exception as e:
            logger.error(f"[TTS] Failed to generate TTS for segment {idx}: {e}")
            continue
    
    logger.info(f"[TTS] Used {used_cnt} segments, overflow (ducked) {overflow_cnt} segments")
    
    # 7) Mix base + TTS
    final_audio_path = out_root / f"{video_id}_ad_mix.wav"
    logger.info(f"[TTS] Mixing base audio with TTS track...")
    mixed = base_audio.overlay(tts_track)
    mixed.export(str(final_audio_path), format="wav")
    logger.info(f"[TTS] Mixed audio saved: {final_audio_path}")
    
    # 8) Mux into video
    final_video_path = out_root / f"{video_id}_ad_mix.mp4"
    final_video_path = Path(os.path.abspath(str(final_video_path)))
    logger.info(f"[TTS] Muxing audio into video...")
    logger.info(f"[TTS] Input video: {video_path}")
    logger.info(f"[TTS] Input audio: {final_audio_path}")
    logger.info(f"[TTS] Output video: {final_video_path}")
    
    # Verify input files exist
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found for muxing: {video_path}")
    if not os.path.exists(final_audio_path):
        raise FileNotFoundError(f"Audio file not found for muxing: {final_audio_path}")
    
    try:
        # Use separate inputs for video and audio, then map streams
        video_input = ffmpeg.input(video_path)
        audio_input = ffmpeg.input(str(final_audio_path))
        
        # Map video stream from video input and audio stream from audio input
        (
            ffmpeg
            .output(
                video_input['v'],  # video stream from video input
                audio_input['a'],  # audio stream from audio input
                str(final_video_path),
                vcodec="copy",      # copy video codec (no re-encoding)
                acodec="aac",       # encode audio as AAC
                **{'shortest': None}  # finish encoding when shortest input stream ends
            )
            .overwrite_output()
            .run(quiet=True)
        )
        logger.info(f"[TTS] Final video saved: {final_video_path}")
        
        # Verify output file was created
        if not os.path.exists(final_video_path):
            raise RuntimeError(f"Output video file was not created: {final_video_path}")
        
        output_size = os.path.getsize(final_video_path)
        logger.info(f"[TTS] Output video size: {output_size} bytes")
    except ffmpeg.Error as e:
        error_msg = e.stderr.decode() if e.stderr else str(e)
        logger.error(f"[TTS] FFmpeg error: {error_msg}")
        raise RuntimeError(f"FFmpeg failed to mux video: {error_msg}") from e
    except FileNotFoundError as e:
        logger.error(f"[TTS] FFmpeg executable not found: {e}")
        raise RuntimeError(
            "ffmpeg executable not found. Please install ffmpeg and ensure it's in your system PATH."
        ) from e
    
    return {
        "videoId": video_id,
        "finalAudioPath": str(final_audio_path),
        "finalVideoPath": str(final_video_path),
        "usedSegments": int(used_cnt),
        "overflowSegments": int(overflow_cnt),  # end_time 넘어간 세그먼트 수 (볼륨 감쇄 적용됨)
        "segmentOutputs": generated_segments,
        "segmentsDir": str(tts_segments_dir),
        "voiceProfile": voice_profile,
        "duckingEnabled": enable_ducking,
    }


if __name__ == "__main__":
    import argparse
    import sys
    
    # Configure logging for CLI
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Synthesize TTS audio from AD JSON and mix with video")
    parser.add_argument("--video_id", required=True, help="Video ID")
    parser.add_argument("--video_path", required=True, help="Path to input video file")
    parser.add_argument("--ad_json_path", required=True, help="Path to AD JSON file")
    parser.add_argument("--output_dir", required=True, help="Output directory for generated files")
    parser.add_argument("--lang", default="ko", help="Language code for TTS (default: ko)")
    parser.add_argument(
        "--voice_profile", 
        default="gtts", 
        choices=[
            "gtts", 
            "kor_male", "kor_female", "eng_male", "eng_female",
            "gemini_kor_female", "gemini_kor_male", "gemini_eng_female", "gemini_eng_male"
        ],
        help="TTS voice profile (default: gtts). Options: gtts (기본), kor_male/kor_female/eng_male/eng_female (Coqui TTS), gemini_kor_female/gemini_kor_male/gemini_eng_female/gemini_eng_male (Gemini TTS)"
    )
    parser.add_argument(
        "--gemini_api_key",
        default=None,
        help="Gemini API key for Gemini TTS (can also use GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--enable_ducking", 
        action="store_true",
        default=True,
        help="Enable VAD-based ducking (reduce AD volume when overlapping with dialogue)"
    )
    parser.add_argument(
        "--no_ducking", 
        action="store_true",
        help="Disable VAD-based ducking"
    )
    
    args = parser.parse_args()
    
    # Process ducking flag
    enable_ducking = not args.no_ducking if args.no_ducking else args.enable_ducking
    
    try:
        result = synthesize_tts_for_video(
            video_id=args.video_id,
            video_path=args.video_path,
            ad_json_path=args.ad_json_path,
            output_dir=args.output_dir,
            lang=args.lang,
            voice_profile=args.voice_profile,
            enable_ducking=enable_ducking,
            gemini_api_key=args.gemini_api_key,
        )
        
        # Print JSON to stdout so Node can parse it
        import json as _json
        print(_json.dumps(result, ensure_ascii=False))
        
    except Exception as e:
        logger.exception("[TTS] Error during TTS synthesis")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
