"""
GPT API용 영상 데이터 추출 파이프라인

이 스크립트는 영상에서 다음을 추출합니다:
1. 대사 (Whisper STT)
2. 무음 구간 (Silero VAD)
3. 2fps 프레임 이미지

출력:
- JSON 파일: 대사 및 무음 구간 정보 (시간 정보 포함)
- 이미지 폴더: 2fps로 추출된 프레임 이미지 (파일명에 시간 정보 포함)
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, asdict

import torch
import torchaudio
import ffmpeg
import scipy.io.wavfile as wavfile
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SpeechSegment:
    """대사 구간 정보"""
    id: int
    start_time: float  # 초 단위
    end_time: float
    duration: float
    text: str
    type: str = "speech"

@dataclass  
class SilenceSegment:
    """무음 구간 정보"""
    id: int
    start_time: float  # 초 단위
    end_time: float
    duration: float
    type: str = "silence"

@dataclass
class FrameInfo:
    """프레임 이미지 정보"""
    frame_number: int
    timestamp: float  # 초 단위
    filename: str


def extract_audio(input_video: str, output_audio: str, sample_rate: int = 16000) -> str:
    """
    비디오에서 오디오를 추출합니다 (모노, 지정된 샘플레이트).
    
    Args:
        input_video: 입력 비디오 경로
        output_audio: 출력 오디오 경로
        sample_rate: 샘플레이트 (기본: 16000)
    
    Returns:
        출력 오디오 경로
    """
    logger.info(f"[Audio] 오디오 추출 중: {input_video} -> {output_audio}")
    
    (
        ffmpeg
        .input(input_video)
        .output(output_audio, ac=1, ar=sample_rate)
        .overwrite_output()
        .run(quiet=True)
    )
    
    logger.info(f"[Audio] 오디오 추출 완료: {output_audio}")
    return output_audio


def load_vad_model():
    """Silero VAD 모델을 로드합니다."""
    logger.info("[VAD] Silero VAD 모델 로딩 중...")
    
    import sys
    import os
    
    # stdout/stderr를 완전히 억제 (torch.hub 다운로드 메시지 차단)
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True,
                verbose=False,
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks = utils
    logger.info("[VAD] 모델 로딩 완료")
    
    return model, get_speech_timestamps


def detect_speech_segments(
    audio_path: str,
    vad_model,
    get_speech_timestamps_fn,
    min_silence_duration: float = 2.5
) -> Tuple[List[Dict], List[Dict], int]:
    """
    VAD를 사용하여 발화/무음 구간을 탐지합니다.
    
    Args:
        audio_path: 오디오 파일 경로
        vad_model: Silero VAD 모델
        get_speech_timestamps_fn: 발화 구간 탐지 함수
        min_silence_duration: 무음으로 간주할 최소 시간 (초)
    
    Returns:
        (발화 구간 리스트, 무음 구간 리스트, 샘플레이트)
    """
    logger.info(f"[VAD] 발화/무음 구간 탐지 중: {audio_path}")
    
    # scipy를 사용하여 wav 파일 로드 (torchaudio 호환성 문제 해결)
    sr, wav_np = wavfile.read(audio_path)
    
    # numpy dtype을 확인하고 정규화 (변환 전에 확인해야 함)
    if wav_np.dtype == np.int16:
        wav = torch.from_numpy(wav_np.astype(np.float32)) / 32768.0
    elif wav_np.dtype == np.int32:
        wav = torch.from_numpy(wav_np.astype(np.float32)) / 2147483648.0
    elif wav_np.dtype == np.float32:
        wav = torch.from_numpy(wav_np)
    else:
        wav = torch.from_numpy(wav_np.astype(np.float32))
    
    # 모노 채널이면 차원 추가
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    
    speech_timestamps = get_speech_timestamps_fn(wav[0], vad_model, sampling_rate=sr)
    
    speech_segments = []
    silence_segments = []
    
    prev_end = 0.0
    speech_id = 1
    silence_id = 1
    
    for seg in speech_timestamps:
        start = seg["start"] / sr
        end = seg["end"] / sr
        
        # 무음 구간 (이전 발화 끝 ~ 현재 발화 시작)
        if start - prev_end >= min_silence_duration:
            silence_segments.append({
                "id": silence_id,
                "start_time": round(prev_end, 3),
                "end_time": round(start, 3),
                "duration": round(start - prev_end, 3),
                "type": "silence"
            })
            silence_id += 1
        
        # 발화 구간
        speech_segments.append({
            "id": speech_id,
            "start_time": round(start, 3),
            "end_time": round(end, 3),
            "duration": round(end - start, 3),
            "start_sample": seg["start"],
            "end_sample": seg["end"],
            "type": "speech"
        })
        speech_id += 1
        prev_end = end
    
    # 마지막 발화 이후의 무음 구간 (오디오 끝까지)
    audio_duration = wav.shape[1] / sr
    if audio_duration - prev_end >= min_silence_duration:
        silence_segments.append({
            "id": silence_id,
            "start_time": round(prev_end, 3),
            "end_time": round(audio_duration, 3),
            "duration": round(audio_duration - prev_end, 3),
            "type": "silence"
        })
    
    logger.info(f"[VAD] 발화 구간: {len(speech_segments)}개, 무음 구간: {len(silence_segments)}개")
    return speech_segments, silence_segments, sr


def transcribe_speech_segments(
    audio_path: str,
    speech_segments: List[Dict],
    sample_rate: int,
    language: str = "en",
    whisper_model_size: str = "medium",
    temp_dir: str = None
) -> List[Dict]:
    """
    Whisper를 사용하여 발화 구간의 텍스트를 인식합니다.
    
    Args:
        audio_path: 오디오 파일 경로
        speech_segments: 발화 구간 리스트
        sample_rate: 샘플레이트
        language: 언어 코드 (en, ko 등)
        whisper_model_size: Whisper 모델 크기 (tiny, base, small, medium, large)
        temp_dir: 임시 파일 저장 디렉토리
    
    Returns:
        텍스트가 추가된 발화 구간 리스트
    """
    import whisper
    
    logger.info(f"[STT] Whisper 모델 로딩 중: {whisper_model_size}")
    model_whisper = whisper.load_model(whisper_model_size)
    
    # scipy를 사용하여 wav 파일 로드
    sr, wav_np = wavfile.read(audio_path)
    
    # numpy dtype을 확인하고 정규화
    if wav_np.dtype == np.int16:
        wav = torch.from_numpy(wav_np.astype(np.float32)) / 32768.0
    elif wav_np.dtype == np.int32:
        wav = torch.from_numpy(wav_np.astype(np.float32)) / 2147483648.0
    elif wav_np.dtype == np.float32:
        wav = torch.from_numpy(wav_np)
    else:
        wav = torch.from_numpy(wav_np.astype(np.float32))
    
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    
    if temp_dir is None:
        temp_dir = os.path.dirname(audio_path)
    os.makedirs(temp_dir, exist_ok=True)
    
    logger.info(f"[STT] {len(speech_segments)}개 발화 구간 텍스트 인식 중...")
    
    for i, seg in enumerate(speech_segments):
        # 해당 발화 구간 잘라내기
        clip = wav[:, seg["start_sample"]:seg["end_sample"]]
        temp_path = os.path.join(temp_dir, f"temp_segment_{i}.wav")
        # scipy를 사용하여 저장
        clip_np = (clip.squeeze(0).numpy() * 32768).astype('int16')
        wavfile.write(temp_path, sr, clip_np)
        
        # Whisper로 인식
        result = model_whisper.transcribe(temp_path, language=language)
        seg["text"] = result["text"].strip()
        
        # 임시 파일 삭제
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # 샘플 정보는 JSON에서 제외
        del seg["start_sample"]
        del seg["end_sample"]
    
    logger.info("[STT] 텍스트 인식 완료")
    return speech_segments


def extract_frames(
    input_video: str,
    output_dir: str,
    fps: float = 2.0,
    output_format: str = "jpg",
    quality: int = 2
) -> List[Dict]:
    """
    비디오에서 프레임 이미지를 추출합니다.
    
    Args:
        input_video: 입력 비디오 경로
        output_dir: 출력 디렉토리
        fps: 추출할 프레임 레이트 (기본: 2fps)
        output_format: 이미지 포맷 (jpg, png)
        quality: JPEG 품질 (1-31, 낮을수록 높은 품질)
    
    Returns:
        프레임 정보 리스트 [{frame_number, timestamp, filename}, ...]
    """
    logger.info(f"[Frame] 프레임 추출 중: {input_video} -> {output_dir} (fps={fps})")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 비디오 정보 가져오기
    probe = ffmpeg.probe(input_video)
    video_stream = next(
        (stream for stream in probe['streams'] if stream['codec_type'] == 'video'),
        None
    )
    
    if video_stream is None:
        raise ValueError("비디오 스트림을 찾을 수 없습니다.")
    
    duration = float(probe['format']['duration'])
    logger.info(f"[Frame] 비디오 길이: {duration:.2f}초")
    
    # 프레임 추출 (파일명: frame_XXXX_T.TT.jpg 형식)
    output_pattern = os.path.join(output_dir, f"frame_%04d.{output_format}")
    
    (
        ffmpeg
        .input(input_video)
        .filter('fps', fps=fps)
        .output(output_pattern, qscale=quality)
        .overwrite_output()
        .run(quiet=True)
    )
    
    # 프레임 정보 리스트 생성
    frame_files = sorted([f for f in os.listdir(output_dir) if f.startswith("frame_")])
    frame_info_list = []
    
    for i, filename in enumerate(frame_files):
        frame_number = i + 1
        timestamp = i / fps  # 프레임 번호에 따른 시간 계산
        
        # 파일명을 시간 정보가 포함된 이름으로 변경
        new_filename = f"frame_{frame_number:04d}_{timestamp:.2f}s.{output_format}"
        old_path = os.path.join(output_dir, filename)
        new_path = os.path.join(output_dir, new_filename)
        os.rename(old_path, new_path)
        
        frame_info_list.append({
            "frame_number": frame_number,
            "timestamp": round(timestamp, 3),
            "filename": new_filename
        })
    
    logger.info(f"[Frame] {len(frame_info_list)}개 프레임 추출 완료")
    return frame_info_list


def match_frames_to_segments(
    frame_info_list: List[Dict],
    speech_segments: List[Dict],
    silence_segments: List[Dict]
) -> Dict:
    """
    프레임과 대사/무음 구간을 시간 기준으로 매칭합니다.
    
    Args:
        frame_info_list: 프레임 정보 리스트
        speech_segments: 발화 구간 리스트
        silence_segments: 무음 구간 리스트
    
    Returns:
        매칭 정보가 포함된 딕셔너리
    """
    logger.info("[Match] 프레임-구간 매칭 중...")
    
    # 모든 세그먼트를 시간순으로 정렬
    all_segments = sorted(
        speech_segments + silence_segments,
        key=lambda x: x["start_time"]
    )
    
    # 각 프레임에 해당하는 세그먼트 찾기
    for frame in frame_info_list:
        timestamp = frame["timestamp"]
        frame["segment_type"] = None
        frame["segment_id"] = None
        
        for seg in all_segments:
            if seg["start_time"] <= timestamp < seg["end_time"]:
                frame["segment_type"] = seg["type"]
                frame["segment_id"] = seg["id"]
                break
    
    # 무음 구간별 프레임 매핑
    for silence in silence_segments:
        silence["frames"] = [
            f["filename"] for f in frame_info_list
            if silence["start_time"] <= f["timestamp"] < silence["end_time"]
        ]
        silence["frame_count"] = len(silence["frames"])
    
    logger.info("[Match] 매칭 완료")
    return {
        "frames": frame_info_list,
        "speech_segments": speech_segments,
        "silence_segments": silence_segments
    }


def process_video_for_gpt(
    input_video: str,
    output_dir: str,
    fps: float = 2.0,
    language: str = "en",
    whisper_model: str = "base",
    min_silence_duration: float = 0.5
) -> Dict:
    """
    GPT API용으로 비디오를 처리합니다.
    
    Args:
        input_video: 입력 비디오 경로
        output_dir: 출력 디렉토리
        fps: 프레임 추출 레이트 (기본: 2fps)
        language: 언어 코드 (en, ko)
        whisper_model: Whisper 모델 크기
        min_silence_duration: 무음 최소 시간 (초)
    
    Returns:
        처리 결과 딕셔너리
    """
    # 출력 디렉토리 설정
    video_name = Path(input_video).stem
    video_output_dir = os.path.join(output_dir, video_name)
    frames_dir = os.path.join(video_output_dir, "frames")
    audio_path = os.path.join(video_output_dir, "audio.wav")
    json_path = os.path.join(video_output_dir, f"{video_name}_data.json")
    
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
    
    logger.info(f"[Pipeline] 처리 시작: {input_video}")
    logger.info(f"[Pipeline] 출력 디렉토리: {video_output_dir}")
    
    # 1. 오디오 추출
    extract_audio(input_video, audio_path)
    
    # 2. VAD로 발화/무음 구간 탐지
    vad_model, get_speech_timestamps_fn = load_vad_model()
    speech_segments, silence_segments, sr = detect_speech_segments(
        audio_path, vad_model, get_speech_timestamps_fn, min_silence_duration
    )
    
    # 3. Whisper로 텍스트 인식
    speech_segments = transcribe_speech_segments(
        audio_path, speech_segments, sr, language, whisper_model, video_output_dir
    )
    
    # 4. 프레임 추출
    frame_info_list = extract_frames(input_video, frames_dir, fps)
    
    # 5. 프레임-구간 매칭
    matched_data = match_frames_to_segments(
        frame_info_list, speech_segments, silence_segments
    )
    
    # 6. 최종 JSON 구조 생성
    result = {
        "video_info": {
            "source_file": os.path.basename(input_video),
            "fps_extracted": fps,
            "language": language,
            "total_frames": len(frame_info_list),
            "total_speech_segments": len(speech_segments),
            "total_silence_segments": len(silence_segments)
        },
        "speech_segments": speech_segments,
        "silence_segments": silence_segments,
        "frames": frame_info_list
    }
    
    # 7. JSON 저장
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"[Pipeline] JSON 저장 완료: {json_path}")
    logger.info(f"[Pipeline] 프레임 저장 완료: {frames_dir}")
    
    # 결과 요약 출력
    print("\n" + "=" * 50)
    print("GPT API용 데이터 추출 완료")
    print("=" * 50)
    print(f"📁 출력 디렉토리: {video_output_dir}")
    print(f"📄 JSON 파일: {json_path}")
    print(f"🖼️  프레임 폴더: {frames_dir}")
    print(f"\n📊 요약:")
    print(f"   - 총 프레임 수: {len(frame_info_list)}")
    print(f"   - 발화 구간: {len(speech_segments)}개")
    print(f"   - 무음 구간: {len(silence_segments)}개 (AD 삽입 가능)")
    print("=" * 50)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="GPT API용 영상 데이터 추출 (대사/무음구간 JSON + 2fps 이미지)"
    )
    parser.add_argument("video_path", help="입력 비디오 파일 경로")
    parser.add_argument(
        "--output_dir", "-o",
        default="./gpt_data",
        help="출력 디렉토리 (기본: ./gpt_data)"
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=2.0,
        help="프레임 추출 레이트 (기본: 2fps)"
    )
    parser.add_argument(
        "--language", "-l",
        default="en",
        choices=["en", "ko"],
        help="오디오 언어 (기본: en)"
    )
    parser.add_argument(
        "--whisper_model", "-w",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper 모델 크기 (기본: medium)"
    )
    parser.add_argument(
        "--min_silence", "-s",
        type=float,
        default=0.5,
        help="무음 구간 최소 시간 (초, 기본: 0.5)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 로그 출력"
    )
    
    args = parser.parse_args()
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 비디오 처리
    process_video_for_gpt(
        input_video=args.video_path,
        output_dir=args.output_dir,
        fps=args.fps,
        language=args.language,
        whisper_model=args.whisper_model,
        min_silence_duration=args.min_silence
    )


if __name__ == "__main__":
    main()

