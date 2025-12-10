"""
GPT API를 활용한 Audio Description 생성 모듈 (v2)

개선사항:
1. 긴 무음구간 장면 분할 (최소 8초 기준)
2. Two-Pass 방식: 전체 맥락 파악 후 개별 AD 생성

입력:
- JSON 파일: 대사 및 무음 구간 정보
- 이미지 폴더: 2fps로 추출된 프레임 이미지

출력:
- AD JSON: 무음 구간별 Audio Description 텍스트
"""

import os
import json
import base64
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# .env 파일 로드
try:
    from dotenv import load_dotenv
    env_paths = [
        Path(__file__).parent.parent.parent / ".env",
        Path(__file__).parent.parent / "server" / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass

from openai import OpenAI

logger = logging.getLogger(__name__)

# GPT 모델 설정
GPT_MODEL = "gpt-4o"

# ============================================================
# 프롬프트 정의
# ============================================================

# 1st Pass: 전체 맥락 파악 프롬프트
CONTEXT_PROMPT_KO = """당신은 영상 분석 전문가입니다.
주어진 키프레임들과 전체 대본을 분석하여 영상의 전체적인 맥락을 파악해주세요.

## 전체 대본:
{full_transcript}

## 요청:
위 키프레임들({frame_count}장)과 대본을 분석하여 다음을 간략히 정리해주세요:
1. **알려진 콘텐츠 확인**: 이 영상이 알려진 영화, 드라마, 애니메이션 등인지 확인하세요. 
   알려진 작품이라면 제목과 시즌/에피소드 정보를 포함하세요.
2. **장소/배경**: 어디서 일어나는 이야기인가?
3. **등장인물**: 주요 인물들을 성별과 등장 순서로 구분
   - 이름이나 외모를 사용하지 않고 "남자1", "남자2", "여자1", "여자2" 형식으로 지칭
   - 동일 인물은 영상 전체에서 일관된 번호 유지
   - 예: "남자1", "남자2", "여자1" (이름이나 외모 특징 사용 금지)
4. **상황 요약**: 어떤 상황이 벌어지고 있는가? (2-3문장)
5. **분위기**: 전체적인 분위기 (긴장감, 슬픔 등)

JSON 형식으로 응답해주세요:
```json
{{
  "known_content": {{
    "is_known": true/false,
    "title": "작품 제목 (알려진 경우)",
    "season_episode": "시즌/에피소드 정보 (해당되는 경우)",
    "description": "작품에 대한 간략한 설명 (알려진 경우)"
  }},
  "location": "장소 설명",
  "characters": ["인물1 정보", "인물2 정보"],
  "situation": "상황 요약",
  "mood": "분위기"
}}
```
"""

CONTEXT_PROMPT_EN = """You are a video analysis expert.
Analyze the given keyframes and full transcript to understand the overall context.

## Full Transcript:
{full_transcript}

## Request:
Analyze the keyframes ({frame_count} frames) and transcript, then summarize:
1. **Known Content Check**: Identify if this is a known movie, TV series, animation, etc.
   If it's a known work, include the title and season/episode information.
2. **Location/Setting**: Where does this take place?
3. **Characters**: Identify main characters by gender and sequential number
   - Do NOT use names or physical descriptions; refer as "Man1", "Man2", "Woman1", "Woman2"
   - Maintain consistent numbering for the same character throughout the video
   - Example: "Man1", "Man2", "Woman1" (no names or appearance descriptions)
4. **Situation Summary**: What is happening? (2-3 sentences)
5. **Mood**: Overall mood (tension, sadness, etc.)

Respond in JSON format:
```json
{{
  "known_content": {{
    "is_known": true/false,
    "title": "Title of the work (if known)",
    "season_episode": "Season/Episode info (if applicable)",
    "description": "Brief description of the work (if known)"
  }},
  "location": "location description",
  "characters": ["character1 info", "character2 info"],
  "situation": "situation summary",
  "mood": "mood"
}}
```
"""

# 2nd Pass: AD 생성 프롬프트 (맥락 포함)
AD_PROMPT_KO = """당신은 시각장애인을 위한 화면 해설(Audio Description) 전문가입니다.

## 작품 정보:
{known_content_info}

## 영상 맥락 정보 (이미 시청자에게 공유됨):
- **장소**: {location}
- **등장인물**: {characters}
- **상황**: {situation}
- **분위기**: {mood}

## 직전 화면 해설:
{prev_ad}

## 이전 대사:
{prev_context}

## 다음 대사:
{next_context}

## 규칙:
1. **간결하게**: {duration}초 내에 읽을 수 있도록 짧고 핵심적인 문장
2. **시각적 정보만**: 소리로 알 수 없는, 눈으로만 볼 수 있는 정보
3. **현재 시제**: "~하고 있다", "~한다" 형태
4. **등장인물 지칭 (매우 중요)**: 
   - 등장인물의 이름을 절대 사용하지 마세요
   - 외모, 옷차림, 나이 등을 묘사하지 마세요
   - 성별과 등장 순서로만 지칭: "남자1", "남자2", "여자1", "여자2" 등
   - 동일 인물은 영상 전체에서 일관된 번호 유지
   - 예: "철수가 문을 연다" (X) → "남자1이 문을 연다" (O)
5. **사전 지식 활용**: 알려진 작품이라도 캐릭터 이름 대신 "남자1", "여자1" 형식을 사용하세요.
6. **중복 피하기 (매우 중요)**:
   - 영상 맥락 정보(장소, 분위기)는 이미 시청자가 알고 있으므로 매번 반복하지 마세요
   - 직전 화면 해설에서 언급한 내용(장소, 배경, 날씨, 환경)은 반복하지 마세요
   - 새로운 행동, 변화, 움직임에 집중하세요
   - 장면이 크게 바뀌지 않았다면 배경 묘사를 생략하고 인물의 동작만 설명하세요

## 요청:
위 이미지들({frame_count}장, 시간: {start_time}s ~ {end_time}s)에서 
{duration}초 안에 읽을 수 있는 화면 해설을 한국어로 작성해주세요.

응답은 화면 해설 텍스트만 출력하세요.
"""

AD_PROMPT_EN = """You are an Audio Description expert for visually impaired viewers.

## Content Information:
{known_content_info}

## Video Context (already known to the viewer):
- **Location**: {location}
- **Characters**: {characters}
- **Situation**: {situation}
- **Mood**: {mood}

## Previous Audio Description:
{prev_ad}

## Previous dialogue:
{prev_context}

## Next dialogue:
{next_context}

## Rules:
1. **Be concise**: Short sentences readable within {duration} seconds
2. **Visual only**: Information not available through sound
3. **Present tense**: Use present tense
4. **Character reference (CRITICAL)**: 
   - NEVER use character names
   - Do NOT describe physical appearance, clothing, or age
   - Refer to characters ONLY by gender and sequential number: "Man1", "Man2", "Woman1", "Woman2"
   - Maintain consistent numbering for the same character throughout the video
   - Example: "John opens the door" (X) → "Man1 opens the door" (O)
5. **Leverage prior knowledge**: Even for known content, use "Man1", "Woman1" format instead of character names.
6. **AVOID REPETITION (CRITICAL)**:
   - The video context (location, mood) is already known to the viewer - do NOT repeat it every time
   - Do NOT repeat information from the previous Audio Description (location, setting, weather, environment)
   - Focus on NEW actions, changes, and movements
   - If the scene hasn't changed significantly, skip background descriptions and focus on character actions only

## Request:
Write an Audio Description for the images ({frame_count} frames, time: {start_time}s ~ {end_time}s)
that can be read within {duration} seconds.

Output only the Audio Description text.
"""


# ============================================================
# 유틸리티 함수
# ============================================================

def encode_image_to_base64(image_path: str) -> str:
    """이미지를 base64로 인코딩합니다."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """이미지의 MIME 타입을 반환합니다."""
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    return media_types.get(ext, "image/jpeg")


def split_long_silence(silence_segment: Dict, frame_info_list: List[Dict], min_split_duration: float = 8.0) -> List[Dict]:
    """
    긴 무음구간을 장면 단위로 분할합니다.
    
    규칙:
    - 16초 미만: 분할 없음
    - 16초 이상: 8초 단위로 분할, 마지막은 나머지
    
    예시:
    - 15초 → [15초]
    - 16초 → [8초, 8초]
    - 23초 → [8초, 15초]
    - 30초 → [8초, 8초, 14초]
    
    Args:
        silence_segment: 무음 구간 정보
        frame_info_list: 전체 프레임 리스트
        min_split_duration: 최소 분할 단위 (기본: 8초)
    
    Returns:
        분할된 무음 구간 리스트
    """
    duration = silence_segment["duration"]
    start_time = silence_segment["start_time"]
    end_time = silence_segment["end_time"]
    original_id = silence_segment["id"]
    
    # 16초 미만이면 분할하지 않음
    if duration < min_split_duration * 2:
        return [silence_segment]
    
    # 분할 개수 계산: duration을 8초로 나눈 몫 (최소 2개)
    num_splits = max(2, int(duration // min_split_duration))
    
    # 마지막 구간이 8초 미만이 되지 않도록 조정
    # 예: 23초 → 8초 + 15초 (2개), 30초 → 8초 + 8초 + 14초 (3개)
    if duration - (num_splits - 1) * min_split_duration < min_split_duration:
        num_splits -= 1
    
    split_segments = []
    current_start = start_time
    
    for i in range(num_splits):
        if i == num_splits - 1:
            # 마지막 구간: 나머지 전부
            split_end = end_time
        else:
            # 8초 구간
            split_end = current_start + min_split_duration
        
        split_duration = split_end - current_start
        
        # 해당 구간의 프레임 찾기
        split_frames = [
            f["filename"] for f in frame_info_list
            if current_start <= f["timestamp"] < split_end
        ]
        
        split_segments.append({
            "id": f"{original_id}_{i+1}",
            "original_id": original_id,
            "start_time": round(current_start, 3),
            "end_time": round(split_end, 3),
            "duration": round(split_duration, 3),
            "type": "silence",
            "frames": split_frames,
            "frame_count": len(split_frames),
            "is_split": True,
            "split_index": i + 1,
            "total_splits": num_splits
        })
        
        current_start = split_end
    
    logger.info(f"[Split] 무음구간 {original_id} ({duration:.1f}초) → {num_splits}개로 분할")
    for seg in split_segments:
        logger.info(f"  - {seg['id']}: {seg['start_time']:.1f}s ~ {seg['end_time']:.1f}s ({seg['duration']:.1f}초, {seg['frame_count']}프레임)")
    
    return split_segments


def get_context_text(segments: List[Dict], current_start_time: float, direction: str = "prev") -> str:
    """현재 무음 구간 전후의 대사 컨텍스트를 가져옵니다."""
    context_parts = []
    
    speech_segments = [s for s in segments if s.get("type") == "speech" and s.get("text")]
    
    if direction == "prev":
        # 현재 시작 시간보다 이전에 끝나는 대사들
        prev_speeches = [s for s in speech_segments if s["end_time"] <= current_start_time]
        prev_speeches = sorted(prev_speeches, key=lambda x: x["end_time"], reverse=True)[:2]
        for s in reversed(prev_speeches):
            context_parts.append(f"[{s['start_time']:.1f}s] {s['text']}")
    else:
        # 현재 끝 시간보다 이후에 시작하는 대사들 (current_start_time은 실제로 end_time을 전달)
        next_speeches = [s for s in speech_segments if s["start_time"] >= current_start_time]
        next_speeches = sorted(next_speeches, key=lambda x: x["start_time"])[:2]
        for s in next_speeches:
            context_parts.append(f"[{s['start_time']:.1f}s] {s['text']}")
    
    return "\n".join(context_parts) if context_parts else "(없음 / None)"


# ============================================================
# 1st Pass: 전체 맥락 파악
# ============================================================

def analyze_video_context(
    client: OpenAI,
    frames_dir: str,
    frame_info_list: List[Dict],
    speech_segments: List[Dict],
    language: str = "ko",
    num_keyframes: int = 15
) -> Dict:
    """
    전체 영상의 맥락을 파악합니다 (1st Pass).
    
    Args:
        client: OpenAI 클라이언트
        frames_dir: 프레임 이미지 디렉토리
        frame_info_list: 전체 프레임 리스트
        speech_segments: 발화 구간 리스트
        language: 언어 (ko, en)
        num_keyframes: 사용할 키프레임 수
    
    Returns:
        맥락 정보 딕셔너리
    """
    logger.info("[Context] 1st Pass: 전체 맥락 파악 중...")
    
    # 전체 대본 구성
    full_transcript = "\n".join([
        f"[{s['start_time']:.1f}s] {s.get('text', '')}"
        for s in sorted(speech_segments, key=lambda x: x['start_time'])
        if s.get('text')
    ])
    
    # 키프레임 균등 샘플링
    if len(frame_info_list) > num_keyframes:
        step = len(frame_info_list) / num_keyframes
        selected_indices = [int(i * step) for i in range(num_keyframes)]
        keyframes = [frame_info_list[i] for i in selected_indices]
    else:
        keyframes = frame_info_list
    
    # 프롬프트 선택
    prompt_template = CONTEXT_PROMPT_KO if language == "ko" else CONTEXT_PROMPT_EN
    prompt = prompt_template.format(
        full_transcript=full_transcript[:3000],  # 토큰 제한
        frame_count=len(keyframes)
    )
    
    # 메시지 구성
    content = [{"type": "text", "text": prompt}]
    
    for frame in keyframes:
        image_path = os.path.join(frames_dir, frame["filename"])
        if os.path.exists(image_path):
            base64_image = encode_image_to_base64(image_path)
            media_type = get_image_media_type(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{base64_image}",
                    "detail": "low"
                }
            })
    
    # GPT API 호출
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=500,
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # JSON 파싱 시도
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            context = json.loads(json_match.group())
        else:
            # 파싱 실패 시 기본값
            context = {
                "known_content": {"is_known": False},
                "location": "알 수 없음",
                "characters": ["인물1", "인물2"],
                "situation": response_text[:200],
                "mood": "알 수 없음"
            }
        
        # known_content 기본값 보장
        if "known_content" not in context:
            context["known_content"] = {"is_known": False}
        
        known = context.get("known_content", {})
        logger.info(f"[Context] 맥락 파악 완료:")
        if known.get("is_known"):
            logger.info(f"  - 🎬 알려진 작품: {known.get('title', 'N/A')} ({known.get('season_episode', 'N/A')})")
        logger.info(f"  - 장소: {context.get('location', 'N/A')}")
        logger.info(f"  - 인물: {context.get('characters', [])}")
        logger.info(f"  - 상황: {context.get('situation', 'N/A')[:50]}...")
        
        return context
        
    except Exception as e:
        logger.error(f"[Context] 맥락 파악 실패: {e}")
        return {
            "known_content": {"is_known": False},
            "location": "알 수 없음",
            "characters": ["인물1", "인물2"],
            "situation": "상황 파악 실패",
            "mood": "알 수 없음"
        }


# ============================================================
# 2nd Pass: 개별 AD 생성
# ============================================================

def generate_ad_for_segment(
    client: OpenAI,
    segment: Dict,
    frames_dir: str,
    all_segments: List[Dict],
    context: Dict,
    prev_ad: str = "",
    language: str = "ko",
    max_frames: int = 10
) -> str:
    """
    단일 세그먼트에 대한 AD를 생성합니다 (2nd Pass).
    
    Args:
        prev_ad: 직전 세그먼트의 AD 텍스트 (중복 방지용)
    """
    frame_files = segment.get("frames", [])
    
    if not frame_files:
        logger.warning(f"[AD] 세그먼트 {segment['id']}에 프레임이 없습니다.")
        return ""
    
    # 프레임 샘플링
    if len(frame_files) > max_frames:
        step = len(frame_files) / max_frames
        selected_indices = [int(i * step) for i in range(max_frames)]
        frame_files = [frame_files[i] for i in selected_indices]
    
    # 컨텍스트
    prev_context = get_context_text(all_segments, segment["start_time"], "prev")
    next_context = get_context_text(all_segments, segment["end_time"], "next")
    
    # 알려진 콘텐츠 정보 구성
    known_content = context.get("known_content", {})
    if known_content.get("is_known"):
        known_content_info = f"""이 영상은 알려진 작품입니다.
- 제목: {known_content.get('title', 'N/A')}
- 시즌/에피소드: {known_content.get('season_episode', 'N/A')}
- 설명: {known_content.get('description', 'N/A')}
→ 당신이 이 작품에 대해 알고 있는 지식을 활용하여 더 정확한 화면 해설을 작성하세요."""
    else:
        known_content_info = "이 영상은 알려지지 않은 콘텐츠입니다. 화면에 보이는 정보만을 기반으로 해설하세요."
    
    # 직전 AD 정보 구성
    if prev_ad:
        prev_ad_text = f'"{prev_ad}"\n→ 위 내용은 이미 언급했으므로 반복하지 마세요. 새로운 행동/변화에 집중하세요.'
    else:
        prev_ad_text = "(첫 번째 화면 해설입니다. 간략하게 장면을 설정하세요.)"
    
    # 프롬프트
    prompt_template = AD_PROMPT_KO if language == "ko" else AD_PROMPT_EN
    prompt = prompt_template.format(
        known_content_info=known_content_info,
        location=context.get("location", "알 수 없음"),
        characters=", ".join(context.get("characters", [])),
        situation=context.get("situation", ""),
        mood=context.get("mood", ""),
        prev_ad=prev_ad_text,
        prev_context=prev_context,
        next_context=next_context,
        duration=segment["duration"],
        frame_count=len(frame_files),
        start_time=segment["start_time"],
        end_time=segment["end_time"]
    )
    
    # 메시지 구성
    content = [{"type": "text", "text": prompt}]
    
    for frame_file in frame_files:
        image_path = os.path.join(frames_dir, frame_file)
        if os.path.exists(image_path):
            base64_image = encode_image_to_base64(image_path)
            media_type = get_image_media_type(image_path)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{base64_image}",
                    "detail": "low"
                }
            })
    
    # GPT API 호출
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=200,
            temperature=0.3
        )
        
        ad_text = response.choices[0].message.content.strip()
        logger.info(f"[AD] 세그먼트 {segment['id']} ({segment['start_time']:.1f}s~{segment['end_time']:.1f}s): {ad_text[:50]}...")
        return ad_text
        
    except Exception as e:
        logger.error(f"[AD] GPT API 오류: {e}")
        return ""


# ============================================================
# 메인 함수
# ============================================================

def generate_ad_from_extracted_data(
    data_dir: str,
    api_key: str = None,
    language: str = "ko",
    min_duration: float = 2.5,
    max_frames_per_segment: int = 10,
    min_split_duration: float = 8.0
) -> Dict:
    """
    추출된 데이터를 사용하여 AD를 생성합니다.
    
    Args:
        data_dir: 데이터 디렉토리
        api_key: OpenAI API 키
        language: 언어 (ko, en)
        min_duration: AD 생성 최소 무음 구간 시간
        max_frames_per_segment: 세그먼트당 최대 프레임 수
        min_split_duration: 장면 분할 최소 단위 (기본: 8초)
    """
    # API 키 설정
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
    
    client = OpenAI(api_key=api_key)
    
    # 데이터 로드
    data_dir = Path(data_dir)
    json_files = list(data_dir.glob("*_data.json"))
    
    if not json_files:
        raise FileNotFoundError(f"데이터 JSON 파일을 찾을 수 없습니다: {data_dir}")
    
    json_path = json_files[0]
    frames_dir = data_dir / "frames"
    
    logger.info(f"[AD] 데이터 로딩: {json_path}")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    speech_segments = data.get("speech_segments", [])
    silence_segments = data.get("silence_segments", [])
    frame_info_list = data.get("frames", [])
    
    # ============================================================
    # 1st Pass: 전체 맥락 파악
    # ============================================================
    context = analyze_video_context(
        client=client,
        frames_dir=str(frames_dir),
        frame_info_list=frame_info_list,
        speech_segments=speech_segments,
        language=language
    )
    
    # ============================================================
    # 긴 무음구간 분할
    # ============================================================
    logger.info("[Split] 긴 무음구간 분할 처리 중...")
    
    all_ad_segments = []
    for silence in silence_segments:
        if silence["duration"] >= min_duration:
            # 긴 구간은 분할
            split_segments = split_long_silence(
                silence_segment=silence,
                frame_info_list=frame_info_list,
                min_split_duration=min_split_duration
            )
            all_ad_segments.extend(split_segments)
    
    logger.info(f"[AD] 총 AD 생성 대상: {len(all_ad_segments)}개 세그먼트")
    
    # 시간순 정렬
    all_ad_segments = sorted(all_ad_segments, key=lambda x: x["start_time"])
    all_segments = sorted(speech_segments + silence_segments, key=lambda x: x["start_time"])
    
    # ============================================================
    # 2nd Pass: 개별 AD 생성
    # ============================================================
    logger.info("[AD] 2nd Pass: 개별 AD 생성 중...")
    
    audio_descriptions = []
    prev_ad = ""  # 이전 AD 추적 (중복 방지용)
    
    for segment in all_ad_segments:
        ad_text = generate_ad_for_segment(
            client=client,
            segment=segment,
            frames_dir=str(frames_dir),
            all_segments=all_segments,
            context=context,
            prev_ad=prev_ad,
            language=language,
            max_frames=max_frames_per_segment
        )
        
        if ad_text:
            prev_ad = ad_text  # 다음 세그먼트를 위해 저장
            audio_descriptions.append({
                "id": segment["id"],
                "original_id": segment.get("original_id", segment["id"]),
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "duration_sec": segment["duration"],
                "description": ad_text,
                "is_split": segment.get("is_split", False),
                "split_info": f"{segment.get('split_index', 1)}/{segment.get('total_splits', 1)}" if segment.get("is_split") else None
            })
    
    # 결과 구성
    result = {
        "video_info": data.get("video_info", {}),
        "video_context": context,
        "full_transcript": [
            {
                "time": f"{s['start_time']:.1f}",
                "speaker": "Speaker",
                "text": s.get("text", "")
            }
            for s in sorted(speech_segments, key=lambda x: x["start_time"])
        ],
        "audio_descriptions": audio_descriptions
    }
    
    # 결과 저장
    output_path = data_dir / f"{data_dir.name}.ad.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"[AD] 결과 저장: {output_path}")
    
    # 요약 출력
    print("\n" + "=" * 60)
    print("GPT AD 생성 완료 (v2 - 장면분할 + Two-Pass)")
    print("=" * 60)
    print(f"📄 결과 파일: {output_path}")
    print(f"📊 생성된 AD: {len(audio_descriptions)}개")
    print(f"📍 맥락 정보:")
    print(f"   - 장소: {context.get('location', 'N/A')}")
    print(f"   - 인물: {', '.join(context.get('characters', []))}")
    print(f"   - 분위기: {context.get('mood', 'N/A')}")
    
    # 분할된 구간 정보
    split_count = sum(1 for ad in audio_descriptions if ad.get("is_split"))
    if split_count > 0:
        print(f"✂️  분할된 구간: {split_count}개")
    
    print("=" * 60)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="GPT API를 활용한 Audio Description 생성 (v2)"
    )
    parser.add_argument(
        "data_dir",
        help="extract_for_gpt.py로 생성된 데이터 디렉토리 경로"
    )
    parser.add_argument(
        "--api_key", "-k",
        help="OpenAI API 키 (또는 OPENAI_API_KEY 환경변수 사용)"
    )
    parser.add_argument(
        "--language", "-l",
        default="ko",
        choices=["ko", "en"],
        help="AD 생성 언어 (기본: ko)"
    )
    parser.add_argument(
        "--min_duration", "-d",
        type=float,
        default=2.5,
        help="AD 생성 최소 무음 구간 시간 (초, 기본: 2.5)"
    )
    parser.add_argument(
        "--max_frames", "-m",
        type=int,
        default=10,
        help="세그먼트당 최대 프레임 수 (기본: 10)"
    )
    parser.add_argument(
        "--min_split", "-s",
        type=float,
        default=8.0,
        help="장면 분할 최소 단위 (초, 기본: 8)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 로그 출력"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    generate_ad_from_extracted_data(
        data_dir=args.data_dir,
        api_key=args.api_key,
        language=args.language,
        min_duration=args.min_duration,
        max_frames_per_segment=args.max_frames,
        min_split_duration=args.min_split
    )


if __name__ == "__main__":
    main()
