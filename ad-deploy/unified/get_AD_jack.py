"""
Jack Mutli-stage AD Generation Module (Server Version)
======================================================
Unified implementation of the "Jack Method" (Multi-stage Context Fusion).
Based on jack_ad_ko.py and jack_ad_en.py updates (Dec 2025).

Pipeline:
1. Parallel Input Generation: Metadata, Core AD (Silent Segments), STT (Transcript)
2. Final Integration: Fills gaps, enriches context, and merges inputs.
3. Stage 2 Compression: Duration-based text Refinement using Gemini.

Model: gemini-3-pro-preview
"""

import logging
import os
import json
import re
import asyncio
import time
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

try:
    from google import genai
    from google.genai import types
    from google.genai.errors import ServerError
except ImportError as e:
    try:
        import google.generativeai as genai
        import google.generativeai.types as types
        ServerError = Exception  # Fallback
    except ImportError:
        raise ImportError(
            "google-genai package is not installed. "
            "Please install it with: pip install google-genai"
        ) from e

logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================
GEMINI_MODEL_VISION = "gemini-3-pro-preview"
GEMINI_MODEL_TEXT = "gemini-3-pro-preview"

# Thread pool for async operations
executor = ThreadPoolExecutor(max_workers=8)

# ==============================================================================
# Prompts - Korean (from jack_ad_ko.py)
# ==============================================================================
PROMPT_METADATA_KO = """
당신은 영상 스토리 분석가입니다.

목표:
영상에서 시각적으로 확인 가능한 핵심 메타데이터를 추출하여,
고품질 Audio Description 제작 참고용 JSON으로 출력하십시오.

규칙:
- 등장인물 이름은 화면에서 식별 가능한 경우에만 기입.
- 이름을 알 수 없으면 "화자" 등 일반 명칭 사용.
- 제목은 제공된 경우만 참고, 추측 금지.
- 영상에서 확인 불가한 정보 절대 생성 금지.
- 감정·내면 묘사는 시각적 근거가 명확할 때만 제한적으로 허용.
- JSON 외 텍스트, 주석, 설명 출력 금지.

출력 형식(변경 불가):
{
  "video_title": "제공된 경우에만 기입, 없으면 \\"null\\"",
  "overall_summary": "영상 전체 흐름을 시각 기반으로 간결하게 설명",
  "scenes": [
    {
      "scene_id": "Scene-1",
      "start_time": "0:00.0",
      "end_time": "0:05.8",
      "summary": "스토리 이해에 필요한 시각 정보 요약",
      "characters": [
        {
          "id": "char_1",
          "name": "화면에서 식별된 이름(불가 시 \\"null\\")",
          "appearance": "외형 정보",
          "visible_emotion": "표정 기반 감정(명확 시만)"
        }
      ],
      "visible_actions": [
        "확실히 보이는 주요 행동"
      ],
      "relationships": [
        "시각 근거 있는 관계만"
      ],
      "visual_focus": "장면에서 시각적 중심 요소"
    }
  ]
}
"""

PROMPT_AD_KO = """
당신의 최우선 임무는 영상의 오디오 분석을 기반으로,
대사가 포함되지 않은 모든 무음 구간을 정확히 탐지하는 것입니다.

[규칙]

무음 구간 탐지
1) 오디오 신호 분석을 기준으로 대사·내레이션·말소리 포함 구간은 제외
2) 2.5초 이상 지속된 무음 구간은 반드시 모두 탐지하여 JSON 배열에 포함
3) 2.5초 미만 구간은 인접 구간과 자연스럽게 연결될 경우에만 병합 가능
4) 무음 구간을 하나만 선택하거나 임의로 축소 또는 생략 금지
5) 무음 구간의 시작/끝 시각을 임의로 조정하거나 생성 금지

타임스탬프
6) start_time, end_time을 실제 무음 시작과 끝으로 정확하게 기입
7) duration_sec = end_time - start_time을 반드시 정확하게 계산하여 기입
8) 타임스탬프의 순서가 실제 영상 진행과 동일해야 하며, 겹치거나 누락된 시간 존재 금지

음성해설(Description) 작성
9) 무음 구간 동안 화면에서 직접 확인 가능한 핵심 시각정보를 상세하게 묘사
10) 메타데이터(등장인물, 배경, 상황)는 화면으로 검증되는 경우에만 활용
11) 장면 내 변화가 있으면 한 문장에 압축하지 말고 핵심 요소 우선 배치
12) 감정·내면 묘사 금지 (표정, 행동 등 시각적 근거가 있을 경우만 제한적으로 허용)
13) duration 내에서 자연스럽게 읽을 수 있는 분량으로 작성
14) 화면에 없는 정보 창작 금지

정보 활용 범위
15) 주변 대화 전후 맥락을 참고해 장면의 이해도를 높일 수 있으나
    새로운 사건, 인물 정보는 추가 금지
16) 카메라 움직임(줌, 패닝), 소품 사용, 몸짓·행동·표정 등
    시각적 변화는 적극적으로 기술

출력 형식
17) JSON 외 다른 텍스트, 주석, 헤더 절대 금지
18) audio_descriptions 항목은 반드시 배열 형태로 제공 (길이 1 금지)

[출력 JSON 샘플]
"audio_descriptions": [
  {
    "start_time": "0:03.5",
    "end_time": "0:06.1",
    "duration_sec": 2.6,
    "description": "인물이 커피잔을 들어 입에 가져가고 조용히 한 모금 마신다."
  }
]
"""

PROMPT_STT_KO = """
**미션(MISSION):**
입력 비디오를 분석하여 영상 내 모든 음성 대사와 사운드를 JSON으로 기록하시오.
무음 구간(2.5초 이상)과 함께 Audio Description을 생성.
JSON 형식만 출력.

예시:
{
  "full_transcript": [
    {"time": "0:01.2", "speaker": "화자1", "text": "대사 내용"},
    {"time": "0:06.6", "speaker": "[Sound]", "text": "차 문 닫히는 소리"}
  ]
}
"""

FINAL_PROMPT_KO = """
입력:
1) 영상 메타데이터 — 장면 요소, 등장인물, 소품, 시각적 포커스, 변화 정보
2) 기존 Audio Description — 무음 구간 중심
3) STT 결과 — 발화 및 주요 소리 전체 기록

목표:
- Core AD, 메타데이터, STT 정보를 종합하여 최종 Audio Description JSON 생성
- 무음 구간 동안 화면에서 확인 가능한 시각적 세부 정보 묘사 (행동, 소품, 표정, 카메라 움직임 등)
- 메타데이터 활용해 맥락 풍부화 및 누락 요소 보완
- STT 발화는 맥락 참고용으로만 활용, 짧은 의미 없는 감탄사나 효과음 무시
- 중복 제거, duration_sec 내 자연스럽게 읽히도록 문장 작성
- start_time, end_time, duration_sec 정확히 유지
- 화면에서 확인 가능한 정보만 사용, 추측 금지
- 출력은 반드시 JSON만

규칙:
1) Core AD에서 누락된 요소는 메타데이터/시각 포커스에서 확인 가능하면 반드시 포함
2) STT 무음 구간을 참조해 Core AD 보완
3) 반복 내용 통합, 핵심 정보만 유지
4) 추정 감정이나 내면 묘사 금지
5) **발화가 없는 구간 ≥2.5초에 대해서만 description 생성**
6) 의미 없는 감탄사나 비정보성 효과음은 무음으로 간주

출력 예시:
"audio_descriptions": [
  {{
    "start_time": "0:03.5",
    "end_time": "0:06.1",
    "duration_sec": 2.6,
    "description": "한 사람이 커피잔을 들어 한 모금 마신다."
  }}
]

[영상 메타데이터]
{metadata}

[기존 AD]
{core_ad}

[STT 결과]
{stt_result}
"""

PROMPT_STAGE2_KO = """
당신은 한국어 Audio Description 전문가입니다.

아래 JSON 배열의 각 항목에 대해:
- "description"을 "allowed_chars" 안에서 3인칭 완전 문장으로 축약
- 핵심 시각 정보만 유지
- 추측 금지
- 한 항목당 한 문장
- JSON 구조(id, start_time, end_time 등)는 유지
- 출력은 JSON 전체만! 여분의 설명, 문장 금지!

입력:
{segments_json}
"""

# ==============================================================================
# Prompts - English (from jack_ad_en.py)
# ==============================================================================
PROMPT_METADATA_EN = """
You are a video story analyst.

Goal:
Extract visually verifiable core metadata from the video and output it
as JSON to support high-quality Audio Description production.

Rules:
- Character names must be included only when they are visually identifiable.
- If a name cannot be determined from the video, use generic labels such as "speaker".
- Only use the title if it is explicitly shown; no assumptions.
- Do NOT generate any information that cannot be visually confirmed.
- Emotional or internal state descriptions are allowed only when there is a clear visual basis (e.g., facial expression).
- Output must be ONLY JSON — no additional text, comments, or explanation.

Output format (strictly unchanged):
{
  "video_title": "Include only if visibly confirmed, otherwise \\"null\\"",
  "overall_summary": "Concise visual summary of the entire video",
  "scenes": [
    {
      "scene_id": "Scene-1",
      "start_time": "0:00.0",
      "end_time": "0:05.8",
      "summary": "Summary of visually essential story information",
      "characters": [
        {
          "id": "char_1",
          "name": "Visually confirmed name (null if unknown)",
          "appearance": "Visible physical features",
          "visible_emotion": "Emotion based on clear facial expression only"
        }
      ],
      "visible_actions": [
        "Major clearly visible actions"
      ],
      "relationships": [
        "Only relationships with clear visual evidence"
      ],
      "visual_focus": "Primary visual focus of the scene"
    }
  ]
}
"""

PROMPT_AD_EN = """
Your primary objective is to accurately detect all silent segments in the video
based on audio analysis, excluding any portion containing dialogue or speech.

[RULES]

Silent Segment Detection
1) Exclude any segment containing speech, narration, or human vocal sounds
2) Detect every silent segment lasting at least 2.5 seconds and include them in a JSON array
3) Segments shorter than 2.5 seconds may be merged only if they naturally connect with adjacent silent regions
4) Do not select, shorten, or omit any valid silent segment
5) Do not modify or invent start/end times

Timestamps
6) Use the exact start_time and end_time of each silent segment
7) duration_sec must be calculated precisely as end_time - start_time
8) Timestamps must follow chronological order without overlap or missing time

Audio Description Creation
9) Describe only the essential on-screen visual information occurring during silence
10) Metadata (characters, setting, context) may be used only if visually verified
11) If multiple visual changes occur, avoid squeezing them into a single sentence — prioritize clarity
12) No emotional or internal state assumptions (only describe visible facial expressions or actions)
13) Ensure the description length is readable within the segment duration
14) Never invent details not seen on screen

Information Use
15) Spoken content before/after the silent segment may be referenced only to enhance clarity,
    but do not introduce new events or characters not visually confirmed
16) Actively include visible changes such as camera movement (zoom, panning),
    gestures, facial expressions, and interaction with objects

Output Format
17) Output must be strictly JSON — no explanations, comments, or headers
18) "audio_descriptions" must be an array containing multiple items (not length 1)

[OUTPUT JSON SAMPLE]
"audio_descriptions": [
  {
    "start_time": "0:03.5",
    "end_time": "0:06.1",
    "duration_sec": 2.6,
    "description": "A character lifts a coffee mug and quietly takes a sip."
  }
]
"""

PROMPT_STT_EN = """
**MISSION:**
Analyze the given video and extract all spoken dialogue and relevant sound events.
Record them in JSON format.
Include silent segments longer than 2.5 seconds as well.
Output must be JSON only.

Example:
{
  "full_transcript": [
    {"time": "0:01.2", "speaker": "Speaker 1", "text": "Dialogue content"},
    {"time": "0:06.6", "speaker": "[Sound]", "text": "Car door closing"}
  ]
}
"""

FINAL_PROMPT_EN = """
INPUT:
1) Video Metadata — scene elements, characters, objects, visual focus, changes
2) Core AD — existing Audio Description for silent segments
3) STT Result — transcript of spoken lines and significant sounds

OBJECTIVE:
- Generate FINAL Audio Description JSON by integrating Core AD, Metadata, and STT
- Describe visual details during silent segments (actions, props, expressions, camera motion)
- Use Metadata to enrich context and recover missing elements
- Use STT dialogue only for context; ignore short meaningless exclamations or non-informative sounds
- Remove redundancy, write readable sentences fitting within duration_sec
- Maintain exact start_time, end_time, duration_sec
- Screen-visible info only, no guesses
- Output strictly JSON

RULES:
1) Include details missing in Core AD if visible in Metadata or visual focus
2) Reference STT silent sections to complement Core AD
3) Merge repeated content, keep essentials
4) No inferred emotions or inner thoughts
5) Generate descriptions **only for segments without spoken dialogue ≥2.5 seconds**
6) Treat meaningless exclamations or non-informative sounds as silence

OUTPUT EXAMPLE:
"audio_descriptions": [
  {{
    "start_time": "0:03.5",
    "end_time": "0:06.1",
    "duration_sec": 2.6,
    "description": "A person lifts a coffee cup and takes a sip."
  }}
]

[VIDEO METADATA]
{metadata}

[CORE AD]
{core_ad}

[STT RESULT]
{stt_result}
"""

PROMPT_STAGE2_EN = """
You are a professional English Audio Description writer.

Task:
- Rewrite each "description" within the "allowed_chars" limit
- Third-person objective narration only
- Full English sentence ending with a period
- Only essential visual information
- No thoughts, feelings, or assumptions
- EXACTLY one sentence per segment
- Maintain JSON structure except update "description"
- OUTPUT JSON ONLY, NO commentary

Input:
{segments_json}
"""


# ==============================================================================
# Async Helpers
# ==============================================================================
def get_gemini_client():
    """Get Gemini client with API key from environment."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required")
    return genai.Client(api_key=api_key)


async def run_async(func, *args, **kwargs):
    """Run a synchronous function in thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor, lambda: func(*args, **kwargs)
    )

# -------------------------------------------------------
# Gemini 호출 기본 Config
# -------------------------------------------------------
BASE_CONFIG = {
    "temperature": 0,
    "system_instruction": "이전 api 사용으로 인한 대화 절대 참조 금지",
    "top_k": 1,
    "top_p": 0.00001,  # 결정적 출력
    "thinking_config": {"thinking_budget": 8192},
}

async def gemini_call(client, contents, model, temp=0, retries=3, delay=5):
    """Call Gemini API with retries and thinking mode support."""
    
    # Config setup
    config_kwargs = BASE_CONFIG.copy()
    config_kwargs["temperature"] = temp
    config_kwargs["max_output_tokens"] = 65536
    
    # Explicitly set ThinkingConfig for compatibility
    if hasattr(types, 'ThinkingConfig'):
        config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=8192)
    else:
        config_kwargs["thinking_config"] = {"thinking_budget": 8192}

    for attempt in range(retries):
        try:
            response = await run_async(
                client.models.generate_content,
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs)
            )
            
            # Check for MAX_TOKENS issue
            try:
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'finish_reason') and 'MAX_TOKENS' in str(candidate.finish_reason):
                             logger.error(f"[Jack] ⚠️ MAX_TOKENS reached! Gemini 3 used all tokens.")
            except:
                pass

            return response.text
            
        except (ServerError, Exception) as e:
            logger.warning(f"[Jack] Gemini error (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                raise Exception(f"Gemini failed after {retries} retries: {e}")


async def wait_for_file_active(client, file_obj, timeout=300, interval=5):
    """Wait for uploaded file to become active."""
    start = time.time()
    while time.time() - start < timeout:
        f = await run_async(client.files.get, name=file_obj.name)
        if f.state == "ACTIVE":
            logger.info(f"[Jack] File {file_obj.name} is now ACTIVE")
            return f
        logger.info(f"[Jack] Waiting for file to become ACTIVE: {file_obj.name} ...")
        await asyncio.sleep(interval)
    raise TimeoutError(f"File {file_obj.name} did not become ACTIVE within {timeout}s")


# ==============================================================================
# JSON Extraction
# ==============================================================================
def extract_json_from_response(text: str) -> Dict:
    """Extract JSON from API response text."""
    if not text:
        raise ValueError("Empty response text")
    
    # Remove markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
    else:
        candidate = text.strip()
    
    # Find JSON object
    match = re.search(r'\{[\s\S]*\}', candidate)
    if not match:
         # Try finding array if object not found (though prompt asks for object)
         match_arr = re.search(r'\[[\s\S]*\]', candidate)
         if match_arr:
             try:
                 return json.loads(match_arr.group(0))
             except:
                 pass
         raise ValueError("No JSON object found in response")
         
    json_str = match.group(0)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"[Jack] JSON parse error: {e}")
        # Try simplistic repair
        json_str = repair_json(json_str)
        return json.loads(json_str)


def repair_json(json_str: str) -> str:
    """Attempt to repair truncated or malformed JSON."""
    brace_count = json_str.count('{') - json_str.count('}')
    bracket_count = json_str.count('[') - json_str.count(']')
    json_str = re.sub(r',\s*$', '', json_str.rstrip())
    json_str += ']' * bracket_count
    json_str += '}' * brace_count
    return json_str


# ==============================================================================
# Stage 2: Compression
# ==============================================================================
async def stage2_compression(client, ad_segments: List[Dict], lang: str = "ko") -> List[Dict]:
    """
    Refines AD text based on duration constraints using Gemini.
    """
    logger.info(f"[Jack] Starting Stage 2 Compression (Lang: {lang})")
    
    chars_per_sec = 7.0 if lang == "ko" else 15.0  # Derived from reference
    prompt_template = PROMPT_STAGE2_KO if lang == "ko" else PROMPT_STAGE2_EN
    
    # 1. Calculate allowed characters
    for seg in ad_segments:
        seg["allowed_chars"] = max(1, int(seg.get("duration_sec", 0) * chars_per_sec))
    
    # 2. Prepare bulk request
    segments_json = json.dumps(ad_segments, ensure_ascii=False, indent=2)
    prompt = prompt_template.format(segments_json=segments_json)
    
    # 3. Call Gemini
    try:
        response_text = await gemini_call(
            client, 
            [types.Part(text=prompt)], 
            GEMINI_MODEL_TEXT, 
            temp=0.2
        )
        
        # 4. Parse result
        compressed_json = extract_json_from_response(response_text)
        
        # Determine compressed list
        if isinstance(compressed_json, list):
            compressed_list = compressed_json
        elif isinstance(compressed_json, dict) and "audio_descriptions" in compressed_json:
             compressed_list = compressed_json["audio_descriptions"]
        else:
             logger.warning("[Jack] Stage 2 returned unexpected format, skipping compression.")
             return ad_segments

        # 5. Merge results (keep timestamps from original, update text)
        final_output = []
        # Fallback if length mismatch
        if len(compressed_list) != len(ad_segments):
            logger.warning(f"[Jack] Stage 2 count mismatch (Original: {len(ad_segments)}, Compressed: {len(compressed_list)}). Trying to match by ID or Index.")
        
        for idx, before in enumerate(ad_segments):
            # Try to find matching after segment
            after = None
            if idx < len(compressed_list):
                 after = compressed_list[idx]
            
            # Use original if match failed or structure broken
            if not after:
                after = before
                
            before_desc = before.get("description", "")
            after_desc = after.get("description", str(after)) # Fallback if not dict
            
            # Basic validation: ensure after_desc is a string
            if not isinstance(after_desc, str):
                after_desc = before_desc

            allowed = before["allowed_chars"]
            before_chars = len(before_desc)
            after_chars = len(after_desc)

            final_output.append({
                "id": before.get("id"),
                "start_time": before.get("start_time"),
                "end_time": before.get("end_time"),
                "duration_sec": before.get("duration_sec"),
                "description": after_desc,
                "allowed_chars": allowed,
                "current_chars_before": before_chars,
                "current_chars_after": after_chars,
                "compressed_by_chars": after_chars < before_chars,
                "exceeds_limit": after_chars > allowed
            })
            
        logger.info("[Jack] Stage 2 Compression completed")
        return final_output

    except Exception as e:
        logger.error(f"[Jack] Stage 2 Compression failed: {e}")
        return ad_segments


# ==============================================================================
# Main Processing
# ==============================================================================
async def process_video_async(video_path: str, lang: str = "ko") -> Tuple[Dict, List[Dict]]:
    logger.info(f"[Jack] Starting Multi-stage AD generation for: {video_path}")
    
    client = get_gemini_client()
    
    # Select prompts
    if lang == "ko":
        prompt_metadata = PROMPT_METADATA_KO
        prompt_ad = PROMPT_AD_KO
        prompt_stt = PROMPT_STT_KO
        final_prompt_template = FINAL_PROMPT_KO
    else:
        prompt_metadata = PROMPT_METADATA_EN
        prompt_ad = PROMPT_AD_EN
        prompt_stt = PROMPT_STT_EN
        final_prompt_template = FINAL_PROMPT_EN

    # 1. Upload Video
    logger.info("[Jack] Uploading video to Gemini...")
    with open(video_path, "rb") as f:
        uploaded = await run_async(
            client.files.upload,
            file=f,
            config={"mime_type": "video/mp4", "display_name": os.path.basename(video_path)},
        )
    
    uploaded = await wait_for_file_active(client, uploaded)
    video_ref = types.Part(
        file_data=types.FileData(file_uri=uploaded.uri, mime_type="video/mp4")
    )
    
    # 2. Parallel Input Generation (Metadata, Core AD, STT)
    logger.info("[Jack] Running sequential steps (Metadata -> Core AD -> STT)...")
    # Note: Running in parallel using gather
    
    async def safe_call(name, contents):
        logger.info(f"[Jack] Starting {name}...")
        try:
            res = await gemini_call(client, contents, GEMINI_MODEL_VISION, temp=0)
            logger.info(f"[Jack] {name} completed.")
            return res
        except Exception as e:
            logger.error(f"[Jack] {name} failed: {e}")
            return None

    results = await asyncio.gather(
        safe_call("Metadata", [types.Part(text=prompt_metadata), video_ref]),
        safe_call("Core AD", [types.Part(text=prompt_ad), video_ref]),
        safe_call("STT", [types.Part(text=prompt_stt), video_ref])
    )
    
    metadata_res, core_ad_res, stt_res = results
    
    if not core_ad_res:
        raise Exception("Core AD generation failed. Cannot proceed.")
    
    # 3. Final Integration
    final_prompt = final_prompt_template.format(
        metadata=metadata_res or "Unavailable",
        core_ad=core_ad_res,
        stt_result=stt_res or "Unavailable"
    )
    
    logger.info("[Jack] Genering Final AD with Context Fusion...")
    final_res = await gemini_call(client, [types.Part(text=final_prompt)], GEMINI_MODEL_TEXT, temp=0)
    
    # Parse Final Result
    try:
        final_json = extract_json_from_response(final_res)
    except Exception:
        logger.warning("[Jack] Failed to parse final JSON, falling back to Core AD")
        final_json = extract_json_from_response(core_ad_res)
        
    ad_segments = final_json.get("audio_descriptions", [])
    if not ad_segments and isinstance(final_json, list):
        ad_segments = final_json
        
    # 4. Stage 2 Compression
    if ad_segments:
        ad_segments = await stage2_compression(client, ad_segments, lang=lang)
    
    # Format for return
    def parse_time(t):
        try:
            if isinstance(t, (int, float)): return float(t)
            parts = str(t).split(':')
            if len(parts) == 2: return float(parts[0])*60 + float(parts[1])
            if len(parts) == 3: return float(parts[0])*3600 + float(parts[1])*60 + float(parts[2])
            return float(t)
        except: return 0.0

    formatted_segments = []
    for idx, seg in enumerate(ad_segments, 1):
        formatted_segments.append({
            "id": idx,
            "start": parse_time(seg.get("start_time") or seg.get("start")),
            "end": parse_time(seg.get("end_time") or seg.get("end")),
            "text": seg.get("description", "")
        })

    # Cleanup
    try:
        await run_async(client.files.delete, name=uploaded.name)
        logger.info("[Jack] Cleanup complete")
    except:
        pass

    return {"audio_descriptions": formatted_segments}, formatted_segments


def generate_ad_for_video(video_path: str, lang: str = "ko") -> Tuple[Dict, List[Dict]]:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop and loop.is_running():
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.run(process_video_async(video_path, lang))
    else:
        return asyncio.run(process_video_async(video_path, lang))

def save_ad_json(video_id: str, data: Any, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_id}.ad.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return output_path

if __name__ == "__main__":
    import argparse
    import sys
    
    # Configure logging for standalone run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Jack AD Generator (Gemini 3 Pro)")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--lang", default="ko", help="Target language (ko/en)")
    parser.add_argument("--output_dir", default=".", help="Output directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        sys.exit(1)
        
    video_id = os.path.splitext(os.path.basename(args.video_path))[0]
    
    print(f"Generating AD for {args.video_path} (Lang: {args.lang})...")
    
    try:
        # Run generation
        full_data, segments = generate_ad_for_video(args.video_path, args.lang)
        
        # Save
        # Make sure output_dir is absolute or correct
        output_dir = os.path.abspath(args.output_dir)
        # Note: save_ad_json expects video_id to be just ID, it appends .ad.json
        # But we want to match server convention: {id}_{lang}.ad.json
        save_name = f"{video_id}_{args.lang}"
        
        out_path = save_ad_json(save_name, full_data, output_dir)
        
        print(json.dumps({
            "success": True, 
            "output_path": out_path, 
            "segments": segments
        }, ensure_ascii=False, indent=2))
        
    except Exception as e:
        logger.error(f"Failed to generate AD: {e}")
        print(json.dumps({
            "success": False,
            "error": str(e)
        }, ensure_ascii=False))
        sys.exit(1)
