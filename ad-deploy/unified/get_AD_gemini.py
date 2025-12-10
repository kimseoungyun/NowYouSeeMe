import logging
import os
import json
import re
import argparse
from typing import List, Dict, Any, Tuple

try:
    from google import genai
    from google.genai import types
except ImportError as e:
    # Fallback for different import styles or missing package
    try:
        import google.generativeai as genai
        import google.generativeai.types as types
    except ImportError:
         raise ImportError(
            "google-genai package is not installed. "
            "Please install it with: pip install google-genai"
        ) from e

logger = logging.getLogger(__name__)

# Use gemini-2.5-flash with thinking mode disabled (thinking_budget=0)
# This provides better performance than 2.0-flash while avoiding MAX_TOKENS issues
# GEMINI_MODEL = "gemini-3-pro-preview"
GEMINI_MODEL = "gemini-2.5-pro"
PROMPT_KO = """**MISSION:** 입력된 비디오를 정밀하게 분석하여 '대사가 없는 구간(침묵 구간)'에 삽입할 화면 해설(Audio Description) 스크립트를 작성하십시오. 모든 결과는 지정된 JSON 형식으로 반환해야 합니다.

**REQUIREMENTS (MANDATORY):**

1.  **전체 대본 (Full Transcript):** 비디오 내의 모든 대사(화자 포함)와 주요 소리 이벤트를 타임스탬프와 함께 정확하게 기록하십시오.

2.  **침묵 구간 식별 (Identify Silent Gaps):** 대사가 전혀 없는 '침묵 구간'의 시작과 끝 타임스탬프(초 단위)를 식별하십시오. 길이가 2.5초 이상인 구간만 선택하십시오.

3.  **화면 해설 생성 (Generate Audio Description):** 식별된 각 침묵 구간에 대해, 그 시간 동안 발생하는 가장 중요하고 결정적인 시각적 행동을 묘사하는 **화면 해설 텍스트를 한국어로** 작성하십시오.

4.  **인물 지칭 규칙 (Character Reference Rules - CRITICAL):**
    - 등장인물의 이름을 절대 사용하지 마십시오. 영상에서 이름이 언급되더라도 화면해설에서는 사용하지 않습니다.
    - 인물의 외모, 옷차림, 나이 등을 묘사하지 마십시오.
    - 대신 성별과 등장 순서에 따라 번호를 부여하여 지칭하십시오: "남자1", "남자2", "여자1", "여자2" 등
    - 동일 인물은 영상 전체에서 일관된 번호를 유지해야 합니다.
    - 예시: "철수가 문을 연다" (X) → "남자1이 문을 연다" (O)
    - 예시: "빨간 드레스를 입은 여자가 걷는다" (X) → "여자1이 걷는다" (O)

5.  **지속 시간 계산 (Duration Calculation):** `duration_sec` 필드는 반드시 `end_time`과 `start_time`의 정확한 차이(end_time - start_time)로 계산되어야 합니다. 3.0초 등으로 임의로 제한하지 마십시오. 실제 침묵 구간의 길이를 반영해야 합니다(예: 2.6, 3.5, 5.2 등). 해설 텍스트는 이 시간 안에 읽을 수 있을 만큼 간결해야 합니다.

6.  **출력 형식 (Output Format):** 반드시 아래의 JSON 구조를 엄격히 따르십시오. 오직 JSON 코드 블록만 출력해야 합니다.

```json
{
  "full_transcript": [
    {"time": "0:01.2", "speaker": "남자1", "text": "대사 내용..."},
    {"time": "0:06.6", "speaker": "[Sound]", "text": "차 문 닫히는 소리"}
  ],
  "audio_descriptions": [
    {
      "start_time": "0:03.5",
      "end_time": "0:06.1",
      "duration_sec": 2.6,
      "description": "남자1이 커피를 한 모금 마신다." 
    },
    {
      "start_time": "0:07.8",
      "end_time": "0:12.3",
      "duration_sec": 4.5,
      "description": "카메라가 건물 외벽을 빠르게 훑고 올라간다."
    }
  ]
}
```

**CRITICAL: 유효한 JSON 객체만 반환하십시오. 설명이나 마크다운 코드 펜스 외의 다른 텍스트는 포함하지 마십시오.**
"""

PROMPT_EN = """**MISSION:** Precisely analyze the input video and generate an Audio Description script for insertion into the 'dialogue-free gaps' (silent sections). Return all findings in the specified JSON format.

**REQUIREMENTS (MANDATORY):**

1.  **Full Transcript (STT):** Accurately record ALL dialogue (spoken words by any speaker) and major sound events within the video, including timestamps. Include every spoken word, not just sound effects. For dialogue, identify the speaker if possible, or use "Speaker" if unknown.

2.  **Identify Silent Gaps (VAD Replacement):** Identify and return the start and end timestamps (in seconds) for all 'silent sections' where there is absolutely no dialogue. Select only sections longer than 2.5sec.

3.  **Generate Audio Description:** For each identified silent section, generate an **Audio Description** text that describes only the most important and critical visual action occurring during that time.

4.  **Character Reference Rules (CRITICAL):**
    - NEVER use character names. Even if names are mentioned in the video, do not use them in the audio description.
    - Do NOT describe physical appearance, clothing, or age of characters.
    - Instead, refer to characters by gender and sequential number: "Man1", "Man2", "Woman1", "Woman2", etc.
    - Maintain consistent numbering for the same character throughout the entire video.
    - Example: "John opens the door" (X) → "Man1 opens the door" (O)
    - Example: "The woman in the red dress walks" (X) → "Woman1 walks" (O)

5.  **Duration Calculation (CRITICAL):** The `duration_sec` field MUST be calculated as the exact difference between `end_time` and `start_time` (end_time - start_time). Do NOT limit it to 3.0 seconds or any fixed value. The duration should reflect the actual length of the silent gap, which can be any value (e.g., 2.6, 3.5, 5.2, 8.1 seconds, etc.). The AD text description should be concise enough to be read within this actual duration, but the duration_sec value itself must match the actual gap length.

6.  **Output Format:** Strictly adhere to the following JSON structure. You must only output the structured JSON code block.

```json
{
  "full_transcript": [
    {"time": "0:01.2", "speaker": "Man1", "text": "You're not just a mover, you're a mover and a shaker."},
    {"time": "0:04.2", "speaker": "Woman1", "text": "What are you talking about?"},
    {"time": "0:06.6", "speaker": "[Sound]", "text": "Car door slams"}
  ],
  "audio_descriptions": [
    {
      "start_time": "0:03.5",
      "end_time": "0:06.1",
      "duration_sec": 2.6,
      "description": "Man1 takes a sip of coffee." 
    },
    {
      "start_time": "0:07.8",
      "end_time": "0:12.3",
      "duration_sec": 4.5,
      "description": "The camera quickly ascends the building exterior."
    }
  ]
}
```

**CRITICAL: Return ONLY a valid JSON object, nothing else. Do not include any natural language explanation, markdown code fences, or any other text.**
"""


def repair_truncated_json(json_str: str) -> str:
    """
    Attempt to repair a truncated JSON string by closing unclosed brackets.
    """
    # Count open brackets
    open_braces = 0
    open_brackets = 0
    in_string = False
    escape_next = False
    
    for char in json_str:
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            open_braces += 1
        elif char == '}':
            open_braces -= 1
        elif char == '[':
            open_brackets += 1
        elif char == ']':
            open_brackets -= 1
    
    # If we're in a string, close it
    if in_string:
        json_str += '"'
    
    # Remove any trailing incomplete content (partial key/value)
    # Find last complete structure
    json_str = json_str.rstrip()
    
    # Remove trailing comma if present
    if json_str.endswith(','):
        json_str = json_str[:-1]
    
    # Close missing brackets
    json_str += ']' * open_brackets
    json_str += '}' * open_braces
    
    return json_str


def extract_segments_from_gemini(raw_text: str) -> Tuple[Dict, List[Dict]]:
    """
    Extract and parse JSON segments from Gemini API response.
    
    Args:
        raw_text: Raw text response from Gemini API
        
    Returns:
        Tuple[Dict, List[Dict]]: (full_json_data, formatted_segments)
    """
    # Check for None or empty response
    if raw_text is None:
        logger.error("[AD] Gemini API returned None response")
        raise ValueError("Gemini API returned empty (None) response. This may indicate an API error, content filtering, or rate limiting.")
    
    if not raw_text or not raw_text.strip():
        logger.error("[AD] Gemini API returned empty string")
        raise ValueError("Gemini API returned empty response. Please try again.")
    
    logger.info("[AD] Raw Gemini response length: %d characters", len(raw_text))
    
    # Log first 2000 characters for debugging
    preview = raw_text[:2000]
    if len(raw_text) > 2000:
        preview += "\\n... (truncated)"
    logger.info("[AD] Raw Gemini response (first 2000 chars):\\n%s", preview)
    
    # Step 1: Remove markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(.+?)\s*```", raw_text, re.DOTALL)
    if fence_match:
        logger.debug("[AD] Found markdown code fence, extracting content")
        candidate = fence_match.group(1).strip()
    else:
        candidate = raw_text.strip()
    
    # Step 2: Try to find JSON object { ... } or array [ ... ]
    # Use non-greedy approach to handle truncated JSON better
    object_start = candidate.find('{')
    if object_start != -1:
        json_str = candidate[object_start:]
    else:
        array_start = candidate.find('[')
        if array_start != -1:
            json_str = candidate[array_start:]
        else:
            json_str = candidate
    
    # Step 3: Clean up common issues
    json_str = json_str.replace("\r\n", "\n").replace("\r", "\n").strip()
    
    # Step 4: Fix unescaped newlines inside strings
    fixed_chars = []
    in_string = False
    escape_next = False
    i = 0
    
    while i < len(json_str):
        char = json_str[i]
        if escape_next:
            if char == 'n': fixed_chars.append('\\n')
            elif char == '\\': fixed_chars.append('\\\\')
            elif char == '"': fixed_chars.append('\\"')
            else: fixed_chars.append('\\' + char)
            escape_next = False
        elif char == '\\':
            escape_next = True
            fixed_chars.append(char)
        elif char == '"' and not escape_next:
            in_string = not in_string
            fixed_chars.append(char)
        elif char == '\n' and in_string and not escape_next:
            fixed_chars.append('\\n')
        else:
            fixed_chars.append(char)
        i += 1
    
    json_str = ''.join(fixed_chars)
    
    # Step 5: Remove trailing commas
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    
    # Step 6: Parse JSON (with repair attempt for truncated JSON)
    segments = None
    try:
        segments = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning("[AD] Initial JSON parse failed: %s", e)
        logger.info("[AD] Attempting to repair truncated JSON...")
        
        # Try to repair truncated JSON
        repaired_json = repair_truncated_json(json_str)
        
        try:
            segments = json.loads(repaired_json)
            logger.info("[AD] JSON repair successful!")
        except json.JSONDecodeError as e2:
            logger.error("[AD] JSON repair failed: %s", e2)
            logger.error("[AD] Original JSON (first 500 chars): %s", json_str[:500])
            logger.error("[AD] Repaired JSON (last 200 chars): %s", repaired_json[-200:])
            logger.warning("[AD] Returning empty segments due to JSON parse error.")
            return {"audio_descriptions": []}, []
    
    # Step 7: Validate structure and extract audio_descriptions
    raw_segments = []
    if isinstance(segments, dict):
        if "audio_descriptions" in segments:
            audio_descriptions = segments["audio_descriptions"]
            if not isinstance(audio_descriptions, list):
                logger.warning(f"audio_descriptions is not a list, got {type(audio_descriptions)}. Returning empty.")
                return {"audio_descriptions": []}, []
            raw_segments = audio_descriptions
        else:
            logger.warning("JSON structure is invalid: dict must contain 'audio_descriptions' array. Returning empty.")
            return {"audio_descriptions": []}, []
    elif isinstance(segments, list):
        raw_segments = segments
        # Wrap in dict for consistency
        segments = {"audio_descriptions": segments}
    else:
        logger.warning(f"Parsed JSON is neither dict nor list: {type(segments)}. Returning empty.")
        return {"audio_descriptions": []}, []

    # Convert to standardized format
    final_segments = []
    
    def parse_time_string(time_str) -> float:
        try:
            if isinstance(time_str, (int, float)):
                return float(time_str)
            if ':' not in str(time_str):
                return float(time_str)
            parts = str(time_str).split(':')
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            return float(time_str)
        except (ValueError, AttributeError):
            return 0.0
    
    for idx, seg in enumerate(raw_segments, start=1):
        start_time = seg.get("start_time") or seg.get("start")
        end_time = seg.get("end_time") or seg.get("end")
        
        final_segments.append({
            "id": idx,
            "start": parse_time_string(start_time),
            "end": parse_time_string(end_time),
            "text": seg.get("description", seg.get("text", ""))
        })

    logger.info("[AD] Finished AD generation. segments=%d", len(final_segments))
    return segments, final_segments


def generate_ad_for_video(video_path: str, api_key: str = None, lang: str = "ko") -> Tuple[Dict, List[Dict]]:
    """
    Run Gemini-based AD generation for the given video_path.
    Uses gemini-2.5-flash with thinking mode disabled (thinking_budget=0)
    to get better performance while avoiding MAX_TOKENS issues.
    
    Returns:
        Tuple[Dict, List[Dict]]: (full_json_data, formatted_segments)
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables or arguments")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        logger.error("Failed to initialize Gemini client: %s", e)
        raise

    logger.info("[AD] Reading video file: %s", video_path)
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    PROMPT = PROMPT_KO if lang == 'ko' else PROMPT_EN
    logger.info(f"[AD] Using prompt for language: {lang}")

    contents = types.Content(
        parts=[
            types.Part(text=PROMPT),
            types.Part(inline_data=types.Blob(data=video_bytes, mime_type="video/mp4"))
        ]
    )

    logger.info("[AD] Sending request to Gemini (%s) with thinking disabled...", GEMINI_MODEL)
    
    # Build config with thinking mode disabled for Gemini 2.5
    # thinking_budget=0 disables the thinking mode to avoid MAX_TOKENS issues
    config_kwargs = {
        "temperature": 0,
        "max_output_tokens": 8192,
    }
    
    # Try to set thinking_budget=0 to disable thinking mode
    try:
        # Method 1: Using ThinkingConfig if available
        if hasattr(types, 'ThinkingConfig'):
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            logger.info("[AD] Using ThinkingConfig with thinking_budget=0")
        else:
            # Method 2: Direct parameter (some versions support this)
            config_kwargs["thinking_budget"] = 0
            logger.info("[AD] Using direct thinking_budget=0 parameter")
    except Exception as e:
        logger.warning("[AD] Could not set thinking config: %s", e)
    
    config = types.GenerateContentConfig(**config_kwargs)
    
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=config
        )
    except Exception as e:
        logger.error("Gemini API request failed: %s", e)
        # If thinking config caused issues, retry without it
        if "thinking" in str(e).lower():
            logger.info("[AD] Retrying without thinking config...")
            config = types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=8192,
            )
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=config
            )
        else:
            raise

    logger.info("[AD] Received response from Gemini")
    
    # Check if response is valid
    if response is None:
        raise ValueError("Gemini API returned None response object")
    
    # Check finish reason for any issues
    def check_finish_reason():
        try:
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = str(candidate.finish_reason)
                        logger.info(f"[AD] Finish reason: {finish_reason}")
                        if 'MAX_TOKENS' in finish_reason:
                            logger.warning("[AD] ⚠️ MAX_TOKENS reached - response may be truncated")
                        return finish_reason
        except Exception:
            pass
        return None
    
    check_finish_reason()
    
    # Check for blocked content or empty response
    response_text = None
    try:
        response_text = response.text
    except Exception as e:
        logger.error("[AD] Failed to get response.text: %s", e)
        # Try to get more details about why response failed
        if hasattr(response, 'prompt_feedback'):
            logger.error("[AD] Prompt feedback: %s", response.prompt_feedback)
        if hasattr(response, 'candidates') and response.candidates:
            for i, candidate in enumerate(response.candidates):
                if hasattr(candidate, 'finish_reason'):
                    logger.error("[AD] Candidate %d finish_reason: %s", i, candidate.finish_reason)
                if hasattr(candidate, 'safety_ratings'):
                    logger.error("[AD] Candidate %d safety_ratings: %s", i, candidate.safety_ratings)
        raise ValueError(f"Failed to extract text from Gemini response: {e}")
    
    return extract_segments_from_gemini(response_text)


def save_ad_json(video_id: str, data: Any, output_dir: str) -> str:
    """
    Save AD data to a JSON file.
    """
    logger.debug("[AD] Saving JSON for video_id: %s to directory: %s", video_id, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_id}.ad.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("[AD] JSON saved to: %s", output_path)
    return output_path


if __name__ == "__main__":
    import sys
    
    # Configure logging for CLI mode
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Generate Audio Description for a video using Gemini.")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--api_key", help="Google Gemini API Key")
    parser.add_argument("--lang", default="ko", choices=["ko", "en"], help="Language for AD generation (ko/en)")
    
    args = parser.parse_args()
    
    try:
        full_data, segments = generate_ad_for_video(args.video_path, api_key=args.api_key, lang=args.lang)
        
        video_id = os.path.splitext(os.path.basename(args.video_path))[0]
        
        # Save the FULL data to the file
        json_path = save_ad_json(video_id, full_data, "./ad_json")
        print(f"\\n=== Saved to {json_path} ===")
        print(f"AD segments: {len(segments)}")
        
    except Exception as e:
        logger.exception("[CLI] Error during AD generation")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

