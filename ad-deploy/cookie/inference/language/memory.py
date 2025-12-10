import torch
import pandas as pd
import json
import re
from tqdm import tqdm
import config

# ==================================================================
# 1. [Helper Functions] ID 발급 & JSON 강력 복구 (업그레이드됨)
# ==================================================================

def assign_next_id(registry, gender_tag):
    """
    NEW_FEMALE -> Woman A, Woman B...
    NEW_MALE   -> Man A, Man B...
    """
    # 성별 구분 (FEMALE 먼저 체크)
    if "FEMALE" in gender_tag:
        prefix = "Woman"
    else:
        prefix = "Man"
        
    existing_letters = []
    for key in registry.keys():
        if key.startswith(prefix):
            parts = key.split()
            if len(parts) >= 2:
                existing_letters.append(parts[1])
    
    if not existing_letters:
        return f"{prefix} A"
    
    max_char = max(existing_letters)
    next_char = chr(ord(max_char) + 1)
    
    return f"{prefix} {next_char}"

def extract_and_repair_json(text):
    """
    1. 뒤에 붙은 잡담(### Explanation 등) 제거
    2. 끊긴 JSON 복구
    """
    text = text.replace("```json", "").replace("```", "").strip()
    
    # 1. 첫 번째 '{' 찾기
    match = re.search(r'\{', text)
    if not match: return None
    text = text[match.start():]
    
    # 2. 가장 똑똑한 방법: JSONDecoder.raw_decode 사용
    # (뒤에 글자가 있든 말든 유효한 JSON 구간까지만 읽음)
    try:
        obj, _ = json.JSONDecoder().raw_decode(text)
        return obj
    except ValueError:
        # 실패하면 기존의 '괄호 짝 맞추기' 로직으로 넘어감 (잘린 경우)
        pass

    # 3. 잘린 JSON 수동 복구 로직
    open_braces = text.count('{')
    close_braces = text.count('}')
    
    if open_braces > close_braces:
        text += '}' * (open_braces - close_braces)
    
    text = re.sub(r',\s*\}', '}', text)
    
    try:
        return json.loads(text)
    except:
        return None

# ==================================================================
# 2. [Prompt] System & User Prompt 분리
# ==================================================================

SYSTEM_PROMPT = """You are a **Video Identity Forensic Analyst**.
Your goal is to track characters across scenes using a **Dual-Layer Memory System**.

### [CORE ALGORITHM: DUAL-LAYER MATCHING]

**LAYER 1: SHORT-TERM CONTINUITY (The "Transient" Check)**
- **Target:** Link to the IMMEDIATE previous scene or current scene montage.
- **Criteria:** Compare `Transient Traits` (Outfit, Action, Accessories).
- **Rule:** If Outfit/Location matches exactly with a Recent ID -> **ASSIGN SAME ID**.

**LAYER 2: LONG-TERM RE-IDENTIFICATION (The "Invariant" Check)**
- **Target:** Link to a character who appeared long ago (and might have changed clothes).
- **Criteria:** Compare **ONLY** `Invariant Traits` (Face, Hair Color/Style, Age, Build, Glasses).
- **Rule:** - Ignore outfit changes.
  - If Invariant traits match strongly -> **ASSIGN EXISTING ID**.
  - If Invariant traits CONFLICT (e.g., Red Hair vs Grey Hair) -> **Must be DIFFERENT**.

**LAYER 3: NEW IDENTITY**
- If neither Layer 1 nor Layer 2 provides a match -> Output **`NEW_MALE`** or **`NEW_FEMALE`**.

### [OUTPUT RULES]
1. **Bias to Existing IDs:** Do not create a new ID if a fuzzy match is possible.
2. **Format:** Return ONLY the JSON object. No explanations.
"""

def get_user_prompt(current_row, global_registry, prev_active_ids_str, prev_visual_context):
    raw_chars = current_row.get('characters', '')
    curr_visual = current_row.get('visual_analysis', '')
    
    # 레지스트리 요약 (너무 길면 자르는 로직 추가 가능)
    registry_str = json.dumps(global_registry, indent=2, ensure_ascii=False) if global_registry else "(None)"
    
    return f"""
### [INPUT DATA]
1. **Previous Context:** "{prev_visual_context}"
2. **Current Visuals:** "{curr_visual}"
3. **Detected Characters:** "{raw_chars}"
4. **Global Registry (Memory Bank):**
{registry_str}
5. **Recently Active IDs (Short-term context):** [{prev_active_ids_str}]

### [INSTRUCTIONS]
- **Step 1:** Check `Recently Active IDs`. Does anyone match the **Transient** traits (Outfit)?
- **Step 2:** Check `Global Registry`. Does anyone match the **Invariant** traits (Face/Body)?
- **Step 3:** If no match, output `NEW_MALE` or `NEW_FEMALE`.

### [OUTPUT FORMAT]
{{
  "id_mapping": {{ "Person 1": "Man A", "Person 2": "NEW_FEMALE" }},
  "registry_update": {{
      "Man A": {{
          "invariant": "Male, Grey hair, Glasses (Long-term)",
          "transient": {{ "outfit": "Blue Sportswear", "last_seen": "Image 12" }}
      }},
      "NEW_FEMALE": {{
          "invariant": "Female, Blonde hair (Long-term)",
          "transient": {{ "outfit": "Green Dress", "last_seen": "Image 12" }}
      }}
  }}
}}
"""

# ==================================================================
# 3. [Main Logic] 실행 루프 (변수명 버그 수정됨 ✅)
# ==================================================================

def run_memory_tracker(model, tokenizer, input_json_path):
    print("\n" + "="*50)
    print(">> Running Stage 2: Dual-Layer Identity Tracker (Fixed)...")
    print("="*50)

    try: df = pd.read_json(input_json_path, convert_dates=False)
    except Exception as e: print(f"❌ Error: {e}"); return

    # [Llama-3 Config] 종료 토큰 설정
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    final_mappings = []
    global_registry = {}
    prev_active_ids = "(None)"
    prev_context = "Start"

    # 로그 초기화
    with open(config.STAGE2_LOG_FILE, "w", encoding="utf-8") as f: f.write("")

    for i in tqdm(range(len(df))):
        row = df.iloc[i]
        
        # 1. 메시지 구성
        user_content = get_user_prompt(row, global_registry, prev_active_ids, prev_context)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]
        
        input_tensor = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        try:
            # 2. 모델 추론
            outputs = model.generate(
                input_ids=input_tensor,
                max_new_tokens=600,
                do_sample=False,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id
            )
            
            raw = tokenizer.decode(outputs[0][input_tensor.shape[-1]:], skip_special_tokens=True).strip()

            # 3. JSON 파싱
            data = extract_and_repair_json(raw)
            if not data:
                tqdm.write(f"\n❌ JSON Parsing Failed at Scene {i}")
                tqdm.write(f"👀 RAW OUTPUT: {raw[:100]}...") 
                raise ValueError("JSON Fail")

            mapping = data.get("id_mapping", {})
            updates = data.get("registry_update", {})

            # ====================================================
            # 🚀 [Hybrid Logic] Python이 ID 발급 관리 (변수명 수정됨!)
            # ====================================================
            
            # (1) ID 매핑 변환 (NEW_MALE -> Man C)
            temp_id_map = {}  # ✅ 여기 이름과
            for key, val in mapping.items():
                if val in ["NEW_MALE", "NEW_FEMALE"]:
                    if val not in temp_id_map:
                        real_id = assign_next_id(global_registry, val)
                        global_registry[real_id] = {} 
                        temp_id_map[val] = real_id
                    mapping[key] = temp_id_map[val]

            # (2) Registry Update 키 변환
            final_registry_update = {}
            for k, v in updates.items():
                # ✅ 여기가 일치해야 합니다 (temp_id_map)
                real_id = temp_id_map.get(k) 
                
                # Case A: 매핑에서 변환된 NEW ID인 경우
                if k in ["NEW_MALE", "NEW_FEMALE"]:
                    if not real_id:
                        real_id = assign_next_id(global_registry, k)
                        global_registry[real_id] = {}
                    final_registry_update[real_id] = v
                
                # Case B: 기존 ID인 경우 (Man A -> Man A)
                else:
                    final_registry_update[k] = v
            
            # (3) 업데이트 적용
            global_registry.update(final_registry_update)
            # ====================================================

            # Active ID 갱신
            active = sorted(list(set([v for v in mapping.values() if v and "Ignore" not in v])))
            final_mappings.append(mapping)
            
            # 로그 저장
            log_entry = {"index": i, "mapping": mapping, "registry_snapshot": final_registry_update}
            with open(config.STAGE2_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            
            tqdm.write(f"[Scene {i}] {mapping}")
            
            prev_context = row.get('visual_analysis', '')
            prev_active_ids = ", ".join(active) if active else "None"

        except Exception as e:
            tqdm.write(f"⚠️ Error at {i}: {e}")
            final_mappings.append({})
            prev_context = row.get('visual_analysis', '')

    df['stage2_mapping'] = final_mappings
    df.to_json(config.STAGE2_OUTPUT_JSON, orient='records', indent=4, force_ascii=False)
    print("Done.")