import pandas as pd
import json
from tqdm import tqdm
import config

# ==================================================================
# 3. [Refinement Stage] Narrative Rewrite with ID Injection
# (원본 프롬프트 보존)
# ==================================================================

def prompt_refine_narrative(row):
    # 매핑 정보 안전하게 가져오기
    mapping = row.get('stage2_mapping', {})
    if not mapping:
        mapping_str = "No specific mapping. Keep original names."
    else:
        # 매핑 정보를 텍스트로 변환
        mapping_str = "\n".join([f"- {k} -> {v}" for k, v in mapping.items()])

    # 기존 문장
    original_text = row.get('stage1_result', '')

    prompt = f"""
### Role
You are a **Professional Screenplay Editor**.
Your task is to polish the raw narrative log into a concise, action-driven script line using the provided **Identity Mapping**.

### INPUT DATA
**1. Identity Mapping (Target IDs):**
{mapping_str}

**2. Raw Narrative Log:**
"{original_text}"

### EDITING RULES (Universal Logic)

**Step 1: IDENTITY INJECTION (Who)**
- Replace "Person X" with the mapped ID (e.g., "Man A").
- If a person maps to "Ignore", remove their mention unless they interact with the main character.

**Step 2: VISUAL DECLUTTERING (Remove Decoration)**
- **DELETE** descriptions of static appearance: Clothing colors, hair styles, height, "wearing a...", "dressed in...".
- **KEEP** critical objects/props: "holding a gun", "carrying a bag", "eating an apple".
- **Logic:** The ID ("Man A") already implies the look. We only need the action.
- *Example:* "Man A, wearing a red coat, holds a knife." -> "Man A holds a knife."

**Step 3: ACTION FLOW (What)**
- Ensure the sentence flows naturally in the **Present Tense**.
- Keep the timeline consistent with the original text.

### OUTPUT
Rewritten Narrative:
"""
    return prompt
# ==================================================================
# 메인 실행 함수
# ==================================================================

def run_narrative_refinement(model, tokenizer, input_json_path):
    print("\n" + "="*50)
    print(">> Running Stage 3: Narrative Refinement (Injecting IDs)...")
    print("="*50)

    try:
        df = pd.read_json(input_json_path, convert_dates=False)
        print(f"✅ Loaded {len(df)} rows from {input_json_path}")
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return

    final_narratives = []

    for i in tqdm(range(len(df))):
        row = df.iloc[i]

        # 매핑 정보나 기존 문장이 없으면 그대로 유지
        if not row.get('stage1_result') or not row.get('stage2_mapping'):
            final_narratives.append(row.get('stage1_result', ''))
            continue

        prompt = prompt_refine_narrative(row)

        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

        try:
            outputs = model.generate(
                inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.2, # 이름을 정확히 바꾸는 것이 목적이므로 온도를 낮게 설정
                eos_token_id=tokenizer.eos_token_id
            )

            result = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True).strip()

            # 불필요한 따옴표 제거
            result = result.replace('"', '').strip()
            final_narratives.append(result)

        except Exception as e:
            print(f"⚠️ Error at index {i}: {e}")
            final_narratives.append(row.get('stage1_result', ''))

    # 결과 컬럼 추가
    df['final_narrative'] = final_narratives

    # ==================================================================
    # 4. 검증 및 저장
    # ==================================================================
    print("\n===== 🧪 REFINEMENT CHECK =====")

    # 샘플 확인 (처음 3개)
    for idx in range(min(3, len(df))):
        print(f"\n[Scene {idx}]")
        print(f"Mapping : {df.iloc[idx].get('stage2_mapping')}")
        print(f"Original: {df.iloc[idx].get('stage1_result')}")
        print(f"Updated : {df.iloc[idx]['final_narrative']}")

    # 최종 저장
    df.to_json(config.STAGE3_OUTPUT_JSON, orient="records", force_ascii=False, indent=4)
    print(f"\n✅ Final refined JSON saved to: {config.STAGE3_OUTPUT_JSON}")