import pandas as pd
from tqdm import tqdm
import torch
import config

def prompt_visual_narrative(row):
    # Qwen이 준 Characters와 Visual Analysis 정보를 문자열로 변환
    chars = row.get('characters', '')
    visuals = row.get('visual_analysis', '')
    
    prompt = f"""
You are a **Video Logic Editor**.
Your task is to convert the raw visual analysis into a structured narrative log.

### INPUT DATA
**[Identified Characters]**
{chars}

**[Visual Action Sequence]**
{visuals}

### INSTRUCTIONS (CRITICAL for Tracking)
1. **PRESERVE IDs:** - **DO NOT** replace "Person 1", "Person 2" with generic descriptions like "The man".
   - **MUST USE** the exact IDs: "Person 1", "Person 2", etc.

2. **Respect Time/Scene Cuts:**
   - Pay attention to the `[Image X-Y]` ranges.
   - If the location or time changes significantly between image ranges, use transition words like **"Then,"** or **"Next,"**.

3. **Narrative Style & Grounding (Strict):**
   - **Structure:** [Person ID] [Action] [Location/Context].
   - **NO HALLUCINATION:** Describe **ONLY** the actions explicitly stated in the **[Visual Action Sequence]**. Do not add extra movements (e.g., do not say "looking up" unless the text says so).
   - Write in the **Present Tense**.

### OUTPUT
Narrative Log:
"""
    return prompt

def generate_narratives(model, tokenizer, input_json_path):
    print("\n>> Running Stage 1: Visual Narrative Synthesis...")
    
    try:
        df = pd.read_json(input_json_path, convert_dates=False)
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return None

    stage1_results = []

    for i in tqdm(range(len(df))):
        row = df.iloc[i]

        if "Error" in str(row['visual_analysis']):
            stage1_results.append("")
            continue

        prompt = prompt_visual_narrative(row)

        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

        outputs = model.generate(
            inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.3,
            eos_token_id=tokenizer.eos_token_id
        )

        result = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True).strip()

        # 불필요한 헤더 제거
        result = result.replace('Descriptive Sentence:', '').replace('"', '').strip()
        stage1_results.append(result)

    df['stage1_result'] = stage1_results
    
    # 저장
    df.to_json(config.STAGE1_OUTPUT_JSON, orient="records", force_ascii=False, indent=2)
    print(f"\n✅ Saved to {config.STAGE1_OUTPUT_JSON}")
    
    return df