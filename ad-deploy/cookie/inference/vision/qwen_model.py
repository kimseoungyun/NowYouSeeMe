from unsloth import FastVisionModel
import config
import torch

def load_model():
    print("🤖 Loading Qwen-VL (Base Model)...")
    try:
        model, tokenizer = FastVisionModel.from_pretrained(
            config.MODEL_ID,
            load_in_4bit=True,
            use_gradient_checkpointing=False
        )
        FastVisionModel.for_inference(model)
        print("🤖 Model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        print(f"ERROR: Failed to load Model. {e}")
        return None, None

def run_inference(model, tokenizer, images, prompt, max_tokens=256):
    messages = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": prompt}
            ]
        }
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        images, input_text,
        add_special_tokens=False, return_tensors="pt",
    ).to("cuda")

    torch.cuda.empty_cache()

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        use_cache=True,
        temperature=0.2,
        do_sample=True,
        repetition_penalty=1.1
    )

    full_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
    if "<|im_start|>assistant" in full_decoded:
        response = full_decoded.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
    elif "assistant" in full_decoded:
         response = full_decoded.split("assistant")[-1].replace("<|im_end|>", "").strip()
    else:
        response = full_decoded
    return response