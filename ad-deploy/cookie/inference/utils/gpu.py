import torch
import gc

def clear_gpu_memory(model=None, tokenizer=None, other_models=None):
    # 1. 기존 모델 변수 제거
    # 함수 인자로 받은 객체들을 삭제합니다.
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    if other_models:
        for m in other_models:
            if m is not None: del m

    # 2. 가비지 컬렉션 및 캐시 비우기
    gc.collect()
    torch.cuda.empty_cache()

    # 3. 확인
    print(f"✅ GPU Memory Cleared. Current device: {torch.cuda.current_device()}")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")