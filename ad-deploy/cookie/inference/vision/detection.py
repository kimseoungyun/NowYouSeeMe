import os
import ffmpeg
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_distances
import config

def detect_scenes_siglip():
    print("[Video] Extracting frames for Scene Detection...")
    info = ffmpeg.probe(config.VIDEO_PATH)
    duration = float(info["format"]["duration"])
    t = 0.0
    sampled_frames = []

    # 1. 프레임 샘플링 (0.5초 단위)
    while t < duration:
        out_path = os.path.join(config.SAMPLED_FRAMES_DIR, f"frame_{t:.2f}.jpg")
        try:
            (ffmpeg.input(config.VIDEO_PATH, ss=float(t)).filter("scale", 768, -1)
             .output(out_path, vframes=1).overwrite_output().run(quiet=True))
            if os.path.exists(out_path):
                sampled_frames.append((t, out_path))
            else:
                break
        except ffmpeg.Error:
            break
        t += 0.5
    
    timestamps = [t for (t, _) in sampled_frames]

    # 2. 임베딩 및 코사인 유사도 계산
    print("[Video] Embedding frames...")
    if not sampled_frames:
        return [{"id": 0, "start": 0.0, "end": duration}]

    siglip_proc = AutoProcessor.from_pretrained(config.SIGLIP_MODEL)
    siglip = AutoModel.from_pretrained(config.SIGLIP_MODEL).to(config.DEVICE)

    def embed_image(path):
        img = Image.open(path).convert("RGB")
        inp = siglip_proc(images=img, return_tensors="pt").to(config.DEVICE)
        with torch.no_grad():
            emb = siglip.get_image_features(**inp)
        return (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy()

    embeddings = np.vstack([embed_image(p) for (_, p) in sampled_frames])

    del siglip, siglip_proc
    torch.cuda.empty_cache()

    # 3. 씬 분할 로직
    segments_idx = []
    current = [0]
    for i in range(1, len(embeddings)):
        d = cosine_distances(embeddings[i-1:i], embeddings[i:i+1])[0][0]
        if d < config.SCENE_THRESHOLD: current.append(i)
        else: segments_idx.append(current); current = [i]
    segments_idx.append(current)

    scene_fixed = []
    for idx, seg in enumerate(segments_idx):
        raw_start, raw_end = timestamps[seg[0]], timestamps[seg[-1]]
        start_time = scene_fixed[-1]["end"] if idx > 0 else raw_start
        scene_fixed.append({"id": idx, "start": start_time, "end": raw_end})
    
    return scene_fixed