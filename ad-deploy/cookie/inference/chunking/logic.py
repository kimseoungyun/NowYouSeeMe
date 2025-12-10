def create_final_chunks(scene_fixed, dialogues):
    # 1. Expand Scenes
    scene_revised = []
    for sc in scene_fixed:
        expanded_end = _expand_scene_to_dialogues(sc["start"], sc["end"], dialogues)
        scene_revised.append({"id": sc["id"], "start": sc["start"], "end": expanded_end})
    
    for i in range(1, len(scene_revised)):
        scene_revised[i]["start"] = scene_revised[i-1]["end"]

    # 2. Split & Chunk Logic
    final_chunks = []
    for sc in scene_revised:
        subs = _split_scene_by_dialogues(sc, dialogues)
        chunks = _chunk_subscenes(subs)
        chunks = _merge_last_chunk(chunks)
        final_chunks.extend(chunks)

    # 3. Refine
    final_chunks = _merge_too_short_chunks(final_chunks)
    refined_chunks = _refine_long_chunks(final_chunks, dialogues)
    final_chunks = _merge_too_short_chunks(refined_chunks, min_len=2.0, max_len=8.0)
    
    return final_chunks

# === 내부 Helper 함수들 (Private) ===
# (원본 코드의 함수들을 그대로 가져옵니다. 공간 절약을 위해 핵심만 표시합니다)
# 사용자가 준 코드의 3번 섹션 함수들을 모두 여기에 넣으세요.

def _expand_scene_to_dialogues(scene_start, scene_end, dialogues):
    new_end = scene_end
    for dlg in dialogues:
        d_start, d_end = dlg["start"], dlg["end"]
        overlap = not (d_end <= scene_start or d_start >= scene_end)
        if overlap: new_end = max(new_end, d_end)
    return new_end

def _split_scene_by_dialogues(scene, dialogues):
    # (원본의 split_scene_by_dialogues 코드 복사)
    s_start, s_end = scene["start"], scene["end"]
    relevant = [dlg for dlg in dialogues if not (dlg["end"] <= s_start or dlg["start"] >= s_end)]
    relevant = sorted(relevant, key=lambda x: x["start"]); subs = []; cur_start = s_start
    for dlg in relevant:
        d_start, d_end = dlg["start"], dlg["end"]
        if d_start > cur_start: subs.append({"scene_id": scene["id"], "start": cur_start, "end": d_start, "type": "no_dialogue"})
        subs.append({"scene_id": scene["id"], "start": d_start, "end": d_end, "type": "dialogue", "text": dlg["text"]})
        cur_start = d_end
    if cur_start < s_end: subs.append({"scene_id": scene["id"], "start": cur_start, "end": s_end, "type": "no_dialogue"})
    return subs

def _split_no_dialogue_evenly(start, end, min_len=4, max_len=6):
    # (원본 코드 복사)
    duration = end - start; best = None
    for parts in range(1, 7):
        seg = duration / parts
        if min_len <= seg <= max_len: best = parts; break
    if best is None: return [(start, end)]
    return [(start + i*duration/best, start + (i+1)*duration/best) for i in range(best)]

def _chunk_subscenes(subscenes, min_len=4, max_len=6):
    # (원본 코드 복사)
    chunks = []; current_start = None; current_end = None; accum = 0
    for ss in subscenes:
        ss_start, ss_end = ss["start"], ss["end"]; ss_len = ss_end - ss_start
        if ss_len >= max_len and ss["type"] == "dialogue":
            if current_start is not None: chunks.append({"scene_id": ss["scene_id"], "start": current_start, "end": current_end})
            chunks.append({"scene_id": ss["scene_id"], "start": ss_start, "end": ss_end})
            current_start = None; accum = 0; continue
        if ss_len >= max_len and ss["type"] == "no_dialogue":
            if current_start is not None: chunks.append({"scene_id": ss["scene_id"], "start": current_start, "end": current_end})
            for st, ed in _split_no_dialogue_evenly(ss_start, ss_end): chunks.append({"scene_id": ss["scene_id"], "start": st, "end": ed})
            current_start = None; accum = 0; continue
        if current_start is None:
            current_start = ss_start; current_end = ss_end; accum = ss_len; continue
        if accum + ss_len > max_len:
            chunks.append({"scene_id": ss["scene_id"], "start": current_start, "end": current_end})
            current_start = ss_start; current_end = ss_end; accum = ss_len; continue
        current_end = ss_end; accum += ss_len
    if current_start is not None: chunks.append({"scene_id": ss["scene_id"], "start": current_start, "end": current_end})
    return chunks

def _merge_last_chunk(chunks, min_len=4, max_len=6):
    # (원본 코드 복사)
    if len(chunks) <= 1: return chunks
    last = chunks[-1]; prev = chunks[-2]; last_len = last["end"] - last["start"]
    if last_len >= min_len: return chunks
    merged_len = last["end"] - prev["start"]
    if merged_len <= max_len:
        return chunks[:-2] + [{"scene_id": last["scene_id"], "start": prev["start"], "end": last["end"]}]
    return chunks

def _merge_too_short_chunks(chunks, min_len=2.0, max_len=6.0):
    # (원본 코드 복사)
    if len(chunks) <= 1: return chunks
    merged = []; i = 0
    while i < len(chunks):
        ch = chunks[i]; dur = ch["end"] - ch["start"]
        if dur >= min_len: merged.append(ch); i += 1; continue
        if merged:
            prev = merged[-1]
            if (prev["end"] - prev["start"]) + dur <= max_len: prev["end"] = ch["end"]; i += 1; continue
        if i + 1 < len(chunks):
            nxt = chunks[i+1]; nxt_dur = nxt["end"] - nxt["start"]
            if dur + nxt_dur <= max_len:
                merged.append({"scene_id": ch["scene_id"], "start": ch["start"], "end": nxt["end"]})
                i += 2; continue
        if merged: merged[-1]["end"] = ch["end"]
        elif i + 1 < len(chunks):
            nxt = chunks[i+1]
            merged.append({"scene_id": ch["scene_id"], "start": ch["start"], "end": nxt["end"]})
            i += 2; continue
        i += 1
    return merged

def _refine_long_chunks(chunks, dialogues, max_len=6.0):
    # (원본 코드 복사 - 길이가 길어 생략했으나 원본 logic 그대로 복붙하세요)
    # ... 원본 코드의 refine_long_chunks 내용 ...
    refined = []
    for ch in chunks:
        duration = ch['end'] - ch['start']
        if duration <= 10.0: refined.append(ch); continue

        inner_dialogues = [d for d in dialogues if d['end'] > ch['start'] and d['start'] < ch['end']]
        if not inner_dialogues:
            split_count = int(duration // max_len) + 1; step = duration / split_count
            for k in range(split_count):
                st = ch['start'] + k * step; ed = ch['start'] + (k+1) * step
                refined.append({"scene_id": ch['scene_id'], "start": st, "end": ed})
            continue

        curr = ch['start']
        for dlg in inner_dialogues:
            gap = dlg['start'] - curr
            if gap > max_len:
                split_cnt = int(gap // max_len) + 1; step = gap / split_cnt
                for k in range(split_cnt):
                    st = curr + k * step; ed = curr + (k+1) * step
                    refined.append({"scene_id": ch['scene_id'], "start": st, "end": ed})
            elif gap > 0.1: refined.append({"scene_id": ch['scene_id'], "start": curr, "end": dlg['start']})

            refined.append({"scene_id": ch['scene_id'], "start": dlg['start'], "end": dlg['end'], "type": "dialogue"})
            curr = dlg['end']

        remaining = ch['end'] - curr
        if remaining > max_len:
            split_cnt = int(remaining // max_len) + 1; step = remaining / split_cnt
            for k in range(split_cnt):
                st = curr + k * step; ed = curr + (k+1) * step
                refined.append({"scene_id": ch['scene_id'], "start": st, "end": ed})
        elif remaining > 0.1:
            refined.append({"scene_id": ch['scene_id'], "start": curr, "end": ch['end']})

    return refined