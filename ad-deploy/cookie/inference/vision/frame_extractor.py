import numpy as np
import ffmpeg
import os
from PIL import Image
from scenedetect import VideoManager, SceneManager, FrameTimecode
from scenedetect.detectors import ContentDetector

def get_shot_based_frames(video_path, chunk_start, chunk_end, max_total_frames=12):
    # (원본 코드의 get_shot_based_frames 전체 내용 복사)
    # ... SceneDetect 로직 ...
    duration = chunk_end - chunk_start
    if duration < 0.5: return [chunk_start + duration/2]

    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))

    try:
        base_fps = video_manager.get_framerate()
        if base_fps is None: base_fps = 24.0

        start_tc = FrameTimecode(timecode=chunk_start, fps=base_fps)
        end_tc = FrameTimecode(timecode=chunk_end, fps=base_fps)

        video_manager.set_duration(start_time=start_tc, end_time=end_tc)
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        scene_list = scene_manager.get_scene_list()
    except Exception as e:
        print(f"  ⚠️ Scene detection failed: {e}")
        return []
    finally:
        video_manager.release()
    
    # ... (이하 원본의 timestamp 계산 로직 동일) ...
    if not scene_list:
        shot_list = [(chunk_start, chunk_end)]
    else:
        shot_list = []
        for i, (start, end) in enumerate(scene_list):
            abs_start = start.get_seconds()
            abs_end = end.get_seconds()
            if abs_start < 0.1 and chunk_start > 10.0:
                 abs_start += chunk_start
                 abs_end += chunk_start
            abs_start = max(chunk_start, abs_start)
            abs_end = min(chunk_end, abs_end)
            if abs_end > abs_start:
                shot_list.append((abs_start, abs_end))

    final_timestamps = []
    for shot_idx, (s_time, e_time) in enumerate(shot_list):
        shot_duration = e_time - s_time
        if shot_duration < 0.5:
            final_timestamps.append(s_time + shot_duration / 2)
            continue
        t_start = s_time + 0.05
        t_end = e_time - 0.05
        if t_start >= t_end: t_start = s_time; t_end = e_time

        target_count = int(shot_duration * 2)
        num_frames = max(2, target_count)
        current_shot_frames = np.linspace(t_start, t_end, num_frames).tolist()
        final_timestamps.extend(current_shot_frames)

    final_timestamps = sorted(list(set(final_timestamps)))
    if len(final_timestamps) > max_total_frames:
        indices = np.linspace(0, len(final_timestamps) - 1, max_total_frames, dtype=int)
        final_timestamps = [final_timestamps[i] for i in indices]

    return final_timestamps

def extract_frames_for_chunk(video_path, timestamps, output_folder):
    frame_paths = []
    pil_images = []
    for fi, ts in enumerate(timestamps):
        out_path = os.path.join(output_folder, f"f{fi:03d}.jpg")
        try:
            (ffmpeg.input(video_path, ss=max(0, float(ts))).filter("scale", 768, -1)
             .output(out_path, vframes=1).overwrite_output().run(quiet=True))
            if os.path.exists(out_path):
                frame_paths.append(out_path)
                pil_images.append(Image.open(out_path).convert("RGB"))
        except ffmpeg.Error: break
    return pil_images