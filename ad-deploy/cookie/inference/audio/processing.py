import whisper
import pandas as pd
from moviepy.editor import VideoFileClip
import torch
import config  # config.py 임포트

def extract_and_transcribe():
    print(">> [Audio] Extracting audio...")
    try:
        video_clip = VideoFileClip(config.VIDEO_PATH)
        video_clip.audio.write_audiofile(config.EXTRACTED_AUDIO_PATH, codec='pcm_s16le', verbose=False, logger=None)
        video_clip.close()
    except Exception as e:
        print(f"Audio Extraction Warning: {e}")
        return pd.DataFrame() # 빈 데이터프레임 반환

    print(">> [Audio] Transcribing with Whisper...")
    whisper_model = whisper.load_model("medium", device=config.DEVICE)
    result = whisper_model.transcribe(config.EXTRACTED_AUDIO_PATH, word_timestamps=True)
    
    del whisper_model
    torch.cuda.empty_cache()
    
    return _process_word_timestamps(result)

def _process_word_timestamps(result):
    # 단어 재조립 내부 로직
    SILENCE_THRESHOLD = 0.5
    MIN_DURATION = 0.5
    new_segments = []
    current_words = []
    current_start = None
    last_end = 0.0

    all_words = [w for seg in result['segments'] for w in seg['words']]

    for w in all_words:
        word_text = w['word'].strip()
        start, end = w['start'], w['end']

        if not current_words:
            current_words.append(word_text); current_start = start; last_end = end; continue

        gap = start - last_end
        ends_sentence = current_words[-1][-1] in ".?!"

        if gap > SILENCE_THRESHOLD or ends_sentence:
            if last_end - current_start >= MIN_DURATION:
                new_segments.append({"start": current_start, "end": last_end, "text": " ".join(current_words)})
            current_words = [word_text]; current_start = start; last_end = end
        else:
            current_words.append(word_text); last_end = end

    if current_words:
        new_segments.append({"start": current_start, "end": last_end, "text": " ".join(current_words)})

    final_dataset = [{"id": i, "start": round(s['start'], 2), "end": round(s['end'], 2), "text": s['text']} for i, s in enumerate(new_segments)]
    df = pd.DataFrame(final_dataset)
    print(f">> Whisper processing complete. {len(df)} segments.")
    return df