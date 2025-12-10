import subprocess
import json
import uuid
import os

def prepare_video_input(video_path: str) -> str:
    """
    AV1 / VP9 / HEVC → 자동으로 H.264 변환.
    """
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return video_path

    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", video_path
        ]
        probe_output = subprocess.check_output(cmd).decode("utf-8")
        info = json.loads(probe_output)

        v_stream = next(s for s in info["streams"] if s["codec_type"] == "video")
        codec = v_stream.get("codec_name", "").lower()

        print(f">> Detected codec: {codec}")

    except Exception as e:
        print(f"⚠️ ffprobe failed: {e}")
        codec = "unknown"

    if codec in ["h264", "avc1"]:
        print(">> Video already H.264 — no conversion needed.")
        return video_path

    output_path = f"{video_path}.converted_{uuid.uuid4().hex}.mp4"
    print(f">> Converting to H.264: {output_path}")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "19",
        "-c:a", "aac",
        "-movflags", "+faststart",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        print(">> Conversion completed.")
        return output_path
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return video_path  # fallback