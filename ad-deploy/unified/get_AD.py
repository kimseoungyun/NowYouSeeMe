"""
Wrapper module for backward compatibility.
This module re-exports everything from get_AD_gemini.py.

For new code, please import directly from get_AD_gemini instead.
"""

# Re-export everything from get_AD_gemini for backward compatibility
from get_AD_gemini import (
    generate_ad_for_video,
    save_ad_json,
    extract_segments_from_gemini,
    repair_truncated_json,
    GEMINI_MODEL,
    PROMPT_KO,
    PROMPT_EN,
    logger
)

# Make all exports available
__all__ = [
    'generate_ad_for_video',
    'save_ad_json', 
    'extract_segments_from_gemini',
    'repair_truncated_json',
    'GEMINI_MODEL',
    'PROMPT_KO',
    'PROMPT_EN',
    'logger'
]

if __name__ == "__main__":
    # Run the main function from get_AD_gemini
    import sys
    import argparse
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Generate Audio Description for a video using Gemini.")
    parser.add_argument("video_path", help="Path to the input video file")
    parser.add_argument("--api_key", help="Google Gemini API Key")
    parser.add_argument("--lang", default="ko", choices=["ko", "en"], help="Language for AD generation (ko/en)")
    
    args = parser.parse_args()
    
    import os
    try:
        full_data, segments = generate_ad_for_video(args.video_path, api_key=args.api_key, lang=args.lang)
        
        video_id = os.path.splitext(os.path.basename(args.video_path))[0]
        
        json_path = save_ad_json(video_id, full_data, "./ad_json")
        print(f"\\n=== Saved to {json_path} ===")
        print(f"AD segments: {len(segments)}")
        
    except Exception as e:
        logger.exception("[CLI] Error during AD generation")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
