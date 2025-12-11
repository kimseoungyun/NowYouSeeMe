## 통합 인퍼런스/배포 가이드

단일 스크립트(`unified_inference.py`)로 AD 생성부터 TTS 입힌 최종 영상까지 만듭니다.

### 입력 예시
```
python unified_inference.py
                    --model gemini
                    --video_path <YOUR_VIDEO_PATH>
                    --lang ko
                    --voice_profile gemini_kor_female
                    --gemini_api_key <YOUR_GEMINI_API_KEY>
```

### 준비물
- Python 3.11+ (3.12도 호환, 3.13은 `audioop-lts` 필요)
- FFmpeg 설치 및 PATH 등록 (윈도우는 choco/winget 또는 다운로드)
- 의존성 설치: `pip install -r requirements.txt`
  - Coqui TTS가 무겁다면 먼저 `install_tts_dependencies.bat`(Windows) 또는 `install_tts_dependencies.sh`(Linux/Mac) 실행

### 환경 변수
- `GEMINI_API_KEY` : Gemini / Jack / Gemini TTS용 (cookie는 불필요)
- `OPENAI_API_KEY` : GPT 파이프라인용 (cookie는 불필요)
- (옵션) `.env` 대신 CLI 인자로 키를 넘겨도 됩니다.

### 핵심 명령어
- Gemini:  
  `python unified_inference.py --model gemini --video_path sample.mp4 --lang ko --voice_profile gemini_kor_female --gemini_api_key <KEY>`
- Jack(Gemini 3 기반):  
  `python unified_inference.py --model jack --video_path sample.mp4 --lang en --gemini_api_key <KEY>`
- GPT:  
  `python unified_inference.py --model gpt --video_path sample.mp4 --lang ko --openai_api_key <KEY> --whisper_model small --min_silence 2.5`
- Cookie(기존 inference 파이프라인):  
  `python unified_inference.py --model cookie --video_path sample.mp4 --lang ko --output_dir ./outputs_cookie`
- JSON만 받고 TTS/영상 스킵: `--skip_tts`
- 더킹 끄기: `--disable_ducking`

### 인자 정리
- `--model` : gemini | jack | gpt | cookie
- `--video_path` : 입력 mp4 경로
- `--lang` : ko | en
- `--voice_profile` : gtts | kor_male | kor_female | eng_male | eng_female | gemini_kor_female | gemini_kor_male | gemini_eng_female | gemini_eng_male
- `[gpt 전용] --fps` (기본 2.0), `--whisper_model` (tiny/base/small/medium/large), `--min_silence` (기본 2.5초)

### 출력
- AD JSON: `outputs/<videoid>_*.ad.json`
- 최종 오디오/영상: `outputs/<videoid>_ad_mix.wav`, `outputs/<videoid>_ad_mix.mp4`
- GPT 파이프라인 중간 산출물: `outputs/gpt_data/<videoid>/...`
- Cookie 파이프라인 결과: `outputs_cookie/<videoid>_<lang>.ad.json` (혹은 지정한 output_dir)

### 트러블슈팅
- FFmpeg 미설치 에러 → FFmpeg 설치 후 PATH 추가
- Coqui TTS 로드 실패 → install_tts_dependencies.* 실행 후 재시도, 또는 `--voice_profile gtts`로 폴백
- Gemini TTS 사용 시에도 `GEMINI_API_KEY` 필요 (음성 프로필이 gemini_*일 때)
- Cookie 실행 시 symlink란? → inference 폴더를 연결하는 “바로가기” 같은 링크입니다. Windows에서 링크 생성이 막히면 자동으로 폴더를 복사하여 동작하게 처리했습니다. (용량 증가 가능)

