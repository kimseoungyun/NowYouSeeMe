## Cookie 모델 단독 배포 패키지

이 디렉터리는 `Ko-AD/inference` 파이프라인과 `run_inference.py`를 분리한 배포용 복사본입니다. GitHub에 이 폴더만 올려도 cookie(=inference) 모델로 AD JSON을 생성할 수 있습니다.

### 구성
- `run_inference.py` : inference/main.py를 호출해 AD JSON을 뽑는 래퍼
- `inference/` : 기존 파이프라인 전체 복사본 (config/main 포함)

### 사전 준비
1) Python 3.11+  
2) FFmpeg 설치 후 PATH 등록  
3) 가상환경 생성 후 필수 라이브러리 설치:
```bash
python -m venv venv
venv/Scripts/activate   # mac/linux: source venv/bin/activate
pip install --upgrade pip
pip install -r inference/requirement.txt
```

### 실행
```bash
python run_inference.py --video_path /path/to/video.mp4 --output_dir ./output --lang ko
```
- 결과: `output/<video>_ko.ad.json` (audio_descriptions 포맷)
- 영어로 출력하려면 `--lang en`

### 참고
- `run_inference.py`는 inference 폴더의 config를 수정하지 않고, 입력 비디오를 복사/심볼릭링크로 맞춰 실행합니다.
- inference 내부 sample 데이터(walter_white_*, chunk_frames 등)는 안전히 삭제해도 실행에는 영향 없습니다.

