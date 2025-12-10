# TTS 기능 설치 가이드

## 문제 해결

TTS 생성 시 다음 오류가 발생하는 경우:

```
❌ Coqui TTS 미설치: No module named 'TTS'
❌ soundfile 미설치: No module named 'soundfile'
❌ librosa 미설치: No module named 'librosa'
⚠️ pyloudnorm 미설치: No module named 'pyloudnorm'
```

## 빠른 설치 방법

### Windows

```bash
cd Cursor/python
install_tts_dependencies.bat
```

또는 수동 설치:

```bash
pip install soundfile librosa pyloudnorm coqui-tts
```

### Linux / macOS

```bash
cd Cursor/python
bash install_tts_dependencies.sh
```

또는 수동 설치:

```bash
pip install soundfile librosa pyloudnorm coqui-tts
```

## 필요한 패키지 목록

### 필수 패키지 (더킹 기능용)
- `soundfile>=0.12.0` - 오디오 파일 I/O
- `librosa>=0.11.0` - 오디오 리샘플링

### 필수 패키지 (Coqui TTS용)
- `coqui-tts>=0.27.0` - 고급 TTS 엔진 (4가지 음성 프로필)

### 선택 패키지 (더킹 볼륨 정규화용)
- `pyloudnorm>=0.1.0` - LUFS 측정

## 설치 확인

설치 후 다음 명령어로 확인:

```bash
pip show soundfile librosa pyloudnorm coqui-tts
```

## 기능별 필요 패키지

| 기능 | 필요 패키지 |
|------|------------|
| 기본 TTS (gTTS) | 없음 (기본 포함) |
| 고급 TTS (Coqui) | `coqui-tts` |
| VAD 더킹 | `soundfile`, `librosa` |
| LUFS 정규화 | `pyloudnorm` |

## 문제 해결

### Coqui TTS 설치 실패

Coqui TTS는 큰 패키지이므로 설치에 시간이 걸릴 수 있습니다.

```bash
# GPU 지원이 있는 경우 (권장)
pip install coqui-tts

# CPU만 있는 경우
pip install coqui-tts --no-deps
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### soundfile 설치 실패 (Windows)

Windows에서 `soundfile` 설치 시 시스템 라이브러리가 필요할 수 있습니다:

```bash
# conda 사용 시
conda install -c conda-forge soundfile

# 또는 pip로 재시도
pip install --upgrade pip
pip install soundfile
```

### librosa 설치 실패

`librosa`는 많은 의존성이 있습니다:

```bash
pip install --upgrade pip setuptools wheel
pip install librosa
```

## 설치 후 테스트

TTS를 다시 실행하면 로그에 다음이 표시됩니다:

```
[TTS] ✅ Coqui TTS 설치됨: 0.27.2
[TTS] ✅ soundfile 설치됨: 0.12.1
[TTS] ✅ librosa 설치됨: 0.11.0
[TTS] ✅ pyloudnorm 설치됨: 0.1.1
```

## 추가 정보

- Coqui TTS 첫 실행 시 모델 다운로드 (~2GB)가 필요합니다
- GPU가 있으면 더 빠르게 동작합니다
- 패키지가 없어도 기본 gTTS로 폴백됩니다

