#!/bin/bash
echo "========================================"
echo "TTS 의존성 패키지 설치 스크립트"
echo "========================================"
echo ""

echo "[1/4] soundfile 설치 중..."
pip install soundfile>=0.12.0 || exit 1

echo "[2/4] librosa 설치 중..."
pip install librosa>=0.11.0 || exit 1

echo "[3/4] pyloudnorm 설치 중..."
pip install pyloudnorm>=0.1.0 || exit 1

echo "[4/4] coqui-tts 설치 중... (시간이 걸릴 수 있습니다)"
pip install coqui-tts>=0.27.0 || exit 1

echo ""
echo "========================================"
echo "모든 패키지 설치 완료!"
echo "========================================"
echo ""
echo "설치된 패키지 확인:"
pip show soundfile librosa pyloudnorm coqui-tts

