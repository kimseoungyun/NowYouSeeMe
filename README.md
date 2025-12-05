# Ko-AD 프로젝트 개요

> **Ko-AD (Korean Audio Description)** 는 시각장애인 및 저시력 이용자를 위해 영상 콘텐츠의 시각 정보를 자동으로 분석하고, **대사와 겹치지 않는 구간에 화면해설(Audio Description, AD)을 삽입**하는 웹 기반 서비스입니다.

---

## 목차

1. [프로젝트 목표](#1-프로젝트-목표)
2. [기술 스택](#2-기술-스택)
3. [프로젝트 구조](#3-프로젝트-구조)
4. [핵심 컴포넌트 설명](#4-핵심-컴포넌트-설명)
5. [전체 처리 흐름 (Flow)](#5-전체-처리-흐름-flow)
6. [API 엔드포인트](#6-api-엔드포인트)
7. [로컬 개발 환경 설정](#7-로컬-개발-환경-설정)
8. [Docker를 이용한 배포](#8-docker를-이용한-배포)
9. [환경 변수](#9-환경-변수)

---

## 1. 프로젝트 목표

- **자동 화면해설 생성**: Google Gemini AI를 활용하여 영상의 시각적 정보를 분석하고, 대사가 없는 구간(침묵 구간)에 삽입할 화면해설 텍스트를 자동 생성합니다.
- **TTS 변환**: 생성된 화면해설 텍스트를 Google TTS(Text-to-Speech)를 통해 음성으로 변환합니다.
- **오디오 믹싱**: 원본 영상의 오디오와 생성된 화면해설 음성을 자동으로 믹싱하여 최종 결과물을 생성합니다.
- **웹 기반 편집 UI**: 사용자가 생성된 화면해설을 확인하고, 필요시 수정하며, 최종 영상을 내보낼 수 있는 직관적인 웹 인터페이스를 제공합니다.

---

## 2. 기술 스택

### Frontend
| 기술 | 용도 |
|------|------|
| **React 18** | UI 컴포넌트 라이브러리 |
| **Vite** | 빌드 도구 및 개발 서버 |
| **TypeScript / JavaScript** | 프론트엔드 개발 언어 |
| **React Router** | 클라이언트 사이드 라우팅 |

### Backend
| 기술 | 용도 |
|------|------|
| **Node.js 20** | 백엔드 런타임 |
| **Express.js** | REST API 서버 프레임워크 |
| **Multer** | 파일 업로드 처리 |
| **Python 3** | AI/TTS 스크립트 실행 |

### AI & 미디어 처리
| 기술 | 용도 |
|------|------|
| **Google Gemini API** | 영상 분석 및 화면해설 텍스트 생성 |
| **gTTS (Google TTS)** | 텍스트를 음성으로 변환 |
| **FFmpeg** | 비디오/오디오 처리 및 믹싱 |
| **pydub** | 오디오 파일 조작 |
| **yt-dlp** | YouTube 영상 다운로드 |

### 배포 인프라
| 기술 | 용도 |
|------|------|
| **Docker** | 컨테이너화 |
| **Docker Compose** | 멀티 컨테이너 오케스트레이션 |
| **Nginx** | 프론트엔드 정적 파일 서빙 및 리버스 프록시 |

---

## 3. 프로젝트 구조

```
Ko-AD/
├── Cursor/                          # 메인 애플리케이션 디렉토리
│   ├── python/                      # Python AI/TTS 스크립트
│   │   ├── get_AD.py               # Gemini API를 이용한 화면해설 생성
│   │   ├── gemini_json_tts.py      # TTS 합성 및 오디오 믹싱
│   │   ├── download_youtube.py     # YouTube 영상 다운로드
│   │   ├── requirements.txt        # Python 의존성 (로컬용)
│   │   └── requirements.docker.txt # Python 의존성 (Docker용)
│   │
│   ├── server/                      # Node.js 백엔드 서버
│   │   ├── src/
│   │   │   ├── index.js            # Express 서버 메인 엔트리포인트
│   │   │   └── jobStore.js         # 작업 상태 관리 (JSON 파일 기반)
│   │   ├── storage/                 # 파일 저장소
│   │   │   ├── uploads/            # 업로드된 원본 영상
│   │   │   ├── ad_json/            # 생성된 화면해설 JSON
│   │   │   ├── tts/                # TTS 변환 결과물
│   │   │   └── exports/            # 최종 내보내기 파일
│   │   ├── Dockerfile              # 백엔드 Docker 이미지 정의
│   │   └── package.json
│   │
│   └── web-ads-app/                 # React 프론트엔드
│       ├── src/
│       │   ├── App.jsx             # 메인 앱 컴포넌트
│       │   ├── pages/
│       │   │   ├── UploadPage.jsx  # 영상 업로드 페이지
│       │   │   └── VideoEditorPage.tsx # 비디오 편집기 페이지
│       │   └── components/
│       │       ├── VideoEditorLayout.tsx # 편집기 레이아웃
│       │       └── VideoTimeline.jsx     # 타임라인 컴포넌트
│       ├── Dockerfile              # 프론트엔드 Docker 이미지 정의
│       ├── nginx.conf              # Nginx 설정 (프록시 포함)
│       └── package.json
│
├── docker-compose.yml               # Docker Compose 설정
├── env.example                      # 환경 변수 예시
└── PROJECT_OVERVIEW.md              # 이 문서
```

---

## 4. 핵심 컴포넌트 설명

### 4.1 Frontend (React/Vite)

#### 페이지 구조
- **`/`**: 제품 소개 페이지 (ProductPage)
- **`/upload`**: 영상 업로드 페이지 (UploadPage)
- **`/video-editor`**: 메인 비디오 편집기 페이지 (VideoEditorPage)

#### 주요 컴포넌트
- **`VideoEditorLayout`**: YouTube Studio 스타일의 편집기 레이아웃
  - 좌측 사이드바: 파일 정보 및 메타데이터
  - 중앙 상단: 비디오 프리뷰 플레이어
  - 우측 상단: AD 스크립트 패널 (생성/수정/TTS 변환)
  - 하단: 타임라인 (웨이브폼 + 썸네일)

- **`VideoTimeline`**: 오디오 웨이브폼과 비디오 썸네일을 표시하는 타임라인 컴포넌트

### 4.2 Backend (Node.js/Express)

#### 주요 역할
1. **파일 업로드 처리**: Multer를 사용하여 대용량 비디오 파일 업로드 (최대 10GB)
2. **작업 관리**: JSON 파일 기반의 작업 상태 관리 (JobStore)
3. **Python 스크립트 실행**: `spawn`을 통해 Python 프로세스 호출
4. **정적 파일 서빙**: 업로드된 영상, TTS 결과물 등

#### API 구조
```
/api/health          - 서버 상태 확인
/api/upload          - 비디오 파일 업로드
/api/upload-youtube  - YouTube URL로 영상 다운로드
/api/generate-ad     - 화면해설 텍스트 생성 (Gemini)
/api/generate-tts    - TTS 변환 및 오디오 믹싱
/api/export-with-ad  - 최종 영상 내보내기
/api/jobs/*          - 작업 상태 조회/관리
```

### 4.3 Python AI 모듈

#### `get_AD.py` - 화면해설 생성
- **입력**: 비디오 파일 경로
- **처리**: Google Gemini API에 영상을 전송하여 분석
- **출력**: 침묵 구간과 해당 구간의 화면해설 텍스트 (JSON)

```json
{
  "audio_descriptions": [
    {
      "start_time": "0:03.5",
      "end_time": "0:06.1",
      "duration_sec": 2.6,
      "description": "운전자가 커피를 한 모금 마신다."
    }
  ]
}
```

#### `gemini_json_tts.py` - TTS 변환
- **입력**: 비디오 파일, AD JSON 파일
- **처리**:
  1. 원본 영상에서 오디오 추출 (WAV)
  2. 각 AD 세그먼트를 gTTS로 음성 변환
  3. 원본 오디오와 AD 오디오 믹싱
  4. FFmpeg로 비디오와 믹싱된 오디오 결합
- **출력**: AD가 포함된 최종 비디오/오디오 파일

### 4.4 Docker 구성

#### 컨테이너 구조
```
┌─────────────────────────────────────────────────────────┐
│                    Docker Network                        │
│  ┌─────────────────┐      ┌─────────────────────────┐   │
│  │   frontend      │      │       backend           │   │
│  │   (Nginx)       │─────▶│   (Node.js + Python)    │   │
│  │   Port: 80      │      │   Port: 4000            │   │
│  └─────────────────┘      └─────────────────────────┘   │
│         │                           │                    │
│         │                           │                    │
│         ▼                           ▼                    │
│    정적 파일 서빙              API 처리 + 파일 저장        │
│    /api/* 프록시              Python 스크립트 실행        │
└─────────────────────────────────────────────────────────┘
```

#### 왜 Docker를 사용하는가?
1. **환경 일관성**: 로컬, 개발, 프로덕션 환경에서 동일한 실행 환경 보장
2. **의존성 관리**: Node.js, Python, FFmpeg 등 복잡한 의존성을 이미지에 포함
3. **배포 간소화**: `docker compose up` 한 명령으로 전체 스택 실행
4. **격리성**: 호스트 시스템과 독립적인 실행 환경

---

## 5. 전체 처리 흐름 (Flow)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           사용자 워크플로우                                │
└──────────────────────────────────────────────────────────────────────────┘

[1] 영상 업로드
    ┌─────────┐     POST /api/upload      ┌─────────┐
    │ Browser │ ────────────────────────▶ │ Backend │
    │ (React) │                           │ (Node)  │
    └─────────┘                           └────┬────┘
                                               │
                                               ▼
                                    storage/uploads/{uuid}.mp4
                                               │
                                               │
[2] AD 생성 (화면해설 텍스트)                    │
    ┌─────────┐     POST /api/generate-ad  ┌───┴─────┐     spawn      ┌────────┐
    │ Browser │ ─────────────────────────▶ │ Backend │ ─────────────▶ │ Python │
    │ (React) │                            │ (Node)  │                │get_AD.py│
    └─────────┘                            └────┬────┘                └────┬───┘
                                                │                          │
                                                │                          ▼
                                                │                   Gemini API 호출
                                                │                          │
                                                │                          ▼
                                                │              storage/ad_json/{uuid}.ad.json
                                                │                          │
                                                ◀──────────────────────────┘
                                                │
                                                │
[3] TTS 변환 (음성 합성)                         │
    ┌─────────┐     POST /api/generate-tts ┌───┴─────┐     spawn      ┌────────────────┐
    │ Browser │ ─────────────────────────▶ │ Backend │ ─────────────▶ │     Python     │
    │ (React) │                            │ (Node)  │                │gemini_json_tts │
    └─────────┘                            └────┬────┘                └───────┬────────┘
                                                │                             │
                                                │                             ▼
                                                │                    1. 원본 오디오 추출
                                                │                    2. gTTS로 AD 음성 생성
                                                │                    3. 오디오 믹싱
                                                │                    4. FFmpeg로 비디오 결합
                                                │                             │
                                                │                             ▼
                                                │                storage/tts/{uuid}_ad_mix.mp4
                                                │                             │
                                                ◀─────────────────────────────┘
                                                │
                                                │
[4] 내보내기 (Export)                            │
    ┌─────────┐   POST /api/export-with-ad ┌───┴─────┐
    │ Browser │ ─────────────────────────▶ │ Backend │
    │ (React) │                            │ (Node)  │
    └─────────┘                            └────┬────┘
                                                │
                                                ▼
                                    storage/exports/{uuid}_export.mp4
                                                │
                                                ▼
                                         다운로드 링크 반환
```

---

## 6. API 엔드포인트

### 6.1 헬스 체크
```http
GET /api/health
```
**응답**: `{ "ok": true, "timestamp": 1234567890 }`

### 6.2 비디오 업로드
```http
POST /api/upload
Content-Type: multipart/form-data

Body: video (file)
```
**응답**:
```json
{
  "id": "uuid-xxxx",
  "videoId": "uuid-xxxx",
  "sourceFileName": "example.mp4",
  "serverPath": "/app/server/storage/uploads/uuid-xxxx.mp4",
  "originalVideoUrl": "/static/upload/uuid-xxxx.mp4",
  "status": "queued"
}
```

### 6.3 YouTube 영상 다운로드
```http
POST /api/upload-youtube
Content-Type: application/json

Body: { "url": "https://www.youtube.com/watch?v=xxxxx" }
```

### 6.4 화면해설 생성
```http
POST /api/generate-ad
Content-Type: application/json

Body: {
  "video_id": "uuid-xxxx",
  "server_path": "/app/server/storage/uploads/uuid-xxxx.mp4",
  "lang": "ko"  // "ko" 또는 "en"
}
```
**응답**:
```json
{
  "status": "generated",
  "segments": [
    { "id": 1, "start": 3.5, "end": 6.1, "text": "운전자가 커피를 마신다." }
  ]
}
```

### 6.5 TTS 변환
```http
POST /api/generate-tts
Content-Type: application/json

Body: {
  "videoId": "uuid-xxxx",
  "lang": "ko"
}
```
**응답**:
```json
{
  "status": "ok",
  "videoId": "uuid-xxxx",
  "adVideoUrl": "/static/tts/uuid-xxxx_ad_mix.mp4",
  "adAudioUrl": "/static/tts/uuid-xxxx_ad_mix.wav"
}
```

### 6.6 영상 내보내기
```http
POST /api/export-with-ad
Content-Type: application/json

Body: {
  "videoId": "uuid-xxxx",
  "serverPath": "/app/server/storage/uploads/uuid-xxxx.mp4",
  "adSegments": [...],
  "options": { "language": "ko" }
}
```
**응답**:
```json
{
  "status": "ok",
  "downloadUrl": "/static/exports/uuid-xxxx_ad_export_2025-11-25.mp4",
  "fileName": "uuid-xxxx_ad_export_2025-11-25.mp4"
}
```

---

## 7. 로컬 개발 환경 설정

### 7.1 필수 요구사항
- **Node.js 20+**
- **Python 3.11+**
- **FFmpeg** (시스템에 설치되어 PATH에 포함)
- **Google Gemini API Key**

### 7.2 설치 및 실행

#### 1) 백엔드 설정
```bash
cd Cursor/server
npm install

# 환경 변수 설정
cp env.sample .env
# .env 파일을 열어 GEMINI_API_KEY 설정
```

#### 2) Python 의존성 설치
```bash
cd Cursor/python
pip install -r requirements.txt
```

#### 3) 프론트엔드 설정
```bash
cd Cursor/web-ads-app
npm install
```

#### 4) 개발 서버 실행
```bash
# 터미널 1: 백엔드
cd Cursor/server
npm run dev    # http://localhost:4000

# 터미널 2: 프론트엔드
cd Cursor/web-ads-app
npm run dev    # http://localhost:5173
```

브라우저에서 `http://localhost:5173`으로 접속합니다.
Vite 개발 서버가 `/api/*` 요청을 백엔드(4000번 포트)로 자동 프록시합니다.

---

## 8. Docker를 이용한 배포

### 8.1 Docker의 역할

Docker는 이 프로젝트에서 다음과 같은 역할을 합니다:

| 역할 | 설명 |
|------|------|
| **환경 패키징** | Node.js, Python, FFmpeg 등 모든 의존성을 하나의 이미지로 패키징 |
| **프로덕션 빌드** | 프론트엔드를 빌드하고 Nginx로 서빙 |
| **네트워크 구성** | 프론트엔드와 백엔드 간 내부 통신 설정 |
| **리버스 프록시** | Nginx가 `/api/*` 요청을 백엔드로 전달 |

### 8.2 컨테이너 구성

#### Frontend 컨테이너 (`ad_frontend`)
- **베이스 이미지**: `nginx:alpine`
- **역할**: 
  - React 앱의 정적 파일 서빙
  - `/api/*`, `/static/*`, `/files/*` 요청을 백엔드로 프록시
- **포트**: 80

#### Backend 컨테이너 (`ad_backend`)
- **베이스 이미지**: `node:20-bullseye`
- **추가 설치**: Python 3, FFmpeg
- **역할**:
  - REST API 서버
  - Python 스크립트 실행 (AD 생성, TTS)
  - 파일 저장소 관리
- **포트**: 4000

### 8.3 로컬에서 Docker 실행

```bash
# 프로젝트 루트에서
cp env.example .env
# .env 파일을 열어 GEMINI_API_KEY 설정

# 빌드 및 실행
docker compose up -d --build

# 상태 확인
docker compose ps

# 로그 확인
docker compose logs -f
```

브라우저에서 `http://localhost`로 접속합니다.

### 8.4 AWS EC2 배포

자세한 EC2 배포 가이드는 별도 문서를 참조하세요.

**간략한 순서:**
1. EC2 인스턴스에 SSH 접속
2. Docker 및 Docker Compose 설치
3. 프로젝트 클론 (`git clone`)
4. 환경 변수 설정 (`.env`)
5. `docker compose up -d --build`
6. 보안 그룹에서 80번 포트 개방

---

## 9. 환경 변수

### `env.example` 파일 내용

```properties
# Google Gemini API Key (필수)
# AD 생성에 사용됩니다.
GEMINI_API_KEY=your_api_key_here

# Python 실행 파일 경로 (Docker에서는 python3)
PYTHON_TTS_EXECUTABLE=python3

# CORS 허용 Origin
# 개발: http://localhost:5173
# 프로덕션: * 또는 특정 도메인
CLIENT_ORIGIN=*
```

### 주요 환경 변수 설명

| 변수 | 필수 | 설명 |
|------|------|------|
| `GEMINI_API_KEY` | ✅ | Google Gemini API 키. AD 생성에 필요 |
| `PYTHON_TTS_EXECUTABLE` | ❌ | Python 실행 파일 경로 (기본: `python`) |
| `CLIENT_ORIGIN` | ❌ | CORS 허용 Origin (기본: `http://localhost:5173`) |
| `PORT` | ❌ | 백엔드 서버 포트 (기본: `4000`) |
| `NODE_ENV` | ❌ | 실행 환경 (`development` / `production`) |

---

## 부록: 주요 파일 요약

| 파일 경로 | 설명 |
|-----------|------|
| `Cursor/server/src/index.js` | Express 서버 메인 파일. 모든 API 엔드포인트 정의 |
| `Cursor/python/get_AD.py` | Gemini API를 호출하여 화면해설 JSON 생성 |
| `Cursor/python/gemini_json_tts.py` | TTS 합성 및 오디오 믹싱 |
| `Cursor/web-ads-app/src/components/VideoEditorLayout.tsx` | 메인 편집기 UI 컴포넌트 |
| `docker-compose.yml` | Docker Compose 설정 |
| `Cursor/server/Dockerfile` | 백엔드 Docker 이미지 정의 |
| `Cursor/web-ads-app/Dockerfile` | 프론트엔드 Docker 이미지 정의 |
| `Cursor/web-ads-app/nginx.conf` | Nginx 리버스 프록시 설정 |

---

## 라이선스

이 프로젝트는 내부 개발용으로 작성되었습니다.

---

*마지막 업데이트: 2025-11-25*

