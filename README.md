# Ko-AD 프로젝트 개요

> **Ko-AD (Korean Audio Description)** 는 시각장애인 및 저시력 이용자를 위해 영상 콘텐츠의 시각 정보를 자동으로 분석하고, **대사와 겹치지 않는 구간에 화면해설(Audio Description, AD)을 삽입**하는 웹 기반 서비스입니다.

---

## 샘플 데모




https://github.com/user-attachments/assets/dbece2b3-b80f-413a-bcae-63b16dae8a87



**Inference 바로 가기:** 테스트 파이프라인을 실행해보고 싶다면 [ad-deploy/unified/README.md](./ad-deploy/unified/README.md) 가이드로 이동하세요.

## 서비스 시연


https://github.com/user-attachments/assets/57276b3f-09ed-4e88-9db5-1a480c01e13e



---

## 목차

1. [프로젝트 목표](#1-프로젝트-목표)
2. [기술 스택](#2-기술-스택)
3. [프로젝트 구조](#3-프로젝트-구조)
4. [핵심 컴포넌트 설명](#4-핵심-컴포넌트-설명)
5. [AD 생성 파이프라인](#5-ad-생성-파이프라인)
6. [TTS 시스템](#6-tts-시스템)
7. [전체 처리 흐름 (Flow)](#7-전체-처리-흐름-flow)
8. [API 엔드포인트](#8-api-엔드포인트)
9. [로컬 개발 환경 설정](#9-로컬-개발-환경-설정)
10. [Docker를 이용한 배포](#10-docker를-이용한-배포)
11. [환경 변수](#11-환경-변수)

---

## 1. 프로젝트 목표

- **자동 화면해설 생성**: Google Gemini AI를 활용하여 영상의 시각적 정보를 분석하고, 대사가 없는 구간(침묵 구간)에 삽입할 화면해설 텍스트를 자동 생성합니다.
- **다중 AI 모델 지원**: Gemini, GPT, Jack (Multi-stage Context Fusion), 로컬 Inference 파이프라인 등 다양한 AD 생성 모델을 지원합니다.
- **고품질 TTS 변환**: gTTS (기본), Coqui XTTS v2 (고급 4종 음성 프로필), Gemini TTS (Google AI) 등 다양한 음성 합성 옵션을 제공합니다.
- **VAD 기반 스마트 더킹**: Silero VAD를 사용하여 대사와 AD가 겹치는 구간의 볼륨을 자동으로 조절합니다.
- **웹 기반 편집 UI**: 사용자가 생성된 화면해설을 확인하고, 필요시 수정하며, 최종 영상을 내보낼 수 있는 직관적인 웹 인터페이스를 제공합니다.
- **사용자 인증 및 크레딧 시스템**: Supabase Auth를 통한 회원가입/로그인 및 크레딧 기반 사용량 관리를 지원합니다.

---

## 2. 기술 스택

### Frontend
| 기술 | 용도 |
|------|------|
| **React 18** | UI 컴포넌트 라이브러리 |
| **Vite 6** | 빌드 도구 및 개발 서버 |
| **TypeScript / JavaScript** | 프론트엔드 개발 언어 |
| **React Router 6** | 클라이언트 사이드 라우팅 |
| **Supabase JS** | 인증 클라이언트 |

### Backend
| 기술 | 용도 |
|------|------|
| **Node.js 20** | 백엔드 런타임 |
| **Express.js 4** | REST API 서버 프레임워크 |
| **Multer** | 파일 업로드 처리 (최대 10GB) |
| **Python 3.11+** | AI/TTS 스크립트 실행 |
| **Supabase** | 사용자 인증 및 데이터베이스 |

### AI & 미디어 처리
| 기술 | 용도 |
|------|------|
| **Google Gemini API** | 영상 분석 및 화면해설 텍스트 생성 (gemini-2.5-pro, gemini-3-pro-preview) |
| **OpenAI GPT** | 대안 AD 생성 모델 (옵션) |
| **gTTS** | 기본 텍스트 음성 변환 |
| **Coqui XTTS v2** | 고급 TTS (4종 음성 프로필 지원) |
| **Gemini TTS** | Google AI 기반 고품질 TTS |
| **Silero VAD** | 음성 활동 감지 (더킹용) |
| **FFmpeg** | 비디오/오디오 처리 및 믹싱 |
| **pydub** | 오디오 파일 조작 |
| **pyloudnorm** | LUFS 기반 오디오 정규화 |
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
│   │   ├── get_AD_gemini.py        # Gemini 3.0 기반 AD 생성 (기본)
│   │   ├── get_AD_jack.py          # Jack Method (Multi-stage Context Fusion)
│   │   ├── get_AD_gpt.py           # GPT 기반 AD 생성
│   │   ├── get_AD.py               # 레거시 AD 생성 스크립트
│   │   ├── gemini_json_tts.py      # TTS 합성 및 오디오 믹싱
│   │   ├── run_inference.py        # 로컬 Inference 파이프라인 래퍼
│   │   ├── inference_server.py     # Inference HTTP 서버
│   │   ├── download_youtube.py     # YouTube 영상 다운로드
│   │   ├── extract_for_gpt.py      # GPT용 비디오 데이터 추출
│   │   ├── get_VADscript_json.py   # VAD 스크립트 생성
│   │   ├── tts_voice/              # Coqui XTTS 음성 프로필 샘플
│   │   │   ├── cut_5min_kor_male.mp3
│   │   │   ├── cut_5min_kor_female.mp3
│   │   │   ├── cut_5min_eng_male.mp3
│   │   │   └── cut_5min_eng_female.mp3
│   │   ├── requirements.txt        # Python 의존성 (로컬용)
│   │   └── requirements.docker.txt # Python 의존성 (Docker용)
│   │
│   ├── server/                      # Node.js 백엔드 서버
│   │   ├── src/
│   │   │   ├── index.js            # Express 서버 메인 엔트리포인트
│   │   │   ├── authRoutes.js       # 인증 및 크레딧 API
│   │   │   ├── jobStore.js         # 작업 상태 관리 (JSON 파일 기반)
│   │   │   └── supabaseClient.js   # Supabase 클라이언트 설정
│   │   ├── storage/                 # 파일 저장소
│   │   │   ├── uploads/            # 업로드된 원본 영상
│   │   │   ├── ad_json/            # 생성된 화면해설 JSON
│   │   │   ├── tts/                # TTS 변환 결과물
│   │   │   ├── exports/            # 최종 내보내기 파일
│   │   │   ├── ratings/            # 사용자 평가 데이터
│   │   │   └── results/            # 기타 결과 파일
│   │   ├── data/                   # 작업 상태 JSON 저장
│   │   ├── Dockerfile              # 백엔드 Docker 이미지 정의
│   │   ├── env.sample              # 환경 변수 샘플
│   │   └── package.json
│   │
│   └── web-ads-app/                 # React 프론트엔드
│       ├── src/
│       │   ├── App.jsx             # 메인 앱 컴포넌트 (라우팅)
│       │   ├── main.jsx            # 엔트리 포인트
│       │   ├── pages/
│       │   │   ├── ProductPage.jsx      # 제품 소개 페이지
│       │   │   ├── UploadPage.jsx       # 영상 업로드 페이지
│       │   │   ├── EditorPage.jsx       # 에디터 진입 페이지
│       │   │   ├── VideoEditorPage.tsx  # 메인 비디오 편집기
│       │   │   ├── LoginPage.jsx        # 로그인 페이지
│       │   │   ├── SignupPage.jsx       # 회원가입 페이지
│       │   │   └── ProfilePage.jsx      # 프로필/크레딧 페이지
│       │   ├── components/
│       │   │   ├── AppHeader.tsx        # 헤더 컴포넌트
│       │   │   ├── VideoEditorLayout.tsx # 편집기 레이아웃
│       │   │   ├── VideoTimeline.jsx    # 타임라인 컴포넌트
│       │   │   ├── UploadPanel.jsx      # 업로드 패널
│       │   │   ├── Hero.jsx             # 히어로 섹션
│       │   │   └── ...
│       │   ├── contexts/
│       │   │   └── AuthContext.jsx      # 인증 상태 관리
│       │   ├── hooks/
│       │   │   ├── useThumbnails.js     # 썸네일 추출 훅
│       │   │   └── useWaveform.js       # 웨이브폼 생성 훅
│       │   └── lib/                     # 유틸리티
│       ├── Dockerfile              # 프론트엔드 Docker 이미지 정의
│       ├── nginx.conf              # Nginx 설정 (프록시 포함)
│       ├── vite.config.js          # Vite 설정
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
| 경로 | 페이지 | 설명 |
|------|--------|------|
| `/` | ProductPage | 제품 소개 및 랜딩 페이지 |
| `/upload` | UploadPage | 영상 업로드 (파일/YouTube URL) |
| `/editor` | EditorPage | 에디터 진입 페이지 |
| `/video-editor` | VideoEditorPage | 메인 비디오 편집기 |
| `/login` | LoginPage | 로그인 |
| `/signup` | SignupPage | 회원가입 |
| `/profile` | ProfilePage | 프로필 및 크레딧 관리 |

#### 주요 컴포넌트
- **`VideoEditorLayout`**: YouTube Studio 스타일의 편집기 레이아웃
  - 좌측 사이드바: 파일 정보 및 메타데이터
  - 중앙 상단: 비디오 프리뷰 플레이어
  - 우측 상단: AD 스크립트 패널 (생성/수정/TTS 변환)
  - 하단: 타임라인 (웨이브폼 + 썸네일)

- **`VideoTimeline`**: 오디오 웨이브폼과 비디오 썸네일을 표시하는 타임라인 컴포넌트

- **`AuthContext`**: Supabase 인증 상태 관리 (세션, 토큰, 사용자 정보)

### 4.2 Backend (Node.js/Express)

#### 주요 역할
1. **파일 업로드 처리**: Multer를 사용하여 대용량 비디오 파일 업로드 (최대 10GB)
2. **작업 관리**: JSON 파일 기반의 작업 상태 관리 (JobStore)
3. **Python 스크립트 실행**: `spawn`을 통해 Python 프로세스 호출
4. **정적 파일 서빙**: 업로드된 영상, TTS 결과물 등
5. **인증 처리**: Supabase Auth 기반 회원가입/로그인/크레딧 관리
6. **평가 데이터 관리**: 사용자 피드백 및 편집 이력 저장

#### 서버 설정
- **포트**: 4000 (기본) 또는 4001
- **최대 파일 크기**: 10GB
- **서버 타임아웃**: 1시간 (AD 생성/TTS 처리 대응)

### 4.3 Python AI 모듈

#### 디렉토리 구조
```
Cursor/python/
├── get_AD_gemini.py      # Gemini 3 Pro 기반 (기본 모델)
├── get_AD_jack.py        # Jack Method (Gemini 3 Pro Multi-stage)
├── get_AD_gpt.py         # GPT 기반 (OpenAI)
├── gemini_json_tts.py    # TTS 합성 엔진
├── run_inference.py      # 로컬 Inference 래퍼
└── download_youtube.py   # YouTube 다운로드
```

### 4.4 인증 및 크레딧 시스템

- **Supabase Auth**: 이메일/비밀번호 기반 인증
- **크레딧 시스템**: AD 생성 시 9.98 크레딧 차감
- **거래 내역**: 모든 크레딧 변동 기록 저장
- **프로필 관리**: 사용자 정보 및 사용량 통계

---

## 5. AD 생성 파이프라인

### 5.1 지원 모델

| 모델 | 설명 | Gemini 버전 |
|------|------|------------|
| **gemini** (기본) | 단일 프롬프트 기반 AD 생성 | gemini-3-pro-preview |
| **Jack** | Multi-stage Context Fusion | gemini-3-pro-preview |
| **gpt** | OpenAI GPT 기반 | - |
| **Cookie** | 로컬 Qwen-VL + Llama 파이프라인 | - |

### 5.2 Gemini 모델 (`get_AD_gemini.py`)

```python
# 주요 기능
- gemini-3-pro-preview with Thinking Mode disabled (thinking_budget=0)
- 침묵 구간 탐지 (2.5초 이상)
- 인물 지칭 규칙: "남자1", "여자1" 등 번호 기반
- 한국어/영어 프롬프트 지원
```

**출력 형식**:
```json
{
  "full_transcript": [
    {"time": "0:01.2", "speaker": "남자1", "text": "대사 내용..."}
  ],
  "audio_descriptions": [
    {
      "start_time": "0:03.5",
      "end_time": "0:06.1",
      "duration_sec": 2.6,
      "description": "남자1이 커피를 한 모금 마신다."
    }
  ]
}
```

### 5.3 Jack 모델 (`get_AD_jack.py`)

**Multi-stage Context Fusion 파이프라인** - Gemini 3 Pro 기반 고품질 AD 생성


JACK 모델은 단순한 장면 설명 생성이 아닌,
**제한된 시간 구간 내에서 시각적으로 가장 중요한 정보를 전달하는 것**을
핵심 목표로 설계되었다.


이를 위해 다음과 같은 설계 원칙을 정의하였다.
- **중요한 Description의 정의**
  - 모든 시각 정보를 나열하는 방식이 아닌, 장면의 이해에 결정적인 영향을 미치는 요소를 우선적으로 설명 대상으로 선정한다.
  - 중요도 판단은 시각 분석 결과뿐만 아니라
    메타데이터 및 대사(STT) 정보를 함께 활용하여 수행하였다.
  - 주요 기준은 다음과 같다.
    - Scene 전환 여부 및 장면 경계 정보
    - 주요 인물의 등장, 퇴장 또는 행동 변화
    - 대사 흐름상 서사 전개에 영향을 미치는 사건 발생
- **시간 제약 인식의 후처리 기반 반영**
  - AD는 항상 특정 무음 구간 또는 제한된 재생 구간 내에서
    재생되어야 하는 특성을 가지므로,
    시간 제약을 생성 단계가 아닌
    **후처리(Post-processing) 단계에서 명시적으로 반영**하였다.
  - 생성된 Description은
    구간 길이, 구간 개수 등의 조건에 따라
    압축, 분할 또는 재생성 과정을 거쳐
    실제 재생 가능 시간에 맞도록 조정된다.
- **최소 생성 길이 기준**
  - 지나치게 짧은 구간에서는
    의미 있는 설명 전달이 어렵다고 판단하여,
    AD 생성의 최소 길이를 약 **2.5~3초 이상**으로 설정하였다.
  - 해당 기준 미만의 구간에 대해서는
    설명 생략, 인접 구간과의 병합,
    또는 후처리 단계에서의 대안적 처리 전략을 적용한다.

    
#### 처리 플로우

```
┌─────────────────────────────────────────────────────────────┐
│                   Jack Method 파이프라인                      │
└─────────────────────────────────────────────────────────────┘

[Phase 1] 비디오 업로드
    │
    └──► Gemini Files API로 비디오 업로드
         wait_for_file_active() → 파일 ACTIVE 상태 대기

[Phase 2] 병렬 입력 생성 (asyncio.gather)
    │
    ├──► Metadata 추출 (PROMPT_METADATA)
    │     • video_title, overall_summary
    │     • scenes[]: scene_id, start/end_time
    │     • characters[]: id, name, appearance, visible_emotion
    │     • visible_actions[], relationships[], visual_focus
    │
    ├──► Core AD 생성 (PROMPT_AD)
    │     • 무음 구간 탐지 (≥2.5초)
    │     • 시각적 핵심 정보 묘사
    │     • audio_descriptions[] 배열 생성
    │
    └──► STT 추출 (PROMPT_STT)
          • full_transcript[]: time, speaker, text
          • [Sound] 태그로 효과음 표시

[Phase 3] 최종 통합 (Context Fusion)
    │
    └──► FINAL_PROMPT로 3개 결과 병합
         • Metadata로 맥락 풍부화
         • Core AD 누락 요소 보완
         • STT 발화 참조 (짧은 감탄사 무시)
         • 중복 제거, duration 내 문장 최적화

[Phase 4] Stage 2 압축 (Compression)
    │
    └──► PROMPT_STAGE2로 텍스트 압축
         • allowed_chars = duration_sec × chars_per_sec
         • 핵심 시각 정보만 유지
         • 한 항목당 한 문장

         최종 출력: audio_descriptions[] (압축 완료)
```

#### 프롬프트 구조

| 프롬프트 | 목적 | 주요 규칙 |
|----------|------|-----------|
| **PROMPT_METADATA** | 영상 메타데이터 추출 | 시각적 확인 가능 정보만, 추측 금지 |
| **PROMPT_AD** | 무음 구간 Core AD | ≥2.5초 무음 탐지, 화면 정보만 묘사 |
| **PROMPT_STT** | 대사/효과음 추출 | 화자별 대사, [Sound] 태그 |
| **FINAL_PROMPT** | 3개 결과 통합 | 중복 제거, duration 내 문장 |
| **PROMPT_STAGE2** | 텍스트 압축 | allowed_chars 제한, 한 문장 |

#### 모델 설정

```python
# 모델
GEMINI_MODEL_VISION = "gemini-3-pro-preview"  # 비전 분석용
GEMINI_MODEL_TEXT = "gemini-3-pro-preview"    # 텍스트 처리용

# API 호출 설정
BASE_CONFIG = {
    "temperature": 0,              # 결정적 출력
    "top_k": 1,
    "top_p": 0.00001,
    "max_output_tokens": 65536,
    "thinking_config": {
        "thinking_budget": 8192    # Thinking Mode 활성화
    },
}

# 압축 설정
chars_per_sec_ko = 7.0   # 한국어: 초당 7자
chars_per_sec_en = 15.0  # 영어: 초당 15자

# 비동기 처리
executor = ThreadPoolExecutor(max_workers=8)
```

#### 출력 형식

```json
{
  "audio_descriptions": [
    {
      "id": 1,
      "start_time": "0:03.5",
      "end_time": "0:06.1",
      "duration_sec": 2.6,
      "description": "남자가 커피잔을 들어 한 모금 마신다.",
      "allowed_chars": 18,
      "current_chars_after": 17,
      "compressed_by_chars": true,
      "exceeds_limit": false
    }
  ]
}
```

#### 에러 처리

- **Gemini 재시도**: 3회 재시도, 5초 딜레이
- **MAX_TOKENS 감지**: 토큰 소진 시 경고 로그
- **JSON 복구**: `repair_json()`으로 잘린 JSON 자동 복구
- **파일 정리**: 처리 완료 후 업로드 파일 자동 삭제

### 5.4 GPT 모델 (`extract_for_gpt.py` + `get_AD_gpt.py`)

**2단계 파이프라인 (프레임 추출 + Two-Pass AD 생성)**:

#### Step 1: 데이터 추출 (`extract_for_gpt.py`)

```
┌─────────────────────────────────────────────────────────────┐
│                   GPT 전처리 파이프라인                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. 오디오 추출 (FFmpeg)                                      │
│    - 16kHz 모노 WAV로 변환                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. VAD 분석 (Silero VAD)                                    │
│    - 발화 구간 탐지 (speech_timestamps)                      │
│    - 무음 구간 계산 (min_silence_duration: 2.5초)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. STT (OpenAI Whisper)                                     │
│    - 발화 구간별 텍스트 인식                                  │
│    - 모델: tiny/base/small/medium/large                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. 프레임 추출 (FFmpeg)                                      │
│    - 2 FPS로 프레임 추출                                     │
│    - 파일명: frame_0001_0.00s.jpg                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
         출력: {video_name}_data.json + frames/
```

#### Step 2: AD 생성 (`get_AD_gpt.py`)

**Two-Pass 방식으로 고품질 AD 생성**:

```
┌─────────────────────────────────────────────────────────────┐
│                    1st Pass: 맥락 파악                       │
├─────────────────────────────────────────────────────────────┤
│ 입력: 15개 균등 샘플링 키프레임 + 전체 대본                    │
│                                                             │
│ 분석 항목:                                                   │
│  • known_content: 알려진 영화/드라마 여부                     │
│  • location: 장소/배경                                       │
│  • characters: 등장인물 (남자1, 여자1 형식)                   │
│  • situation: 상황 요약                                      │
│  • mood: 분위기                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              긴 무음구간 분할 (Scene Split)                   │
├─────────────────────────────────────────────────────────────┤
│ • 16초 미만: 분할 없음                                       │
│ • 16초 이상: 8초 단위 분할                                   │
│   예: 23초 → [8초, 15초]                                     │
│       30초 → [8초, 8초, 14초]                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    2nd Pass: AD 생성                         │
├─────────────────────────────────────────────────────────────┤
│ 각 무음 구간에 대해:                                         │
│  • 해당 구간 프레임 (최대 10장) + 맥락 정보 전달              │
│  • 이전 AD 텍스트 전달 → 중복 방지                           │
│  • 전후 대사 컨텍스트 포함                                   │
│                                                             │
│ 규칙:                                                        │
│  • duration 내 읽을 수 있는 길이                             │
│  • 현재 시제, 객관적 묘사                                    │
│  • 인물: 이름 X → "남자1", "여자1" 형식                      │
└─────────────────────────────────────────────────────────────┘
```

**GPT 모델 설정**:
```python
GPT_MODEL = "gpt-4o"
max_tokens = 200 (AD), 500 (Context)
temperature = 0.3
image_detail = "low"  # 토큰 절약
```

### 5.5 Cookie 모델 (로컬 Inference 파이프라인)

**Qwen-VL + Llama-70B 조합의 로컬 GPU 파이프라인**

#### 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                   inference/ 폴더 구조                       │
├─────────────────────────────────────────────────────────────┤
│ main.py              ← 메인 파이프라인 엔트리포인트           │
│ config.py            ← 설정 (VIDEO_PATH, MODEL_ID 등)        │
│ inference_server.py  ← FastAPI HTTP 서버 (포트 8001)         │
│                                                             │
│ audio/               ← 오디오 처리 (Whisper)                 │
│ vision/              ← Qwen-VL 모델 및 프레임 추출           │
│ chunking/            ← 씬 기반 청크 분할                     │
│ language/            ← Llama-70B 서사 생성                   │
│ utils/               ← GPU 메모리 관리 등                    │
│ output/              ← 결과 JSON 저장                        │
└─────────────────────────────────────────────────────────────┘
```

#### 처리 플로우

```
┌─────────────────────────────────────────────────────────────┐
│                    Part A: 시각/청각 분석                    │
│                    (Qwen-VL + Whisper)                       │
└─────────────────────────────────────────────────────────────┘
        │
        ├──► 1. 오디오 추출 + Whisper STT
        │       └→ dialogues 리스트 생성
        │
        ├──► 2. 씬 감지 (SigLIP)
        │       └→ SCENE_THRESHOLD = 0.28
        │
        ├──► 3. 청크 생성
        │       └→ MIN_CHUNK: 2초, MAX_CHUNK: 8초
        │
        ├──► 4. Qwen-VL 모델 로드
        │       └→ unsloth/Qwen3-VL-8B-Instruct
        │
        └──► 5. 2-Stage Qwen-VL 추론
                │
                ├── Stage 1: 인물 식별
                │   └→ "Person 1: Body/Face, Attire..."
                │
                └── Stage 2-A: 행동 분석 (인물 있을 때)
                    Stage 2-B: 배경 분석 (인물 없을 때)

                출력: final_ad_creation_output.json

        ⬇ GPU 메모리 해제 (Qwen 언로드)

┌─────────────────────────────────────────────────────────────┐
│                    Part B: 서사 생성                         │
│                    (Llama-3.1-70B)                           │
└─────────────────────────────────────────────────────────────┘
        │
        ├──► [Stage 1] 서사 생성 (Narrative Generation)
        │       └→ 청크별 AD 텍스트 초안 생성
        │       └→ stage1_narrative_output.json
        │
        ├──► [Stage 2] 기억 추적 (Memory Tracker)
        │       └→ 인물 일관성 유지 (동일인물 추적)
        │       └→ stage2_final_output.json
        │
        └──► [Stage 3] 서사 다듬기 (Refinement)
                └→ 중복 제거, 길이 조절
                └→ stage3_final_narrative_refined.json

        ⬇ GPU 메모리 해제 (Llama 언로드)

                최종 출력: stage3_final_narrative_refined.json
```

#### HTTP 서버 API (`inference_server.py`)

```http
# 서버 시작
cd /mnt/Ko-AD/inference
source venv/bin/activate
uvicorn inference_server:app --host 0.0.0.0 --port 8001
```

**엔드포인트**:
```http
GET  /health              # GPU 상태 확인
POST /generate-ad         # AD 생성 요청

# 요청 예시
{
  "video_path": "/path/to/video.mp4",
  "output_dir": "/path/to/output",
  "lang": "ko",          // "ko" 또는 "en"
  "video_id": "uuid-xxx"
}
```

**모델 설정**:
```python
# config.py
QWEN_MODEL = "unsloth/Qwen3-VL-8B-Instruct"    # 비전 모델
LLAMA_MODEL = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"  # 언어 모델
SIGLIP_MODEL = "google/siglip-large-patch16-384"  # 씬 감지

# 파라미터
SCENE_THRESHOLD = 0.28    # 씬 변화 감지 임계값
MIN_CHUNK_LEN = 2.0       # 최소 청크 길이 (초)
MAX_CHUNK_LEN = 8.0       # 최대 청크 길이 (초)
```

**요구 사항**:
- NVIDIA GPU (VRAM 24GB+ 권장, RTX 4090 등)
- CUDA 12.x
- 별도 venv 환경 (`inference/venv/`)

---

## 6. TTS 시스템

### 6.1 지원 음성 프로필

| 프로필 | 엔진 | 언어 | 설명 |
|--------|------|------|------|
| `gtts` | Google TTS | ko/en | 기본 (무료) |
| `kor_male` | Coqui XTTS v2 | ko | 한국어 남성 음성 |
| `kor_female` | Coqui XTTS v2 | ko | 한국어 여성 음성 |
| `eng_male` | Coqui XTTS v2 | en | 영어 남성 음성 |
| `eng_female` | Coqui XTTS v2 | en | 영어 여성 음성 |
| `gemini_kor_male` | Gemini TTS | ko | Puck 음성 |
| `gemini_kor_female` | Gemini TTS | ko | Kore 음성 |
| `gemini_eng_male` | Gemini TTS | en | Charon 음성 |
| `gemini_eng_female` | Gemini TTS | en | Aoede 음성 |

### 6.2 VAD 기반 더킹

**Silero VAD를 사용한 스마트 볼륨 조절**:
```python
# 더킹 하이퍼파라미터
DIALOGUE_LUFS_MARGIN = 2.0      # AD를 대사보다 2 LUFS 작게
AD_OVERLAP_EXTRA_ATTENUATION_DB = -3.0  # 겹침 시 추가 -3dB
AD_OVERFLOW_ATTENUATION_DB = -6.0       # overflow 시 -6dB
```

**처리 흐름**:
1. 원본 오디오에서 VAD로 대사 구간 감지
2. 대사 구간의 LUFS 측정
3. AD 오디오를 목표 LUFS로 정규화
4. 겹치는 구간 추가 감쇄
5. overflow 구간 볼륨 감소
6. 다음 AD와의 나레이션 겹침 방지

### 6.3 Gemini TTS

```python
# 화면해설 최적화 프롬프트
GEMINI_TTS_AD_PROMPT_KO = """다음 화면해설을 차분하고 명확한 톤으로 빠르게 읽어주세요.
- 전문적인 다큐멘터리 나레이터처럼 중립적이고 객관적인 어조로 말해주세요.
[extremely fast] """
```

---

## 7. 전체 처리 흐름 (Flow)

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

[2] AD 생성 (화면해설 텍스트)
    ┌─────────┐     POST /api/generate-ad  ┌─────────┐     spawn      ┌─────────────┐
    │ Browser │ ─────────────────────────▶ │ Backend │ ─────────────▶ │   Python    │
    │ (React) │                            │ (Node)  │                │get_AD_*.py  │
    └─────────┘                            └────┬────┘                └──────┬──────┘
         ▲                                      │                            │
         │                                      │                            ▼
         │                                      │                    Gemini/GPT API
         │                                      │                            │
         │                                      │                            ▼
         └──────────────────────────────────────┴── storage/ad_json/{uuid}_{lang}.ad.json

[3] TTS 변환 (음성 합성 + 믹싱)
    ┌─────────┐     POST /api/generate-tts ┌─────────┐     spawn      ┌─────────────────┐
    │ Browser │ ─────────────────────────▶ │ Backend │ ─────────────▶ │     Python      │
    │ (React) │                            │ (Node)  │                │gemini_json_tts  │
    └─────────┘                            └────┬────┘                └────────┬────────┘
                                                │                              │
                                                │    1. 원본 오디오 추출 (WAV)    │
                                                │    2. VAD 분석 (더킹용)        │
                                                │    3. TTS 생성 (gTTS/Coqui/Gemini)
                                                │    4. LUFS 정규화 + 더킹       │
                                                │    5. 오디오 믹싱              │
                                                │    6. FFmpeg로 비디오 결합     │
                                                │                              ▼
                                                └─────── storage/tts/{uuid}_ad_mix.mp4

[4] 내보내기 (Export)
    ┌─────────┐   POST /api/export-with-ad ┌─────────┐
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

## 8. API 엔드포인트

### 8.1 헬스 체크
```http
GET /api/health
```
**응답**: `{ "ok": true, "timestamp": 1234567890 }`

### 8.2 비디오 업로드
```http
POST /api/upload
Content-Type: multipart/form-data

Body: video (file, mp4 only, max 10GB)
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

### 8.3 YouTube 영상 다운로드
```http
POST /api/upload-youtube
Content-Type: application/json

Body: { "url": "https://www.youtube.com/watch?v=xxxxx" }
```

### 8.4 화면해설 생성
```http
POST /api/generate-ad
Content-Type: application/json
Authorization: Bearer <token> (optional, for credit deduction)

Body: {
  "video_id": "uuid-xxxx",
  "server_path": "/app/server/storage/uploads/uuid-xxxx.mp4",
  "lang": "ko",              // "ko" 또는 "en"
  "model": "gemini"          // "gemini" | "jack" | "gpt" | "inference"
}
```
**응답**:
```json
{
  "status": "generated",     // "generated" | "cached"
  "segments": [
    { "id": 1, "start": 3.5, "end": 6.1, "text": "남자1이 커피를 마신다." }
  ]
}
```

### 8.5 TTS 변환
```http
POST /api/generate-tts
Content-Type: application/json

Body: {
  "videoId": "uuid-xxxx",
  "lang": "ko",
  "adSegments": [...],           // 편집된 세그먼트 (선택사항)
  "voiceProfile": "gtts",        // "gtts" | "kor_male" | "gemini_kor_female" 등
  "enableDucking": true,
  "geminiApiKey": null           // Gemini TTS용 (선택사항)
}
```
**응답**:
```json
{
  "status": "ok",
  "videoId": "uuid-xxxx",
  "adVideoUrl": "/static/tts/uuid-xxxx_ad_mix.mp4",
  "adAudioUrl": "/static/tts/uuid-xxxx_ad_mix.wav",
  "segmentClips": [...],
  "meta": {
    "usedSegments": 5,
    "cutSegments": 1
  }
}
```

### 8.6 영상 내보내기
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
  "downloadUrl": "/static/exports/uuid-xxxx_ad_export_2025-12-06.mp4",
  "fileName": "uuid-xxxx_ad_export_2025-12-06.mp4"
}
```

### 8.7 인증 API
```http
POST /api/auth/signup     # 회원가입
POST /api/auth/login      # 로그인
POST /api/auth/logout     # 로그아웃
GET  /api/auth/me         # 현재 사용자 정보
GET  /api/auth/credits    # 크레딧 조회
GET  /api/auth/transactions  # 거래 내역
```

### 8.8 평가 API
```http
GET    /api/ratings/:videoId                    # 평가 조회
POST   /api/ratings/:videoId                    # 평가 저장
PATCH  /api/ratings/:videoId/segment/:segmentId # 세그먼트 평가 업데이트
POST   /api/ratings/:videoId/apply-edits        # 편집 적용 시 자동 평가
```

---

## 9. 로컬 개발 환경 설정

### 9.1 필수 요구사항
- **Node.js 20+**
- **Python 3.11+** (3.13은 audioop-lts 설치 필요)
- **FFmpeg** (시스템에 설치되어 PATH에 포함)
- **Google Gemini API Key**
- **(옵션)** NVIDIA GPU + CUDA (Coqui TTS, Inference 모델)

### 9.2 설치 및 실행

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

# Python 3.13+인 경우
pip install audioop-lts
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
npm run dev    # http://localhost:4000 또는 4001

# 터미널 2: 프론트엔드
cd Cursor/web-ads-app
npm run dev    # http://localhost:5173
```

브라우저에서 `http://localhost:5173`으로 접속합니다.
Vite 개발 서버가 `/api/*` 요청을 백엔드로 자동 프록시합니다.

---

## 10. Docker를 이용한 배포

### 10.1 Docker의 역할

| 역할 | 설명 |
|------|------|
| **환경 패키징** | Node.js, Python, FFmpeg, Silero VAD 등 모든 의존성을 하나의 이미지로 패키징 |
| **프로덕션 빌드** | 프론트엔드를 빌드하고 Nginx로 서빙 |
| **네트워크 구성** | 프론트엔드와 백엔드 간 내부 통신 설정 |
| **리버스 프록시** | Nginx가 `/api/*`, `/static/*` 요청을 백엔드로 전달 |

### 10.2 컨테이너 구성

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Network                        │
│  ┌─────────────────┐      ┌─────────────────────────┐   │
│  │   frontend      │      │       backend           │   │
│  │   (Nginx)       │─────▶│   (Node.js + Python)    │   │
│  │   Port: 80      │      │   Port: 4000            │   │
│  └─────────────────┘      └─────────────────────────┘   │
│         │                           │                    │
│         ▼                           ▼                    │
│    정적 파일 서빙              API 처리 + 파일 저장        │
│    /api/* 프록시              Python 스크립트 실행        │
│    /static/* 프록시           Silero VAD 모델 로드        │
└─────────────────────────────────────────────────────────┘
```

#### Frontend 컨테이너 (`ad_frontend`)
- **베이스 이미지**: `nginx:alpine`
- **포트**: 80
- **역할**: 
  - React 앱의 정적 파일 서빙
  - `/api/*`, `/static/*`, `/files/*` 요청을 백엔드로 프록시
  - 1시간 타임아웃 (긴 AD/TTS 작업 지원)

#### Backend 컨테이너 (`ad_backend`)
- **베이스 이미지**: `node:20-bullseye`
- **추가 설치**: Python 3, FFmpeg, libsndfile
- **포트**: 4000
- **역할**:
  - REST API 서버
  - Python 스크립트 실행 (AD 생성, TTS)
  - 파일 저장소 관리
  - Silero VAD 모델 사전 로드

### 10.3 로컬에서 Docker 실행

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

### 10.4 AWS EC2 배포

자세한 EC2 배포 가이드는 별도 문서를 참조하세요.

**간략한 순서:**
1. EC2 인스턴스에 SSH 접속
2. Docker 및 Docker Compose 설치
3. 프로젝트 클론 (`git clone`)
4. 환경 변수 설정 (`.env`)
5. `docker compose up -d --build`
6. 보안 그룹에서 80번 포트 개방

---

## 11. 환경 변수

### `.env` 파일 내용

```properties
# Google Gemini API Key (필수)
# AD 생성 및 Gemini TTS에 사용됩니다.
GEMINI_API_KEY=your_api_key_here

# Python 실행 파일 경로 (Docker에서는 python3)
PYTHON_TTS_EXECUTABLE=python3

# CORS 허용 Origin
# 개발: http://localhost:5173
# 프로덕션: * 또는 특정 도메인
CLIENT_ORIGIN=*

# 서버 포트
PORT=4000

# 노드 환경
NODE_ENV=production

# Supabase 설정 (인증 시스템 사용 시)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key

# Inference 서버 설정 (옵션)
INFERENCE_SERVER_URL=http://host.docker.internal:8001
HOST_STORAGE_BASE=/mnt/Ko-AD/Cursor/server/storage
HOST_AD_JSON_DIR=/mnt/Ko-AD/Cursor/server/storage/ad_json
```

### 주요 환경 변수 설명

| 변수 | 필수 | 설명 |
|------|------|------|
| `GEMINI_API_KEY` | ✅ | Google Gemini API 키 |
| `PYTHON_TTS_EXECUTABLE` | ❌ | Python 실행 파일 경로 (기본: `python3`) |
| `CLIENT_ORIGIN` | ❌ | CORS 허용 Origin (기본: `http://localhost:5173`) |
| `PORT` | ❌ | 백엔드 서버 포트 (기본: `4000`) |
| `NODE_ENV` | ❌ | 실행 환경 (`development` / `production`) |
| `SUPABASE_URL` | ❌ | Supabase 프로젝트 URL |
| `SUPABASE_ANON_KEY` | ❌ | Supabase 익명 키 |
| `SUPABASE_SERVICE_ROLE_KEY` | ❌ | Supabase 서비스 역할 키 |
| `INFERENCE_SERVER_URL` | ❌ | Inference 서버 URL (기본: `http://host.docker.internal:8001`) |

---

## 부록: 주요 파일 요약

| 파일 경로 | 설명 |
|-----------|------|
| `Cursor/server/src/index.js` | Express 서버 메인 파일. 모든 API 엔드포인트 정의 |
| `Cursor/server/src/authRoutes.js` | 인증 및 크레딧 API |
| `Cursor/python/get_AD_gemini.py` | Gemini 2.5 Pro 기반 AD 생성 |
| `Cursor/python/get_AD_jack.py` | Jack Method (Multi-stage Context Fusion) |
| `Cursor/python/gemini_json_tts.py` | TTS 합성, VAD 더킹, 오디오 믹싱 |
| `Cursor/python/run_inference.py` | 로컬 Inference 파이프라인 래퍼 |
| `Cursor/web-ads-app/src/components/VideoEditorLayout.tsx` | 메인 편집기 UI 컴포넌트 |
| `Cursor/web-ads-app/src/contexts/AuthContext.jsx` | 인증 상태 관리 |
| `docker-compose.yml` | Docker Compose 설정 |
| `Cursor/server/Dockerfile` | 백엔드 Docker 이미지 정의 |
| `Cursor/web-ads-app/Dockerfile` | 프론트엔드 Docker 이미지 정의 |
| `Cursor/web-ads-app/nginx.conf` | Nginx 리버스 프록시 설정 |

---

## 라이선스

이 프로젝트는 내부 개발용으로 작성되었습니다.

---

*마지막 업데이트: 2025-12-06*
