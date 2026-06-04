# Skin AI Analyst 🩺

> 얼굴 사진 1장으로 8개 부위 × 11개 항목의 피부 상태를 진단하는 AI 웹·앱 서비스

[![Frontend](https://img.shields.io/badge/Frontend-Vercel-black)](https://ai-skin-analyst.vercel.app)
[![Backend](https://img.shields.io/badge/Backend-HF%20Spaces-yellow)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8-ee4c2c)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688)]()
[![React](https://img.shields.io/badge/React-18-61dafb)]()

---

## 1. 프로젝트 개요

**Skin AI Analyst**는 사용자가 정면 얼굴 사진 1장을 업로드하면, AI가 얼굴을 부위별로
자동 분할하고 각 부위의 피부 상태를 **등급(분류) + 측정값(회귀)** 두 갈래로 진단해
**0~100 건강 점수**로 환산해 보여주는 서비스입니다.

진단 결과는 사용자별로 저장되어 추이를 확인할 수 있고, **Gemini 기반 AI 피부 상담 챗봇**이
결과에 대한 후속 상담을 제공합니다. 웹(PWA)과 안드로이드 앱(TWA APK) 양쪽으로 동작합니다.

### 핵심 기능

| 기능 | 설명 |
|---|---|
| 🔍 **AI 피부 진단** | 사진 1장 → MediaPipe 얼굴 분할 → 부위별 ResNet50 추론 → 0~100 점수 |
| 📊 **결과 시각화** | 종합 점수 링, 항목별 막대, 레이더 차트, 경고 카드 |
| 💬 **AI 상담 챗봇** | Gemini 2.5 Flash 기반, 진단 결과 맥락 피부 상담 |
| 👤 **마이페이지** | 진단 이력 저장·조회·삭제, 점수 추이 |
| 🔐 **회원 인증** | JWT + bcrypt 기반 회원가입/로그인 |
| 📱 **PWA + APK** | 설치형 웹앱(서비스워커 캐시) + Bubblewrap TWA 안드로이드 앱 |

### 한눈에 보는 진단 파이프라인

```
사진 업로드
   │
   ▼
[MediaPipe Face Mesh] 468 랜드마크 → 8개 facepart 패치 + 전체 얼굴 crop
   │
   ├─▶ [분류 모델 ×6] ResNet50  →  부위별 등급(0~N) ── 하이브리드(argmax/E-value) + TTA
   │
   └─▶ [회귀 모델 ×5] ResNet50  →  부위별 측정값(수분/탄력/주름/모공/색소)
                                    │
                                    ▼
                          [캘리브레이션] 학습 분포 percentile rank
                                    │
                                    ▼
                      항목별 0~100 건강 점수 + 종합 점수 + 경고 카드
```

---

## 2. 기술 스택

### 2.1 AI / 모델

| 구분 | 기술 | 비고 |
|---|---|---|
| 프레임워크 | **PyTorch 2.8** (CPU 빌드) | 추론 서버는 CPU — CUDA 휠 2GB+ 라 HF Spaces 빌드 회피 |
| 백본 | **ResNet50** (ImageNet pretrained) | `torchvision.models.resnet50(weights=DEFAULT)`, fc만 교체 |
| 얼굴 분할 | **MediaPipe Face Mesh 0.10.9** | 468 랜드마크 기반 facepart bbox (legacy `mp.solutions` 보존 버전 고정) |
| 영상 처리 | OpenCV (`opencv-python-headless`), Pillow, NumPy | |
| 추론 기법 | TTA(좌우 flip 평균), Hybrid argmax/Expected-value, Percentile 캘리브레이션 | |

### 2.2 백엔드

| 구분 | 기술 |
|---|---|
| 웹 프레임워크 | **FastAPI 0.128** + Uvicorn |
| ORM / DB | **SQLAlchemy 2.0** + PyMySQL → MySQL (로컬은 SQLite fallback 가능) |
| 인증 | **JWT** (`python-jose[cryptography]`) + **bcrypt** (`passlib`) |
| 검증 | Pydantic 2.12 + email-validator |
| 챗봇 | **Google Gemini 2.5 Flash** (`google-genai`) |
| 파일 | python-multipart (이미지 업로드) |
| 모델 배포 | huggingface_hub (`snapshot_download`로 체크포인트 다운로드) |

### 2.3 프론트엔드

| 구분 | 기술 |
|---|---|
| 라이브러리 | **React 18** + React Router 6 |
| 빌드 | Create React App (react-scripts 5) |
| 스타일 | Vanilla CSS (브랜드 그린 `#27ae60`), 반응형 `@media (max-width: 768px)` |
| 상태 | React Context (`AuthContext`) — JWT 토큰/유저 세션 |
| PWA | manifest.json + Service Worker (precache shell + network-first navigation) |

### 2.4 배포 / 인프라

| 대상 | 플랫폼 | 방식 |
|---|---|---|
| 프론트엔드 | **Vercel** | `wep/skinai` CRA 빌드 자동 배포 |
| 백엔드 | **HuggingFace Spaces** (Docker) | `Dockerfile` (python:3.9-slim), 포트 7860 |
| 모델 가중치 | **HF Model Hub** | 빌드 시 `snapshot_download` (Space repo 1GB LFS 한계 회피) |
| 안드로이드 앱 | **Bubblewrap TWA** | PWA URL을 Chrome 기반 APK로 래핑 (Digital Asset Links) |

---

## 3. 모델 아키텍처

### 3.1 부위(facepart) 정의

학습 데이터(AI Hub)의 사람이 라벨링한 bbox를 MediaPipe 468 랜드마크에 대응시켜
**해부학 랜드마크 그룹의 min/max + 여백**으로 부위 박스를 정의합니다. 비율 곱이 아니라
랜드마크 기반이므로 얼굴 크기·거리·각도와 무관하게 사진마다 동일한 영역을 잘라냅니다.

| ID | 부위 | ID | 부위 |
|---|---|---|---|
| 1 | 이마 (forehead) | 5 | 왼쪽 볼 (l_cheek) |
| 2 | 미간 (glabellus) | 6 | 오른쪽 볼 (r_cheek) |
| 3 | 왼쪽 눈가 (l_perocular) | 7 | 입술 (lip) |
| 4 | 오른쪽 눈가 (r_perocular) | 8 | 턱 (chin) |
| 0 | 전체 얼굴 (회귀 전용) | | |

> **모델 공유**: area 4(오른쪽 눈가)·6(오른쪽 볼)은 전용 모델 없이 area 3·5 모델을 공유합니다.
> (학습 코드의 str/int 비교 이슈로 좌우 flip이 실제로 적용된 적이 없어, 추론도 flip 없이 일치시킴)

### 3.2 분류 모델 (Multi-class, 6개)

각 부위마다 **ResNet50 1개**를 두고, 마지막 `fc`를 그 부위가 담당하는 진단 항목들의
등급 수 합으로 교체합니다. 출력 로짓을 항목별로 슬라이스해 softmax → 등급을 산출합니다.

- `MODEL_NUM_CLASS = [nan, 13, 7, 7, 0, 12, 0, 5, 6]` (area별 출력 차원)
- 진단 항목: pigmentation(색소), wrinkle/forehead_wrinkle(주름), pore(모공), dryness(건조), sagging(처짐)
- 손실: **항목별 class weight를 가진 CrossEntropy** (분포 불균형 → `label_distribution`의 balanced 가중치에 sqrt로 극단치 완화)
- 모델 선정: valid **±1 accuracy** 최대 기준 체크포인트 저장

**하이브리드 추론 정책** — argmax ±1 < 85%인 항목만 Expected-value(softmax 기대등급 반올림)로 예측:

| 항목 | 정책 | 효과(±1) |
|---|---|---|
| wrinkle / sagging / forehead_wrinkle / pore | Expected-value | 84.5→87.8 / 84.2→85.6 / 81.3→85.6 / 78.1→85.3 |
| pigmentation / dryness | argmax 유지 | 86.6 / 94.2 (E-value 시 소수등급 recall 붕괴) |

### 3.3 회귀 모델 (5개)

피부 측정 장비의 실측값(수분/탄력/모공 수/색소 개수/주름 거칠기)을 정규화해 회귀합니다.

- `REG_MODEL_NUM_CLASS = [1, 2, nan, 1, 0, 3, 0, nan, 2]`
- 항목: moisture(수분), R2(탄력 R²), Ra(주름 거칠기), pore(모공 수), count(색소 개수)
- 미간(2)·입(7)은 회귀 학습 데이터 자체가 없음
- 손실: **L1 Loss**, 모델 선정은 valid L1 최소 기준
- 역정규화(`REG_DENORM`): moisture ×100, Ra ×100μm, pore ×3000, count ×350, R2 ×1

**0~100 점수 환산 + 캘리브레이션**: 모델 raw 출력을 학습 valid 분포의 **percentile rank**로
변환해 도메인 shift를 보정합니다. 측정값이 높을수록 양호(moisture/R²)면 `severity=100-rank`,
낮을수록 양호(Ra/pore/count)면 `severity=rank`, 건강점수 `= 100 - severity`.

### 3.4 모델 성능 (test split)

| 회귀 항목 | Pearson r | 신뢰도 |
|---|---|---|
| pigmentation_count | 0.92 | 정밀 |
| l_cheek_pore | 0.82 | 정밀 |
| l_perocular_wrinkle_Ra | 0.72 | 참고 |
| l_cheek_elasticity_R2 | 0.70 | 참고 |
| l_cheek_moisture | 0.63 | 참고 |
| chin_moisture | 0.49 | 참고 |
| chin_elasticity_R2 | 0.39 | 한계 |

**핵심 발견 — 도메인 갭**: 학습 패치(bbox crop)와 추론 패치(MediaPipe crop) 간 도메인 갭이
회귀 추론 저하의 주범. 전량 MediaPipe 재추출 후 재학습으로 색소 외 항목 r 0~0.1 → 0.5~0.7 회복.
분류는 ±1 정확도 ≥ 85% 확보(가중치 조정 + augmentation + Expected-value 하이브리드).

---

## 4. 서버 아키텍처

### 4.1 전체 구성

```
┌─────────────────────────┐        ┌──────────────────────────────────────┐
│  Client                 │        │  Backend (HF Spaces · Docker · :7860)  │
│                         │        │                                        │
│  React PWA (Vercel)     │        │  FastAPI                               │
│  · Home / Analysis      │ HTTPS  │   ├─ /auth      회원가입·로그인 (JWT)   │
│  · Chat / MyPage        │◀──────▶│   ├─ /diagnosis 분석·이력·이미지         │
│  · ScoreVisualization   │  CORS  │   └─ /chat      Gemini 상담            │
│                         │        │                                        │
│  Android APK (TWA)      │        │  SkinEngine (싱글톤, prewarm)          │
│   = 같은 URL 래핑        │        │   분류모델×6 + 회귀모델×5 + 캘리브       │
└─────────────────────────┘        │                                        │
                                   │  SQLAlchemy → MySQL                     │
                                   │   users / diagnosis_results / chat      │
                                   └───────────────┬────────────────────────┘
                                                   │ snapshot_download (빌드 시)
                                                   ▼
                                       HF Model Hub (체크포인트)
```

### 4.2 백엔드 모듈 구성

| 파일 | 역할 |
|---|---|
| `backend/main.py` | FastAPI 진입점. lifespan에서 **DB 테이블 자동 생성** + **모델 prewarm**, CORS 설정, 라우터 등록 |
| `backend/auth.py` | 회원가입/로그인, JWT 발급·검증, `get_current_user` 의존성 |
| `backend/diagnosis.py` | `/diagnosis/analyze` 이미지 업로드→추론→DB 저장, 이력 조회·상세·삭제, 이미지 정적 제공 |
| `backend/chat.py` | `/chat/message` Gemini 2.5 Flash 호출(최근 10턴 컨텍스트), 대화 이력 저장·조회·초기화 |
| `backend/skin_inference.py` | **`SkinEngine` 싱글톤** — `ai/inference.py` 함수를 호출하되 모델은 프로세스당 1회 로드. 프론트 스키마로 결과 매핑 |
| `backend/models.py` | SQLAlchemy 모델 — `User` / `DiagnosisResult` / `ChatHistory` |
| `backend/database.py` | 엔진·세션·Base, `get_db` 의존성 |

### 4.3 추론 엔진 동작 (SkinEngine)

1. **prewarm** (서버 startup): 분류·회귀 모델 + 캘리브레이션을 메모리에 로드 → 첫 요청 30초 지연 제거
2. **analyze(image_path)**:
   - `extract_faceparts` — MediaPipe로 8개 패치 + 전체 얼굴 crop (얼굴 미감지 시 `ValueError` → 400)
   - `infer` — 분류 등급 (TTA + 하이브리드)
   - `infer_regression` + `aggregate_reg_scores` — 회귀 측정값 → 항목별 평균 점수
   - 프론트 호환 `classification` / `regression_per_area` / `regression_aggregate` dict 반환
3. `diagnosis.py`가 5개 항목 평균으로 **종합 점수** 산출, < 65 항목은 **경고 카드** 생성, 결과 JSON을 DB 저장

### 4.4 데이터 모델

| 테이블 | 주요 컬럼 |
|---|---|
| `users` | id, name, email(unique), password(bcrypt), created_at |
| `diagnosis_results` | id, user_id(FK), image_path, score, classification(JSON), regression(JSON), created_at |
| `chat_history` | id, user_id(FK), role(user/assistant), message, created_at |

### 4.5 주요 환경변수

| 변수 | 용도 |
|---|---|
| `DATABASE_URL` | `mysql+pymysql://...` 또는 `sqlite:///./skinai.db` |
| `JWT_SECRET_KEY` | JWT 서명 키 |
| `GEMINI_API_KEY` | Gemini 챗봇 |
| `CORS_ORIGINS` | 허용 오리진 (콤마 구분, 기본 `http://localhost:3000`) |
| `PORT` | 서버 포트 (HF Spaces 기본 7860) |
| `MODEL_REPO` / `MODEL_REVISION` | 빌드 타임 체크포인트 다운로드 대상 (Docker `ARG`) |

> 🔒 비밀값(`GEMINI_API_KEY`, DB 비밀번호, `JWT_SECRET_KEY`)은 절대 커밋하지 말고 호스팅 플랫폼 secrets로 주입하세요. `apk/android.keystore`도 커밋 금지.

---

## 5. 프로젝트 구조

```
NIA_019-028/
├── ai/                          # AI 모델 코드
│   ├── inference.py             #   분류·회귀 추론 통합 (CLI + 백엔드 공용)
│   ├── model.py                 #   학습 루프, 손실, class weight, 메트릭
│   ├── main.py                  #   분류 학습 진입점
│   ├── regression_train.py      #   회귀 재학습 (L1)
│   ├── data_loader.py / regression_data_loader.py
│   ├── extract_patches_mediapipe.py  # MediaPipe crop 패치 추출 (도메인 정합)
│   └── test.py
├── backend/                     # FastAPI 서버
│   ├── main.py auth.py chat.py diagnosis.py
│   ├── skin_inference.py models.py database.py
│   └── requirements.txt
├── wep/skinai/                  # React 프론트엔드 (Vercel)
│   ├── src/pages/               #   Home / Analysis / Chat / MyPage
│   ├── src/components/          #   Navbar / AuthModal / ScoreVisualization ...
│   ├── src/context/AuthContext.jsx
│   └── public/                  #   manifest.json / service-worker.js / .well-known
├── apk/                         # Bubblewrap TWA (gitignore — keystore 포함)
├── checkpoint2/                 # 모델 가중치 (HF Model Hub 미러)
├── reg_calibration_mp.json      # 회귀 점수 환산 캘리브레이션
├── Dockerfile                   # 백엔드 Docker (HF Spaces)
└── README.md / WIKI.md
```

---

## 6. 빠른 실행

### 백엔드
```bash
cd backend
./venvs/myapi/Scripts/python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```
서버 기동 시 모델 로드 30~60초 → `[SkinEngine] ready` 출력 후 요청 받음.

### 프론트엔드
```bash
cd wep/skinai
npm install
npm start          # http://localhost:3000
```
`.env.local`에 `REACT_APP_API_BASE_URL=http://localhost:8000` 지정.

### 추론 CLI 단독
```bash
cd ai
python inference.py \
  --image ../test_img.jpg \
  --checkpoint "../checkpoint2/class/100%_augw/1,2,3" \
  --reg_checkpoint "../checkpoint2/regression/100%/_mp" \
  --calibration ../reg_calibration_mp.json --tta
```

### Docker (백엔드)
```bash
docker build -t skinai-backend .
docker run -p 7860:7860 \
  -e GEMINI_API_KEY=... -e JWT_SECRET_KEY=... -e DATABASE_URL=... \
  -e CORS_ORIGINS=https://ai-skin-analyst.vercel.app \
  skinai-backend
```

---

## 7. 데이터셋

**AI Hub - 한국인 피부상태 측정 데이터** (NIA-019-028) 기반.
10~50대 1,100명 × 3장비 × 13각도 ≈ 13만장 안면 이미지.

데이터셋 문의: 단국대학교 컴퓨터학과 박사과정 이정호 (72210297@dankook.ac.kr)

---

## 8. 팀

**Team Skinmate** · 단국대학교 졸업작품 (2025)

자세한 내부 설계·학습 과정·한계·운영 가이드는 [WIKI.md](WIKI.md) 참고.
</content>
</invoke>
