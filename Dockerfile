# SkinAI 백엔드 (FastAPI + AI 추론) — HuggingFace Spaces / Railway 등 Docker 호스팅용
#
# 빌드 컨텍스트 = 프로젝트 루트 (이 파일 위치 기준).
# 포함: ai/ (모델·전처리 코드), backend/ (FastAPI), checkpoint2/ (가중치),
#       reg_calibration_mp.json (회귀 점수 환산).
#
# 환경변수 (호스팅 플랫폼 secrets에 설정):
#   DATABASE_URL    = mysql+pymysql://... 또는 sqlite:///./skinai.db
#   GEMINI_API_KEY  = ...
#   JWT_SECRET_KEY  = ...
#   CORS_ORIGINS    = https://your-app.vercel.app (콤마 구분)
#   PORT            = 7860 (HF Spaces 기본) / Railway는 자동 주입

FROM python:3.9-slim

# OS 패키지 — mediapipe / opencv가 필요로 함
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 의존성 — 코드 변경에 영향 안 받게 먼저 설치 (Docker 캐시 활용)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r backend/requirements.txt

# 앱 코드 + 모델 가중치 + 캘리브 (이미지에 베이크)
COPY ai/ ./ai/
COPY backend/ ./backend/
COPY checkpoint2/ ./checkpoint2/
COPY reg_calibration_mp.json ./

# FastAPI 포트 — HF Spaces 기본 7860, Railway는 $PORT 자동 주입
ENV PORT=7860
EXPOSE 7860

# uvicorn 진입 (backend/ 안에서 실행, imports auth.py 등 그대로 작동)
WORKDIR /app/backend
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]
