import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from auth import router as auth_router
from chat import router as chat_router
from diagnosis import router as diagnosis_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 서버 시작 시 모델 사전 로드 (첫 요청 지연 제거)
    try:
        from skin_inference import prewarm
        prewarm()
    except Exception as e:
        print(f"[lifespan] prewarm 실패(요청 시 다시 시도): {e}")
    yield


app = FastAPI(title="SkinAI API", lifespan=lifespan)

# ── CORS — 환경변수 CORS_ORIGINS (콤마 구분). 로컬은 localhost:3000 fallback ──
_cors_env = os.getenv("CORS_ORIGINS", "http://localhost:3000")
allow_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 라우터 등록 ──
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(diagnosis_router)

@app.get("/")
def root():
    return {"message": "SkinAI API 서버 실행 중"}
