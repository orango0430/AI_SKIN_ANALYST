import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# DATABASE_URL은 호스팅 platform secrets에서 주입.
# 미설정 시 컨테이너 내부 SQLite로 fallback (개발/임시 데모용; 재배포 시 데이터 소실).
DATABASE_URL = os.getenv("DATABASE_URL") or "sqlite:///./skinai.db"

# SQLite는 멀티스레드 FastAPI에서 connect_args 필요. MySQL/PG는 기본값으로 OK.
_engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **_engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# DB 세션 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
