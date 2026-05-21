from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(String(50), nullable=False)
    email      = Column(String(100), unique=True, nullable=False, index=True)
    password   = Column(String(255), nullable=False)        # bcrypt 암호화
    created_at = Column(DateTime, server_default=func.now())

    # 관계 설정
    results      = relationship("DiagnosisResult", back_populates="user")
    chat_history = relationship("ChatHistory", back_populates="user")


class DiagnosisResult(Base):
    __tablename__ = "diagnosis_results"

    id             = Column(Integer, primary_key=True, index=True)
    user_id        = Column(Integer, ForeignKey("users.id"), nullable=False)
    image_path     = Column(String(255))                    # 저장된 이미지 경로
    score          = Column(Integer)                        # 종합 점수
    classification = Column(Text)                           # 분류 결과 JSON
    regression     = Column(Text)                           # 회귀 결과 JSON
    created_at     = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="results")


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    role       = Column(String(20), nullable=False)         # "user" or "assistant"
    message    = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="chat_history")
