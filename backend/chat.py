import os
import json
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from dotenv import load_dotenv

from database import get_db
from models import ChatHistory, DiagnosisResult, User
from auth import get_current_user

load_dotenv()

router = APIRouter(prefix="/chat", tags=["chat"])

# Gemini 클라이언트 설정
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 피부 상담 시스템 프롬프트
SYSTEM_PROMPT = """
당신은 SkinAI의 전문 피부 상담사입니다.
사용자의 피부 고민을 듣고 전문적이고 친절하게 상담해주세요.

상담 가능한 분야:
- 피부 타입 (건성, 지성, 복합성, 민감성)
- 트러블 관리 (여드름, 블랙헤드, 화이트헤드)
- 수분 및 유분 관리
- 색소침착, 기미, 주근깨
- 모공 관리
- 피부 탄력 및 주름
- 스킨케어 루틴 및 성분 추천
- 선크림 선택

주의사항:
- 의학적 진단은 하지 않습니다.
- 심각한 피부 질환은 피부과 전문의 상담을 권유합니다.
- 답변은 한국어로 해주세요.
- 친절하고 공감적인 톤을 유지해주세요.
"""


def build_diagnosis_context(db: Session, user_id: int) -> Optional[str]:
    """사용자의 최신 진단 결과를 시스템 프롬프트에 주입할 텍스트로 변환.

    분석 기록이 없으면 None. 있으면 종합점수/항목별 점수/주의항목/요약을
    Gemini가 참고하도록 한국어 컨텍스트로 구성.
    """
    row = (
        db.query(DiagnosisResult)
        .filter(DiagnosisResult.user_id == user_id)
        .order_by(DiagnosisResult.created_at.desc())
        .first()
    )
    if not row:
        return None

    # regression 컬럼 = {"details": [{label, value, grade}], "summary": ...}
    details, summary = [], ""
    try:
        reg = json.loads(row.regression or "{}")
        details = reg.get("details", []) or []
        summary = reg.get("summary", "") or ""
    except (ValueError, TypeError):
        pass

    # classification 컬럼 = issues 카드 [{title, desc, ...}]
    issues = []
    try:
        issues = json.loads(row.classification or "[]") or []
    except (ValueError, TypeError):
        pass

    lines = ["[사용자의 최근 피부 분석 결과 — 이 데이터를 근거로 맞춤 상담하세요]"]
    when = row.created_at.strftime("%Y-%m-%d") if row.created_at else ""
    lines.append(f"분석일: {when}    종합 점수: {row.score or 0}/100 (높을수록 양호)")

    if details:
        lines.append("항목별 점수:")
        for d in details:
            label = d.get("label", "")
            value = d.get("value", "")
            grade = d.get("grade", "")
            lines.append(f"  - {label}: {value}점 ({grade})")

    issue_titles = [i.get("title", "") for i in issues
                    if i.get("levelClass") in ("high", "mid")]
    if issue_titles:
        lines.append("주의가 필요한 항목: " + ", ".join(issue_titles))

    if summary:
        lines.append(f"종합 요약: {summary}")

    lines.append(
        "\n위 결과에서 약한 항목을 우선으로, 점수가 낮은 부분에 맞춰 구체적으로 조언하세요. "
        "사용자가 점수/항목을 직접 언급하지 않아도 이 결과를 자연스럽게 반영하세요."
    )
    return "\n".join(lines)


# ── 요청/응답 스키마 ──────────────────────────
class MessageRequest(BaseModel):
    message: str

class MessageResponse(BaseModel):
    role: str
    message: str

class ChatHistoryResponse(BaseModel):
    id: int
    role: str
    message: str

# ── 채팅 ─────────────────────────────────────
@router.post("/message", response_model=MessageResponse)
def send_message(
    req: MessageRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    # 이전 대화 내역 불러오기 (최근 10개)
    history = db.query(ChatHistory)\
        .filter(ChatHistory.user_id == current_user.id)\
        .order_by(ChatHistory.created_at.desc())\
        .limit(10).all()
    history.reverse()

    # Gemini 메시지 구성
    contents = []
    for h in history:
        role = "user" if h.role == "user" else "model"
        contents.append({"role": role, "parts": [{"text": h.message}]})
    contents.append({"role": "user", "parts": [{"text": req.message}]})

    # 최신 진단 결과를 시스템 프롬프트에 주입 → 분석 결과 기반 맞춤 상담
    diag_context = build_diagnosis_context(db, current_user.id)
    system_instruction = (
        f"{SYSTEM_PROMPT}\n\n{diag_context}" if diag_context else SYSTEM_PROMPT
    )

    # Gemini API 호출
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config={"system_instruction": system_instruction},
            contents=contents,
        )
        answer = response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API 오류: {str(e)}")

    # 대화 내역 DB 저장
    db.add(ChatHistory(user_id=current_user.id, role="user", message=req.message))
    db.add(ChatHistory(user_id=current_user.id, role="assistant", message=answer))
    db.commit()

    return MessageResponse(role="assistant", message=answer)


# ── 대화 내역 조회 ────────────────────────────
@router.get("/history", response_model=List[ChatHistoryResponse])
def get_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    history = db.query(ChatHistory)\
        .filter(ChatHistory.user_id == current_user.id)\
        .order_by(ChatHistory.created_at.asc())\
        .all()
    return [ChatHistoryResponse(id=h.id, role=h.role, message=h.message) for h in history]


# ── 대화 내역 초기화 ──────────────────────────
@router.delete("/history")
def clear_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    db.query(ChatHistory)\
        .filter(ChatHistory.user_id == current_user.id)\
        .delete()
    db.commit()
    return {"message": "대화 내역이 초기화되었습니다."}
