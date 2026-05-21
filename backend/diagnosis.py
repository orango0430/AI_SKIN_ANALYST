import os
import json
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from database import get_db
from models import DiagnosisResult, User
from auth import get_current_user
from skin_inference import get_engine

router = APIRouter(prefix="/diagnosis", tags=["diagnosis"])

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── 응답 스키마 ───────────────────────────────
class DetailItem(BaseModel):
    label: str
    value: int
    grade: str = ""   # "매우 양호" / "양호" / "보통" / "주의" / "심각"


class IssueItem(BaseModel):
    emoji: str
    title: str
    desc: str
    level: str
    levelClass: str  # 'high' | 'mid' | 'low'


class DiagnosisResponse(BaseModel):
    id: int
    score: int
    image_url: Optional[str] = None
    details: List[DetailItem]
    issues: List[IssueItem]
    summary: str
    created_at: str


class DiagnosisListItem(BaseModel):
    id: int
    score: int
    image_url: Optional[str] = None
    summary: str
    created_at: str


# 회귀 모델이 점수화하는 항목 — inference.aggregate_reg_scores 출력 5종
# (수분/탄력/주름(Ra)/모공/색소침착). 유분/민감도는 모델에 없어 제외.
ISSUE_TEMPLATE = {
    "moisture": ("💧", "수분", "부족", "수분 보유량이 평균보다 낮아요. 보습 강화가 필요합니다."),
    "R2":       ("✨", "탄력", "저하", "피부 탄력이 다소 떨어져 있어요."),
    "Ra":       ("〰️", "주름", "관찰", "잔주름 등 거칠기 신호가 보입니다."),
    "pore":     ("🔬", "모공", "확장", "모공이 두드러지게 보입니다. 각질 관리를 권장해요."),
    "count":    ("☀️", "색소침착", "UV 손상", "색소침착 부위가 관찰됩니다. 자외선 차단이 중요해요."),
}


def _infer_skin(image_path: str) -> dict:
    """실제 모델 추론. 실패 시 ValueError → 400으로 변환."""
    res = get_engine().analyze(image_path)
    agg = res["regression_aggregate"]   # 5항목 [{key, label_kr, score, grade, tier}, ...]

    # 프론트 호환: details = [{label, value, grade}]
    details = [
        {
            "label": item["label_kr"],
            "value": int(round(item["score"])),
            "grade": item.get("grade", ""),
        }
        for item in agg
    ]

    # 종합 점수: 5항목 균등 평균 (모델 신뢰도 차이 있으나 UI상 단일값 필요)
    score = int(round(sum(d["value"] for d in details) / max(1, len(details))))

    # issues: 항목 점수 < 65면 경고 카드 추가
    issues = []
    for item in agg:
        emoji_title_label_desc = ISSUE_TEMPLATE.get(item["key"])
        if not emoji_title_label_desc:
            continue
        emoji, title, level_label, desc = emoji_title_label_desc
        v = int(round(item["score"]))
        if v < 65:
            level_class = "high" if v < 55 else "mid"
            issues.append({
                "emoji": emoji,
                "title": f"{title} {level_label}",
                "desc": desc,
                "level": "주의" if level_class == "mid" else "관리필요",
                "levelClass": level_class,
            })
    if not issues:
        issues.append({
            "emoji": "🌿",
            "title": "전반적 양호",
            "desc": "특별히 두드러진 이슈가 없어요. 지금처럼 꾸준히 관리해 주세요.",
            "level": "양호",
            "levelClass": "low",
        })

    if score >= 80:
        summary = "전반적으로 건강한 피부 상태예요. 지금 루틴을 유지해 주세요."
    elif score >= 60:
        summary = "보통 수준의 피부 상태입니다. 약점 항목 위주로 케어를 강화해 보세요."
    else:
        summary = "관리가 필요한 항목이 있어요. 추천 가이드와 AI 상담을 활용해 보세요."

    return {
        "score": score,
        "details": details,
        "issues": issues,
        "summary": summary,
        # DB 저장용 — 추후 정책 변경/재집계 가능하도록 풀 결과 보관
        "raw": res,
    }


# ── 이미지 정적 제공 ──────────────────────────
@router.get("/image/{filename}")
def get_image(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")
    return FileResponse(path)


# ── 분석 실행 + 결과 저장 ─────────────────────
@router.post("/analyze", response_model=DiagnosisResponse)
def analyze(
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    ext = os.path.splitext(image.filename or "")[1].lower() or ".jpg"
    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
        ext = ".jpg"
    fname = f"{current_user.id}_{uuid.uuid4().hex}{ext}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    with open(fpath, "wb") as f:
        f.write(image.file.read())

    try:
        inferred = _infer_skin(fpath)
    except ValueError as e:
        # 얼굴 미감지/이미지 로드 실패 → 업로드 파일 삭제 후 400
        try:
            os.remove(fpath)
        except OSError:
            pass
        raise HTTPException(status_code=400, detail=str(e))

    record = DiagnosisResult(
        user_id=current_user.id,
        image_path=fname,
        score=inferred["score"],
        classification=json.dumps(inferred["issues"], ensure_ascii=False),
        regression=json.dumps(
            {
                "details": inferred["details"],
                "raw": inferred["raw"],  # 회귀 모델 원본 0~1 실수
                "summary": inferred["summary"],
            },
            ensure_ascii=False,
        ),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return DiagnosisResponse(
        id=record.id,
        score=record.score,
        image_url=f"/diagnosis/image/{fname}",
        details=inferred["details"],
        issues=inferred["issues"],
        summary=inferred["summary"],
        created_at=record.created_at.isoformat() if record.created_at else datetime.utcnow().isoformat(),
    )


# ── 내 진단 결과 목록 ─────────────────────────
@router.get("/list", response_model=List[DiagnosisListItem])
def list_diagnosis(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = (
        db.query(DiagnosisResult)
        .filter(DiagnosisResult.user_id == current_user.id)
        .order_by(DiagnosisResult.created_at.desc())
        .all()
    )
    items = []
    for r in rows:
        summary = ""
        try:
            summary = json.loads(r.regression or "{}").get("summary", "")
        except Exception:
            pass
        items.append(
            DiagnosisListItem(
                id=r.id,
                score=r.score or 0,
                image_url=f"/diagnosis/image/{r.image_path}" if r.image_path else None,
                summary=summary,
                created_at=r.created_at.isoformat() if r.created_at else "",
            )
        )
    return items


# ── 진단 결과 상세 ───────────────────────────
@router.get("/{diag_id}", response_model=DiagnosisResponse)
def get_diagnosis(
    diag_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = (
        db.query(DiagnosisResult)
        .filter(DiagnosisResult.id == diag_id, DiagnosisResult.user_id == current_user.id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다.")

    issues = []
    details = []
    summary = ""
    try:
        issues = json.loads(row.classification or "[]")
    except Exception:
        pass
    try:
        reg = json.loads(row.regression or "{}")
        details = reg.get("details", [])
        summary = reg.get("summary", "")
    except Exception:
        pass

    return DiagnosisResponse(
        id=row.id,
        score=row.score or 0,
        image_url=f"/diagnosis/image/{row.image_path}" if row.image_path else None,
        details=details,
        issues=issues,
        summary=summary,
        created_at=row.created_at.isoformat() if row.created_at else "",
    )


# ── 진단 결과 삭제 ───────────────────────────
@router.delete("/{diag_id}")
def delete_diagnosis(
    diag_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    row = (
        db.query(DiagnosisResult)
        .filter(DiagnosisResult.id == diag_id, DiagnosisResult.user_id == current_user.id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다.")

    if row.image_path:
        try:
            os.remove(os.path.join(UPLOAD_DIR, row.image_path))
        except OSError:
            pass

    db.delete(row)
    db.commit()
    return {"message": "삭제되었습니다."}
