"""백엔드↔inference.py 어댑터.

inference.py의 분류/회귀 추론 함수를 그대로 호출하되, 모델은 프로세스당 1회만
로드(싱글톤). FastAPI startup에서 prewarm 호출 권장.

반환 스키마는 프론트(ScoreVisualization)가 기대하는 details/issues 형태로
매핑하기 쉬운 dict.
"""
import os
import sys
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AI_DIR = os.path.join(PROJECT_ROOT, "ai")
if AI_DIR not in sys.path:
    sys.path.insert(0, AI_DIR)

import cv2  # noqa: E402
import torch  # noqa: E402

from inference import (  # noqa: E402
    extract_faceparts,
    load_models,
    infer,
    infer_regression,
    aggregate_reg_scores,
    load_calibration,
    REG_LABEL_KR,
    _kr_label,
    _grade_label,
)


# 체크포인트/캘리브 경로 (프로젝트 루트 기준)
CLS_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoint2", "class", "100%_augw", "1,2,3")
REG_CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoint2", "regression", "100%", "_mp")
CALIBRATION_PATH = os.path.join(PROJECT_ROOT, "reg_calibration_mp.json")

# 회귀 항목 신뢰도 티어 (test split Pearson r)
RELIABILITY_TIER = {
    "count": "정밀",   # pigmentation r=0.94
    "pore":  "정밀",   # r=0.73~0.82
    "Ra":    "참고",   # perocular wrinkle r=0.47~0.72
    "moisture": "참고",
    "R2":    "참고",
}


class SkinEngine:
    """분류/회귀 모델 + 캘리브를 메모리에 보관. 인스턴스 1개."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SkinEngine] device={self.device}", flush=True)

        print(f"[SkinEngine] 분류 모델 로드: {CLS_CHECKPOINT}", flush=True)
        self.cls_models = load_models(CLS_CHECKPOINT, self.device, mode="class")
        if not self.cls_models:
            raise RuntimeError(f"분류 모델 로드 실패: {CLS_CHECKPOINT}")

        print(f"[SkinEngine] 회귀 모델 로드: {REG_CHECKPOINT}", flush=True)
        self.reg_models = load_models(REG_CHECKPOINT, self.device, mode="regression")
        if not self.reg_models:
            raise RuntimeError(f"회귀 모델 로드 실패: {REG_CHECKPOINT}")

        self.calibration = load_calibration(CALIBRATION_PATH)
        if self.calibration:
            print(f"[SkinEngine] 캘리브레이션 로드: "
                  f"{len(self.calibration)}항목", flush=True)
        else:
            print(f"[SkinEngine] 캘리브 없음 — raw 환산 fallback "
                  f"({CALIBRATION_PATH})", flush=True)

        # TTA: 학습 시 use_aug로 좌우 flip 적용했으므로 추론에서도 켬
        self.tta = True
        print("[SkinEngine] ready", flush=True)

    def analyze(self, image_path: str) -> dict:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지 로드 실패: {image_path}")

        patches, _, face_patch = extract_faceparts(image, draw_debug_path=None)
        if not patches:
            raise ValueError("얼굴 감지 실패 — 정면 얼굴이 잘 보이는 사진인지 확인")

        cls_res = infer(patches, self.cls_models, self.device, tta=self.tta)
        reg_res = infer_regression(
            patches, face_patch, self.reg_models, self.device,
            calibration=self.calibration, tta=self.tta,
        )
        reg_agg = aggregate_reg_scores(reg_res)

        # 분류: 부위별 등급(낮을수록 양호) — 화면용
        classification = [
            {
                "key": lbl,
                "label_kr": _kr_label(lbl),
                "grade": int(g),
                "max_grade": int(maxg),
                "top1": round(float(top1) * 100, 1),
                "off1": round(float(off1) * 100, 1),
            }
            for lbl, (g, maxg, top1, off1) in sorted(cls_res.items())
        ]

        # 회귀 부위별 raw + 점수
        regression_per_area = [
            {
                "key": lbl,
                "label_kr": _kr_label(lbl),
                "raw": round(float(raw), 4),
                "real": round(float(real), 2),
                "score": round(100 - float(sev), 1),
            }
            for lbl, (raw, real, sev) in sorted(reg_res.items())
        ]

        # 회귀 항목별 평균 (사용자 화면 메인 표시) — 5항목
        regression_aggregate = [
            {
                "key": item_class,
                "label_kr": REG_LABEL_KR.get(item_class, item_class),
                "score": round(100 - float(sev), 1),
                "grade": _grade_label(100 - float(sev)),
                "tier": RELIABILITY_TIER.get(item_class, "참고"),
            }
            for item_class, sev in reg_agg.items()
        ]

        return {
            "classification": classification,
            "regression_per_area": regression_per_area,
            "regression_aggregate": regression_aggregate,
        }


_engine: Optional[SkinEngine] = None


def get_engine() -> SkinEngine:
    global _engine
    if _engine is None:
        _engine = SkinEngine()
    return _engine


def prewarm():
    """FastAPI startup에서 호출. 첫 요청 30초 지연 방지."""
    get_engine()
