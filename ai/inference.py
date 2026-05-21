"""
임의의 얼굴 사진으로 학습된 모델 추론.

흐름:
  1) MediaPipe Face Mesh로 얼굴 랜드마크 감지
  2) 학습 데이터의 9개 facepart 영역과 유사한 bbox 계산
  3) 각 patch를 학습한 모델로 추론 → 등급
  4) 항목별 등급 → 0~100 점수 환산

사용:
  pip install mediapipe
  python inference.py --image my_photo.jpg
  python inference.py --image my_photo.jpg --debug_dir debug   # bbox 시각화 저장
"""
import argparse
import copy
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

try:
    import mediapipe as mp
except ImportError:
    print("MediaPipe가 없어요. 'pip install mediapipe' 후 다시 실행하세요.")
    sys.exit(1)

from data_loader import class_num_list
from model import diag_name


# ────────────────────────────────────────────────────────────
# Landmark-anchored facepart 정의.
#
# 학습 JSON bbox(사람이 라벨링한 영역)를 MediaPipe 468 landmark에 대응시켜
# 도출. 비율 곱이 아니라 '특정 해부학 landmark 그룹의 min/max + 여백'으로
# 박스를 정의 → 얼굴 크기·거리·각도와 무관하게 사진마다 동일한 영역.
#
# FACEPART_LANDMARKS: 그 부위 학습 bbox 안에 들어오던 landmark 인덱스들.
# FACEPART_PAD: (pad_x, pad_top, pad_bottom) — landmark bbox 한 변 길이 대비
#   확장 비율. 학습 bbox가 landmark 외곽보다 약간 넓어서 보정.
# ────────────────────────────────────────────────────────────
FACEPART_LANDMARKS = {
    # forehead: 이마 중앙~측면 landmark. 위(헤어라인)는 pad_top으로 크게 확장,
    # 아래는 눈썹(105,334)까지.
    1: [10, 67, 69, 108, 109, 151, 297, 299, 337, 338, 105, 334, 9],
    # glabellus: 안쪽 눈썹(55,285) ~ 코뼈 상단(9,8,168). 타이트하게.
    2: [9, 8, 168, 55, 285],
    # l_perocular(사진 왼쪽=피검자 오른쪽): 눈 바깥 측두부 잔주름대
    3: [31, 35, 46, 111, 113, 124, 143, 156, 225, 226, 127],
    # r_perocular: 3의 대칭
    4: [265, 276, 340, 342, 345, 353, 368, 372, 383, 446, 356],
    # l_cheek: 눈밑~코끝, 코옆~얼굴 가장자리 (큰 영역)
    5: [50, 36, 101, 118, 119, 120, 142, 205, 206, 117, 123, 147, 187, 207, 216],
    # r_cheek: 5의 대칭
    6: [280, 266, 330, 347, 348, 349, 371, 376, 425, 346, 352, 411, 427, 436],
    # lip: 입 둘레 + 약간
    7: [61, 291, 0, 17, 40, 270, 84, 314, 181, 405, 57, 287, 37, 267],
    # chin: 아래입술 밑 ~ 턱끝(152) ~ 턱선(좌우 172/397), 넓게.
    # 2026-05-21: 17/18/200 제거 + lip clamp 실험했으나 chin 회귀 r 개선
    # 미미(평균 Δ+0.005, elasticity 오히려 −0.02) → 원복. 모델 학습 시점의
    # 패치 정의와 inference crop 정의를 일치시켜 도메인갭 재발 방지.
    8: [152, 148, 377, 176, 400, 149, 378, 150, 379, 32, 262, 83, 313,
        18, 200, 172, 397, 136, 365, 169, 394, 17],
}

# (pad_x, pad_top, pad_bottom) : landmark bbox 변 길이 대비 확장 비율
FACEPART_PAD = {
    1: (0.02, 0.12, 0.02),   # forehead: landmark가 이미 헤어라인 근처라 top 소폭만
    2: (0.00, 0.40, 0.00),   # glabellus: 위로 눈썹까지만 (좌우/아래 그대로)
    3: (0.06, 0.12, 0.18),   # l_perocular: 측두부 과확장 방지
    4: (0.06, 0.12, 0.18),   # r_perocular
    5: (0.05, 0.02, 0.04),   # l_cheek (이미 IoU 0.66, 미세 조정)
    6: (0.05, 0.02, 0.04),   # r_cheek
    7: (0.06, 0.10, 0.15),   # lip
    8: (0.04, 0.30, 0.03),   # chin: 위(아래입술쪽)로 확장
}

# main.py와 동일
MODEL_NUM_CLASS = [np.nan, 13, 7, 7, 0, 12, 0, 5, 6]

# area별로 출력 차원이 어떤 라벨 키로 잘리는지
AREA_LABELS = {
    1: ["forehead_wrinkle", "forehead_pigmentation"],
    2: ["glabellus_wrinkle"],
    3: ["l_perocular_wrinkle"],
    4: ["r_perocular_wrinkle"],   # area 3 모델 + flip
    5: ["l_cheek_pigmentation", "l_cheek_pore"],
    6: ["r_cheek_pigmentation", "r_cheek_pore"],   # area 5 모델 + flip
    7: ["lip_dryness"],
    8: ["chin_sagging"],
}

# ── 회귀 모델 (regression_model_main.py + regression_data_loader.py 기준) ──
REG_MODEL_NUM_CLASS = [1, 2, np.nan, 1, 0, 3, 0, np.nan, 2]

# area별 회귀 출력 항목 (regression_data_loader.norm_reg type_class와 동일 순서)
# 미간(2)/입(7)은 회귀 학습 데이터 자체가 없음
REG_AREA_LABELS = {
    0: ["pigmentation_count"],
    1: ["forehead_moisture", "forehead_elasticity_R2"],
    3: ["l_perocular_wrinkle_Ra"],
    4: ["r_perocular_wrinkle_Ra"],                                       # area 3 + flip
    5: ["l_cheek_moisture", "l_cheek_elasticity_R2", "l_cheek_pore"],
    6: ["r_cheek_moisture", "r_cheek_elasticity_R2", "r_cheek_pore"],   # area 5 + flip
    8: ["chin_moisture", "chin_elasticity_R2"],
}

# norm_reg 정규화의 역변환: 모델 출력(≈0~1) × DENORM = 실측값
REG_DENORM = {
    "moisture": 100.0,   # 0~100 피부 수분량
    "R2":         1.0,   # 0~1 탄력 R² (raw)
    "Ra":       100.0,   # 0~100 거칠기 Ra (μm)
    "pore":    3000.0,   # 0~3000 모공 개수
    "count":    350.0,   # 0~350 색소침착 개수
}

# 0~100 점수 환산 방향: 모델 출력값(0~1)이 클수록 양호한지 / 작을수록 양호한지
SCORE_DIRECTION = {
    "moisture": "higher",   # 수분 많을수록 양호
    "R2":       "higher",   # 탄력 R² 1에 가까울수록 양호
    "Ra":       "lower",    # 거칠기 낮을수록 양호 (주름 적음)
    "pore":     "lower",    # 모공 적을수록 양호
    "count":    "lower",    # 색소침착 적을수록 양호
}

REG_LABEL_KR = {
    "moisture": "수분",
    "R2":       "탄력",
    "Ra":       "주름(Ra)",
    "pore":     "모공",
    "count":    "색소침착",
}

AREA_NAMES = {
    1: "forehead", 2: "glabellus", 3: "l_perocular", 4: "r_perocular",
    5: "l_cheek", 6: "r_cheek", 7: "lip", 8: "chin",
}

# 한글 부위명 (사용자 화면 표시용)
AREA_NAMES_KR = {
    1: "이마", 2: "미간", 3: "왼쪽 눈가", 4: "오른쪽 눈가",
    5: "왼쪽 볼", 6: "오른쪽 볼", 7: "입술", 8: "턱",
    0: "전체 얼굴",
}

# 진단 항목명 (분류·회귀 공용)
ITEM_NAMES_KR = {
    "wrinkle": "주름",
    "forehead_wrinkle": "이마 주름",
    "pigmentation": "색소침착",
    "pore": "모공",
    "dryness": "건조",
    "sagging": "처짐",
    # 회귀 측정 항목
    "moisture": "수분",
    "R2": "탄력",
    "Ra": "주름(거칠기)",
    "count": "색소침착",
}


def _kr_label(raw_label):
    """라벨 키 → 한글 부위+항목 표시.

    예시:
      chin_sagging              → "턱 처짐"
      forehead_wrinkle          → "이마 주름"
      l_cheek_pore              → "왼쪽 볼 모공"
      l_cheek_elasticity_R2     → "왼쪽 볼 탄력"
      l_perocular_wrinkle_Ra    → "왼쪽 눈가 주름(Ra)"
      pigmentation_count        → "전체 색소침착"
    """
    # area 키가 명확하지 않은 라벨은 명시적 매핑
    SPECIAL = {
        "pigmentation_count": "전체 색소침착",
    }
    if raw_label in SPECIAL:
        return SPECIAL[raw_label]

    parts = raw_label.split("_")
    # _R2 / _Ra처럼 두 토큰짜리 항목명은 area = 앞쪽 토큰 - 2개
    if parts[-1] in ("R2", "Ra"):
        item_key = parts[-1]
        area_key = "_".join(parts[:-2])
    else:
        item_key = parts[-1]
        area_key = "_".join(parts[:-1])

    area_kr_map = {
        "forehead": "이마", "glabellus": "미간",
        "l_perocular": "왼쪽 눈가", "r_perocular": "오른쪽 눈가",
        "l_cheek": "왼쪽 볼", "r_cheek": "오른쪽 볼",
        "lip": "입술", "chin": "턱",
    }
    area_kr = area_kr_map.get(area_key, area_key)
    item_kr = ITEM_NAMES_KR.get(item_key, item_key)
    return f"{area_kr} {item_kr}"


# ────────────────────────────────────────────────────────────
# 얼굴 분할
# ────────────────────────────────────────────────────────────
def extract_faceparts(image_bgr, draw_debug_path=None):
    """이미지에서 9개 facepart 패치 추출.

    face contour bbox 기준 상대 비율로 영역 분할 → 얼굴 형태/거리 무관 일관성 ↑.

    Returns:
        patches:    dict {facepart_id(1~8): patch_bgr_image}
        debug_img:  bbox 시각화 (or None)
        face_patch: 회귀 area 0(전체 얼굴)용 face bbox crop (or None)
    """
    h, w = image_bgr.shape[:2]

    # MediaPipe 처리용 축소본
    max_side = 1024
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        proc_img = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
        print(f"  처리용 축소: {h}x{w} → {proc_img.shape[0]}x{proc_img.shape[1]}")
    else:
        proc_img = image_bgr

    print("  Face Mesh 감지 중… (최대 10초)", flush=True)
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.3,
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(proc_img, cv2.COLOR_BGR2RGB))
    print("  감지 완료", flush=True)

    if not results.multi_face_landmarks:
        return {}, None, None

    landmarks = results.multi_face_landmarks[0].landmark
    # 모든 landmark를 원본 픽셀 좌표로 (468, 2)
    P = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

    # 전체 얼굴 bbox (회귀 area 0 / debug 참고용) — 외곽 landmark 기준
    fx1 = int(max(0, P[:, 0].min()))
    fx2 = int(min(w, P[:, 0].max()))
    fy1 = int(max(0, P[:, 1].min()))
    fy2 = int(min(h, P[:, 1].max()))
    face_patch = (
        image_bgr[fy1:fy2, fx1:fx2].copy()
        if (fy2 - fy1) > 8 and (fx2 - fx1) > 8 else None
    )

    patches = {}
    debug_img = image_bgr.copy() if draw_debug_path else None
    if debug_img is not None:
        cv2.rectangle(debug_img, (fx1, fy1), (fx2, fy2), (255, 100, 0), 2)

    for fid, idxs in FACEPART_LANDMARKS.items():
        pts = P[idxs]
        bx1, by1 = pts[:, 0].min(), pts[:, 1].min()
        bx2, by2 = pts[:, 0].max(), pts[:, 1].max()
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)

        pad_x, pad_top, pad_bot = FACEPART_PAD[fid]
        x1 = int(max(0, bx1 - bw * pad_x))
        x2 = int(min(w, bx2 + bw * pad_x))
        y1 = int(max(0, by1 - bh * pad_top))
        y2 = int(min(h, by2 + bh * pad_bot))

        if y2 - y1 < 8 or x2 - x1 < 8:
            continue

        patches[fid] = image_bgr[y1:y2, x1:x2].copy()

        if debug_img is not None:
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                debug_img, AREA_NAMES[fid], (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

    return patches, debug_img, face_patch


# ────────────────────────────────────────────────────────────
# 전처리
#   class      : data_loader.py make_double  → 중앙 letterbox, 항상 128×128
#   regression : regression_data_loader.py make_double  → corner 정렬,
#                작은 차원 < 64면 2x 확대 후 128×256 또는 256×128
#   area0(reg) : 전체 얼굴 → 128×128 직접 resize (no padding)
# ────────────────────────────────────────────────────────────
def preprocess_patch(patch_bgr, res=128, mode="class", is_area0=False):
    """patch BGR → 모델 입력 tensor."""
    if mode == "regression":
        # 128×128 고정 통일 (regression_train/regression_data_loader와 동일).
        # make_double/split-forward 폐기 — BN train/eval 일치 위함.
        out = cv2.resize(patch_bgr, (res, res))
        pil = Image.fromarray(out)
        return transforms.ToTensor()(pil).unsqueeze(0)

    h, w = patch_bgr.shape[:2]
    reduction = max(h, w) / res
    if reduction > 1.0:
        new_w = max(1, int(w / reduction))
        new_h = max(1, int(h / reduction))
        patch_bgr = cv2.resize(patch_bgr, (new_w, new_h))
        h, w = patch_bgr.shape[:2]

    if mode == "class":
        # 중앙 letterbox (data_loader.py make_double)
        out = np.zeros((res, res, 3), dtype=np.uint8)
        y0 = (res - h) // 2
        x0 = (res - w) // 2
        out[y0:y0 + h, x0:x0 + w] = patch_bgr
    else:
        # regression_data_loader.py make_double: corner 정렬 + 작은 차원 2x
        if h < 64:
            out = np.zeros((128, 256, 3), dtype=np.uint8)
            doubled = cv2.resize(patch_bgr, (w * 2, h * 2))
            out[:h * 2, :w * 2] = doubled
        elif w < 64:
            out = np.zeros((256, 128, 3), dtype=np.uint8)
            doubled = cv2.resize(patch_bgr, (w * 2, h * 2))
            out[:h * 2, :w * 2] = doubled
        else:
            out = np.zeros((128, 128, 3), dtype=np.uint8)
            out[:h, :w] = patch_bgr

    pil = Image.fromarray(out)
    return transforms.ToTensor()(pil).unsqueeze(0)


# ────────────────────────────────────────────────────────────
# 모델 로드
# ────────────────────────────────────────────────────────────
def load_models(checkpoint_root, device, mode="class"):
    """학습된 area별 모델 로드.

    Args:
        mode: "class" or "regression" — 출력 차원이 달라서 분기 필요

    Returns: dict {area_idx: model}
    """
    base = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    num_class_list = MODEL_NUM_CLASS if mode == "class" else REG_MODEL_NUM_CLASS
    model_list = {}

    for idx, num_class in enumerate(num_class_list):
        if isinstance(num_class, float) and np.isnan(num_class):
            continue
        if idx in (4, 6) or int(num_class) == 0:
            continue

        m = copy.deepcopy(base)
        m.fc = nn.Linear(m.fc.in_features, int(num_class))

        ckpt_path = os.path.join(checkpoint_root, str(idx), "state_dict.bin")
        if not os.path.isfile(ckpt_path):
            print(f"  [경고] 체크포인트 없음: {ckpt_path}")
            continue

        state = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(state["model_state"], strict=False)
        m.to(device).eval()
        model_list[idx] = m
        area_label = AREA_NAMES.get(idx, "all" if idx == 0 else f"area{idx}")
        print(f"  [OK] area {idx} ({area_label}) 로드")

    return model_list


# ────────────────────────────────────────────────────────────
# 추론
# ────────────────────────────────────────────────────────────
# 하이브리드 추론 정책 (test.py Final Report, 2026-05-17 augw 기준):
# argmax ±1 < 85%인 항목만 Expected-value(softmax 기대등급 반올림)로 예측 →
# 멀리 튀는 오차 완화로 ±1 ↑. 이미 ±1 ≥ 85%인 pigmentation/dryness는
# argmax 유지(E-value 시 소수등급 recall 붕괴, 특히 dryness Macro-F1 -13.6).
#   E-value 적용:  wrinkle 84.5→87.8 / sagging 84.2→85.6(F1 +3.6) /
#                  forehead_wrinkle 81.3→85.6 / pore 78.1→85.3
#   argmax 유지:   pigmentation 86.6 / dryness 94.2
# 주의: pore는 ±1만 통과(Macro-F1 19, 등급0/4/5 recall 0%) — 구조적 한계 수용.
EV_DIGS = {"wrinkle", "sagging", "forehead_wrinkle", "pore"}


@torch.no_grad()
def infer(patches, model_list, device, tta=True):
    """각 facepart의 등급 추론.

    area 4/6은 area 3/5 모델 + 좌우 반전으로 처리 (학습 코드와 동일).
    tta=True면 좌우 flip한 입력도 추론해서 logits 평균 → 안정성 ↑.
    항목별 하이브리드 정책(EV_DIGS): 일부는 Expected-value, 나머지는 argmax.

    Returns: dict {라벨키: (grade, max_grade, top1_prob, off1_prob)}
    """
    results = {}

    for area_id, patch_bgr in patches.items():
        # area 4/6은 전용 모델 없이 area 3/5 모델 공유.
        # 학습 코드(model.py)의 str vs int 비교 버그로 area 4/6 좌우 flip이
        # 실제로 적용된 적이 없음 → 추론도 flip 없이 그대로 넣어야 학습과 일치.
        if area_id == 4:
            model_idx = 3
        elif area_id == 6:
            model_idx = 5
        else:
            model_idx = area_id

        if model_idx not in model_list:
            continue

        x = preprocess_patch(patch_bgr, mode="class").to(device)
        logits = model_list[model_idx](x)  # [1, total_class]
        if tta:
            logits = (logits + model_list[model_idx](torch.flip(x, dims=[3]))) / 2

        num = 0
        for lbl in AREA_LABELS[area_id]:
            dig = diag_name(lbl)
            cn = class_num_list[dig]
            slice_ = logits[:, num:num + cn]
            probs = torch.softmax(slice_, dim=1)[0]
            argmax_grade = int(torch.argmax(probs).item())
            if dig in EV_DIGS:
                # Expected-value: 등급 기대값 반올림 (±1↑)
                ev = (probs * torch.arange(cn, device=probs.device,
                                           dtype=probs.dtype)).sum()
                grade = int(min(cn - 1, max(0, round(float(ev.item())))))
            else:
                grade = argmax_grade
            top1 = float(probs[grade].item())
            # ±1 합산 (인접 등급까지 신뢰도)
            lo = max(0, grade - 1)
            hi = min(cn, grade + 2)
            off1 = float(probs[lo:hi].sum().item())
            results[lbl] = (grade, cn - 1, top1, off1)
            num += cn

    return results


def _grade_label(score):
    """0~100 건강 점수 → 5단계 등급 라벨 (높을수록 양호).

    0~20:   심각
    20~40:  주의
    40~60:  보통
    60~80:  양호
    80~100: 매우 양호
    """
    if score >= 80:
        return "매우 양호"
    if score >= 60:
        return "양호"
    if score >= 40:
        return "보통"
    if score >= 20:
        return "주의"
    return "심각"


def load_calibration(path):
    """캘리브레이션 JSON 로드. 없으면 None 반환."""
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _percentile_rank(raw, calib_entry):
    """raw가 학습 valid 분포의 어느 percentile에 위치하는지 (0~100).

    저장된 percentile 배열을 linear interpolation으로 inverse lookup.
    분포 밖이면 0/100으로 클램프.
    """
    pct = calib_entry["percentiles"]
    p_keys = sorted(int(k) for k in pct.keys())
    p_vals = [pct[str(k)] for k in p_keys]

    if raw <= p_vals[0]:
        return 0.0
    if raw >= p_vals[-1]:
        return 100.0

    for i in range(len(p_vals) - 1):
        if p_vals[i] <= raw <= p_vals[i + 1]:
            if p_vals[i + 1] == p_vals[i]:
                return float(p_keys[i])
            frac = (raw - p_vals[i]) / (p_vals[i + 1] - p_vals[i])
            return float(p_keys[i] + frac * (p_keys[i + 1] - p_keys[i]))
    return 50.0


def _reg_score(raw, item_class, calibration=None, label=None):
    """회귀 출력 → 0~100 점수 (낮을수록 양호, 내부 severity).

    calibration 있으면 학습 valid 분포 기준 percentile rank로 환산.
      - 측정값 자체가 높을수록 양호(moisture/R²)면 percentile 높을수록 양호
        → severity = 100 - percentile
      - 측정값이 낮을수록 양호(Ra/pore/count)면 percentile 낮을수록 양호
        → severity = percentile

    calibration 없으면 raw 자체로 단순 환산 (도메인 shift 보정 안 됨).
    """
    direction = SCORE_DIRECTION.get(item_class, "higher")

    if calibration is not None and label and label in calibration:
        rank = _percentile_rank(raw, calibration[label])
        severity = (100 - rank) if direction == "higher" else rank
    else:
        if direction == "higher":
            severity = (1 - raw) * 100
        else:
            severity = raw * 100

    return round(max(0.0, min(100.0, severity)), 1)


@torch.no_grad()
def infer_regression(patches, face_patch, model_list, device, calibration=None, tta=True):
    """회귀: 각 부위 측정값 추론.

    학습 시 model.py run()과 동일하게 patch shape에 따라 분할 forward + 합산.
    calibration 있으면 percentile rank로 환산해 도메인 shift 보정.
    tta=True면 128×128 단일 forward 케이스에서 flip 평균 추가
    (shape>128 케이스는 학습 시점에 이미 split+flip 합산이라 TTA 미적용).

    Returns: dict {라벨키: (raw, real_value, severity_0_100)}
    """
    results = {}

    if 0 in model_list and face_patch is not None:
        x = preprocess_patch(face_patch, mode="regression", is_area0=True).to(device)
        out = model_list[0](x)
        if tta:
            out = (out + model_list[0](torch.flip(x, dims=[3]))) / 2
        out = out[0]
        lbl = REG_AREA_LABELS[0][0]
        item_class = lbl.split("_")[-1]
        raw = float(out[0].item())
        real = raw * REG_DENORM.get(item_class, 1.0)
        results[lbl] = (raw, real, _reg_score(raw, item_class, calibration, lbl))

    for area_id, patch_bgr in patches.items():
        # (infer와 동일) 학습 시 str/int 버그로 area 4/6 flip 미적용 →
        # 추론도 flip 없이 그대로. area 3/5 모델만 공유.
        if area_id == 4:
            model_idx = 3
        elif area_id == 6:
            model_idx = 5
        else:
            model_idx = area_id

        if model_idx not in model_list or area_id not in REG_AREA_LABELS:
            continue

        x = preprocess_patch(patch_bgr, mode="regression").to(device)
        model = model_list[model_idx]

        if x.shape[-1] > 128:
            x_l = x[:, :, :, :128]
            x_r = torch.flip(x[:, :, :, 128:], dims=[3])
            out = model(x_l) + model(x_r)
        elif x.shape[-2] > 128:
            x_l = x[:, :, :128, :]
            x_r = torch.flip(x[:, :, 128:, :], dims=[2])
            out = model(x_l) + model(x_r)
        else:
            out = model(x)
            if tta:
                out = (out + model(torch.flip(x, dims=[3]))) / 2

        out = out[0]
        for i, lbl in enumerate(REG_AREA_LABELS[area_id]):
            item_class = lbl.split("_")[-1]
            raw = float(out[i].item())
            real = raw * REG_DENORM.get(item_class, 1.0)
            results[lbl] = (raw, real, _reg_score(raw, item_class, calibration, lbl))

    return results


def aggregate_reg_scores(reg_results):
    """라벨별 회귀 점수 → 진단 항목별 평균 점수.

    예: moisture는 forehead + l_cheek + r_cheek + chin 4개 → 평균.
    """
    agg = {}
    for lbl, (_, _, score) in reg_results.items():
        item_class = lbl.split("_")[-1]
        agg.setdefault(item_class, []).append(score)
    return {k: round(sum(v) / len(v), 1) for k, v in agg.items()}


# ────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="입력 얼굴 사진 경로")
    parser.add_argument(
        "--checkpoint",
        default=r"checkpoint2\class\100%\1,2,3",
        help="분류 체크포인트 루트 (모델별 {idx}/state_dict.bin 위치)",
    )
    parser.add_argument(
        "--reg_checkpoint",
        default=r"checkpoint2\regression\100%\1,2,3",
        help="회귀 체크포인트 루트 (없거나 --no_regression이면 회귀 skip)",
    )
    parser.add_argument("--no_regression", action="store_true",
                        help="회귀 추론 끄기 (분류만 실행)")
    parser.add_argument(
        "--calibration", default="reg_calibration.json",
        help="회귀 캘리브레이션 JSON 경로 (calibrate_regression.py 결과). "
             "파일 없으면 raw 기반 단순 환산으로 fallback.",
    )
    parser.add_argument("--tta", action="store_true",
                        help="TTA(좌우 flip 평균) 켜기. 기본 off — "
                             "현재 모델은 학습 시 flip aug 미적용이라 TTA가 정확도를 떨어뜨림. "
                             "augmentation 재학습 후 다시 켜는 것을 권장.")
    parser.add_argument(
        "--debug_dir", default=None,
        help="bbox 시각화 jpg를 저장할 폴더 (없으면 저장 안 함)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"디바이스: {device}\n")

    # 1) 이미지 로드
    image = cv2.imread(args.image)
    if image is None:
        print(f"이미지 로드 실패: {args.image}")
        return
    print(f"입력: {args.image} (shape={image.shape})\n")

    # 2) 얼굴 분할
    print("[1/3] MediaPipe로 얼굴 분할…")
    debug_path = None
    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)
        debug_path = os.path.join(args.debug_dir, "faceparts.jpg")

    patches, debug_img, face_patch = extract_faceparts(image, draw_debug_path=debug_path)
    if not patches:
        print("  얼굴 감지 실패 — 정면 얼굴이 잘 보이는 사진인지 확인하세요.")
        return
    print(f"  추출 {len(patches)}개: {[AREA_NAMES[k] for k in sorted(patches.keys())]}")
    if debug_img is not None:
        cv2.imwrite(debug_path, debug_img)
        print(f"  bbox 시각화 저장: {debug_path}")

    # 3) 분류 모델 로드
    print("\n[2/3] 분류 모델 로드…")
    model_list = load_models(args.checkpoint, device, mode="class")
    if not model_list:
        print("  로드된 분류 모델이 없습니다.")
        return

    # 4) 분류 추론
    tta = args.tta
    print(f"\n[3/3] 분류 추론 중… (TTA {'on' if tta else 'off'})")
    results = infer(patches, model_list, device, tta=tta)

    # 5) 회귀 모델 로드 + 추론
    reg_results, reg_scores = {}, {}
    calibration = None
    if not args.no_regression and os.path.isdir(args.reg_checkpoint):
        print("\n[+] 회귀 모델 로드…")
        reg_models = load_models(args.reg_checkpoint, device, mode="regression")
        if reg_models:
            calibration = load_calibration(args.calibration)
            if calibration:
                print(f"[+] 캘리브레이션 로드: {args.calibration} ({len(calibration)}항목)")
            else:
                print(f"[알림] 캘리브레이션 JSON 없음 — raw 기반 환산 ({args.calibration})")
            print(f"[+] 회귀 추론 중… (TTA {'on' if tta else 'off'})")
            reg_results = infer_regression(
                patches, face_patch, reg_models, device, calibration, tta=tta
            )
            reg_scores = aggregate_reg_scores(reg_results)
    elif not args.no_regression:
        print(f"\n[알림] 회귀 체크포인트 없음 — skip ({args.reg_checkpoint})")

    # 6) 결과 출력
    import unicodedata

    def _wlen(s):
        return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)

    def _wpad(s, width):
        return s + " " * max(0, width - _wlen(s))

    print("\n" + "=" * 64)
    print("진단 결과")
    print("=" * 64)

    # ── 분류 (등급만 표시, 한글 라벨) ──
    print("\n[분류: 부위별 등급]  (낮을수록 양호 · Top-1 / ±1 합산)")
    for lbl, (g, max_g, top1, off1) in sorted(results.items()):
        kr = _kr_label(lbl)
        print(f"  {_wpad(kr, 22)} 등급 {g}/{max_g}   top-1 {top1*100:5.1f}%   ±1 {off1*100:5.1f}%")

    # ── 회귀 (사용자 표시: 건강 점수 = 100 - 심각도, 5단계 등급) ──
    if reg_results:
        print("\n" + "-" * 64)
        print("\n[회귀: 부위별 측정값 → 건강 점수]  (0~100, 높을수록 양호)")
        for lbl, (raw, real, severity) in sorted(reg_results.items()):
            item_class = lbl.split("_")[-1]
            good_dir = "↑양호" if SCORE_DIRECTION.get(item_class) == "higher" else "↓양호"
            health = round(100 - severity, 1)
            grade = _grade_label(health)
            kr = _kr_label(lbl)
            print(f"  {_wpad(kr, 22)} raw={raw:6.3f}  실측={real:8.2f} ({good_dir})  "
                  f"점수 {health:>5.1f}  [{grade}]")

        if reg_scores:
            # 신뢰도 티어 (test split Pearson r): 정밀 vs 참고
            #   count r0.94 / pore r0.73 → 정밀
            #   Ra r0.47 / moisture·R2 r~0.5 → 참고(평균회귀로 보수적)
            reliability = {"count": "정밀", "pore": "정밀",
                           "Ra": "참고", "moisture": "참고", "R2": "참고"}
            print("\n[회귀: 항목별 건강 점수]  (0~100, 높을수록 양호)")
            for item_class, severity in reg_scores.items():
                name_kr = REG_LABEL_KR.get(item_class, item_class)
                health = round(100 - severity, 1)
                grade = _grade_label(health)
                tier = reliability.get(item_class, "참고")
                print(f"  {_wpad(name_kr, 20)} {health:>5.1f}점  [{grade}]  ({tier})")
            # 종합점수 제거: 신뢰도 다른 항목 균등평균은 오해 소지
            # (약한 항목 평균회귀가 종합을 보수적으로 끌어내림).
            # 사용자/발표엔 항목별 + 신뢰도 티어로 제시.

    print("=" * 64)


if __name__ == "__main__":
    main()
