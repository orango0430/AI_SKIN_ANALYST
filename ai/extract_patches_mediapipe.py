"""학습 패치를 inference와 동일한 MediaPipe crop으로 추출 (도메인 통일).

문제: 회귀/분류 모델은 데이터셋 json bbox crop으로 학습됐는데 inference는
MediaPipe 랜드마크 crop. 이 도메인 갭 때문에 추론 성능이 학습 지표보다
떨어짐. 캘리브는 점수 환산만 보정할 뿐 모델이 못 본 도메인 문제는 못 고침.

해법(근본): 학습 패치를 inference.extract_faceparts(= 추론과 같은 코드)로
잘라 저장 → 그 패치로 재학습하면 모델이 실제 추론 도메인을 학습.

명명/매니페스트는 crop_img / regression_extract_patches와 동일 →
regression_train.py --patch_dir 그대로 사용 가능.

소규모 검증 권장: --max_images 2000 정도로 먼저 (전량은 수시간).

사용:
  set PYTHONIOENCODING=utf-8
  python extract_patches_mediapipe.py ^
     --img_path "D:\korean_skin_data\open_data\data\Training\train_data" ^
     --json_path "C:\skin data\label data" ^
     --out "C:\skin_mp_patch" --max_images 2000
"""
import argparse
import faulthandler
import os
import pickle

faulthandler.enable()  # 네이티브(C) 크래시도 스택 덤프

import cv2
import numpy as np
from tqdm import tqdm

from regression_data_loader import CustomDataset
from regression_extract_patches import (
    sub_fold_of, patch_key, patch_path, read_meta_jsononly,
)
from inference import extract_faceparts

# 회귀 대상 area. 0=전체얼굴(face_patch), 1/3/5/8 학습, 4/6은 inference용 보존
REG_AREAS = [0, 1, 3, 4, 5, 6, 8]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_path", required=True, type=str)
    p.add_argument("--json_path", required=True, type=str)
    p.add_argument("--out", required=True, type=str,
                   help="MediaPipe 패치 저장 루트 (예: C:\\skin_mp_patch)")
    p.add_argument("--max_images", type=int, default=2000,
                   help="0=전량. 소규모 검증은 2000 권장(전량은 수시간)")
    p.add_argument("--res", default=128, type=int)
    p.add_argument("--seed", default=523, type=int)
    p.add_argument("--mode", default="regression", type=str)  # loader 호환
    p.add_argument("--areas", default="", type=str,
                   help="추출할 area 쉼표(예: '8'). 비우면 전체(REG_AREAS). "
                        "한 area만 재추출 시 별도 --out 권장(manifest 분리).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    print("[1/3] import 완료, 인덱싱 시작 "
          f"(img_path={args.img_path})", flush=True)

    if args.areas.strip():
        sel = [int(a) for a in args.areas.split(",") if a.strip()]
        target_areas = [a for a in REG_AREAS if a in sel]
        assert target_areas, f"--areas '{args.areas}'가 REG_AREAS({REG_AREAS}) 와 교집합 없음"
    else:
        target_areas = list(REG_AREAS)
    print(f"[0/3] 추출 대상 area = {target_areas}", flush=True)

    base = CustomDataset(args)            # load_list (전수 인덱스)
    samples = list(base.dataset)
    print(f"[2/3] 인덱싱 완료: 전체 {len(samples):,}장", flush=True)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(samples)
    if args.max_images > 0:
        samples = samples[: args.max_images]
    print(f"[2.5/3] 샘플 {len(samples):,}장 선택, MediaPipe 모델 로딩 시도",
          flush=True)
    # 첫 extract_faceparts 호출에서 MediaPipe 그래프가 초기화됨.
    # 여기서 죽으면 학습 이미지에 MediaPipe 자체가 안 도는 것.
    _probe = cv2.imread(os.path.join(samples[0]["folder_path"],
                                     samples[0]["img_name"])) if samples else None
    print(f"[2.7/3] 첫 이미지 read: "
          f"{None if _probe is None else _probe.shape}", flush=True)
    if _probe is not None:
        _p, _, _fp = extract_faceparts(_probe, draw_debug_path=None)
        print(f"[2.9/3] 첫 extract_faceparts OK: keys={sorted(_p)} "
              f"face_patch={'있음' if _fp is not None else 'None'}", flush=True)
    print(f"[3/3] 대상 이미지 {len(samples):,}장 → MediaPipe crop 추출 "
          f"(area {target_areas})", flush=True)

    manifest = {}
    n_ok, n_noface, n_skip, n_imgfail = 0, 0, 0, 0
    mf_path = os.path.join(args.out, "manifest.pkl")
    print(f"[3.5/3] 루프 진입 직전 (samples={len(samples)}, mf_path={mf_path})",
          flush=True)
    import sys as _sys; _sys.stdout.flush(); _sys.stderr.flush()

    for k, v in enumerate(tqdm(samples, desc="mediapipe-extract", file=_sys.stdout)):
        if k == 0:
            print(f"[3.6/3] 첫 iter 진입: img_name={v.get('img_name')}",
                  flush=True)
        equ_name = v["equ_name"]
        folder_path = v["folder_path"]
        img_name = v["img_name"]
        sub_fold = sub_fold_of(folder_path)
        angle = img_name.split(".")[0].split("_")[-1]
        full_img = os.path.join(folder_path, img_name)

        try:
            img = cv2.imread(full_img)
            if img is None:
                n_imgfail += 1
                if k == 0:
                    print(f"\n[진단] 첫 샘플 cv2.imread 실패: {full_img}")
                continue
            patches, _, face_patch = extract_faceparts(img, draw_debug_path=None)
        except Exception as e:
            n_imgfail += 1
            if k == 0:
                import traceback
                print(f"\n[진단] 첫 샘플 처리 예외: {full_img}\n{traceback.format_exc()}")
            continue

        if k == 0:
            # 첫 샘플 자가진단 — 무엇이 막히는지 즉시 보임
            jp0 = None
            try:
                _m = read_meta_jsononly(args.json_path, equ_name,
                                        img_name, angle, 0)
                jp0 = f"OK (equipment={isinstance(_m.get('equipment'), dict)})"
            except Exception as e:
                jp0 = f"실패: {type(e).__name__} {e}"
            print(f"\n[진단] img={full_img} shape={None if img is None else img.shape}")
            print(f"[진단] extract_faceparts → patches keys={sorted(patches)} "
                  f"face_patch={'있음' if face_patch is not None else 'None'}")
            print(f"[진단] json(area0) {jp0}\n")

        if not patches and face_patch is None:
            n_noface += 1
            continue

        for area in target_areas:
            src = face_patch if area == 0 else patches.get(area)
            if src is None or getattr(src, "size", 0) == 0:
                n_skip += 1
                if n_skip <= 10:
                    print(f"[skip] {img_name} area{area}: src 없음/빈배열",
                          flush=True)
                continue
            try:
                meta = read_meta_jsononly(
                    args.json_path, equ_name, img_name, angle, area)
                if not isinstance(meta.get("equipment"), dict):
                    n_skip += 1
                    if n_skip <= 10:
                        print(f"[skip] {img_name} area{area}: "
                              f"equipment dict 아님", flush=True)
                    continue
                label = np.asarray(base.norm_reg(meta, area),
                                   dtype=np.float32).tolist()  # Er→nan
            except Exception as e:
                n_skip += 1
                if n_skip <= 10:
                    import traceback
                    print(f"[skip] {img_name} area{area}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    if n_skip <= 3:
                        print(traceback.format_exc(), flush=True)
                continue

            dst = patch_path(args.out, equ_name, sub_fold, img_name, area)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            cv2.imwrite(dst, cv2.resize(src, (args.res, args.res)))
            manifest[patch_key(equ_name, sub_fold, img_name, area)] = label
            n_ok += 1

        if (k + 1) % 200 == 0:          # 중간 저장 (크래시 대비)
            with open(mf_path, "wb") as f:
                pickle.dump(manifest, f)

    with open(mf_path, "wb") as f:
        pickle.dump(manifest, f)

    print(f"\n완료: 패치 {n_ok:,} / 얼굴미검출 {n_noface:,} / "
          f"skip {n_skip:,} / 이미지실패 {n_imgfail:,}")
    print(f"manifest: {os.path.join(args.out, 'manifest.pkl')} "
          f"({len(manifest):,} keys)")
    print(f"패치 루트: {args.out}")


if __name__ == "__main__":
    main()
