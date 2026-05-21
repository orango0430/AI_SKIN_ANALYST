"""회귀용 128×128 패치 1회 사전추출 (D 원본 → 빠른 드라이브).

배경: regression_train/eval이 매 에폭 D드라이브 4K 원본을 재디코딩 →
느림. 분류가 crop_img 쓰듯, 회귀도 한 번만 디코딩해 작은 패치로 저장.

핵심 원칙: regression_data_loader의 load_img(json bbox crop) + 128×128
resize 를 그대로 써서 추출 → 학습(regression_train --patch_dir),
평가, inference 전처리가 픽셀 단위로 완전 동일.

분류의 crop_img는 재사용 불가 (랜드마크/letterbox crop이라 정의 다르고
area0=전체얼굴 없음). 회귀 전용으로 새로 추출.

저장 구조:
  <out>/<equ_name>/<sub_fold>/<imgbase>__a<area>.png   (128×128 BGR)
  <out>/manifest.pkl  : { key(str) -> label(list[float], Er는 nan) }
  key = f"{equ_name}|{sub_fold}|{img_name}|{area}"  (split 무관, 전수 저장)

split은 추출과 분리 — regression_train/eval이 CustomDataset+시드523
random_split 그대로 재현하므로 train/val/test 동일 보장.

사용:
  set PYTHONIOENCODING=utf-8
  python regression_extract_patches.py ^
     --img_path "D:\korean_skin_data\open_data\data\Training\train_data" ^
     --json_path "<라벨>" --out "C:\skin_reg_patch" --res 128
"""
import argparse
import json
import os
import pickle

import cv2
import numpy as np
from tqdm import tqdm

from regression_data_loader import CustomDataset

REG_AREAS = [0, 1, 3, 5, 8]   # 회귀 대상 area (inference.REG_AREA_LABELS 기준)


def sub_fold_of(folder_path):
    """Windows/Unix 모두 안전한 마지막 폴더명."""
    return os.path.basename(os.path.normpath(folder_path))


def patch_key(equ_name, sub_fold, img_name, area):
    return f"{equ_name}|{sub_fold}|{img_name}|{area}"


def patch_path(out, equ_name, sub_fold, img_name, area):
    # 분류 extract_patches.py와 동일 명명 → crop_img 그대로 재사용 가능.
    base = os.path.splitext(img_name)[0]
    return os.path.join(out, equ_name, sub_fold, f"{base}_area{area:02d}.jpg")


def read_meta_jsononly(json_root, equ_name, img_name, angle, area):
    """원본 디코딩 없이 per-area json만 읽음 (regression_data_loader.load_img
    의 json 경로 규칙 그대로). 라벨(equipment)·bbox 모두 이 파일에 있음."""
    json_name = "_".join(img_name.split("_")[:2]) + f"_{angle}_{area:02d}.json"
    path = os.path.join(json_root, equ_name, json_name.split("_")[0], json_name)
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_path", required=True, type=str)
    p.add_argument("--json_path", required=True, type=str)
    p.add_argument("--out", required=True, type=str,
                   help="manifest.pkl 저장 위치. crop_img 재사용 시 그 폴더 권장.")
    p.add_argument("--crop_img", default="", type=str,
                   help="분류가 만든 기존 패치 폴더(예: C:\\skin_data\\crop_img). "
                        "지정 시 4K 재추출 안 함 — json만 스캔해 라벨 manifest "
                        "생성하고 그 패치를 그대로 학습에 씀(가장 빠름).")
    p.add_argument("--res", default=128, type=int)
    p.add_argument("--mode", default="regression", type=str)  # loader 호환용
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    base = CustomDataset(args)   # load_list (전수 인덱스). 디코딩 안 함.
    samples = base.dataset
    patch_root = args.crop_img if args.crop_img else args.out
    mode = "crop_img 재사용(json만 스캔)" if args.crop_img else "원본 4K 추출"
    print(f"총 이미지 {len(samples):,}장 → area {REG_AREAS}  [{mode}]")

    manifest = {}
    n_ok, n_skip, n_imgfail = 0, 0, 0

    for v in tqdm(samples):
        equ_name = v["equ_name"]
        folder_path = v["folder_path"]
        img_name = v["img_name"]
        sub_fold = sub_fold_of(folder_path)
        angle = img_name.split(".")[0].split("_")[-1]
        img = None  # crop_img 모드면 4K 디코딩 자체를 안 함

        for area in REG_AREAS:
            dst = patch_path(patch_root, equ_name, sub_fold, img_name, area)
            key = patch_key(equ_name, sub_fold, img_name, area)
            try:
                if args.crop_img:
                    # 패치가 이미 있는 것만 (bbox None이면 분류도 안 만들었음)
                    if not os.path.isfile(dst):
                        n_skip += 1
                        continue
                    meta = read_meta_jsononly(
                        args.json_path, equ_name, img_name, angle, area)
                else:
                    if img is None:
                        img = cv2.imread(os.path.join(folder_path, img_name))
                        if img is None:
                            n_imgfail += 1
                            break
                    _, _, _, meta, ori = base.load_img(
                        img_name, angle, area, equ_name, img, args)
                if not isinstance(meta.get("equipment"), dict):
                    n_skip += 1
                    continue
                label = np.asarray(base.norm_reg(meta, area),
                                   dtype=np.float32).tolist()  # Er → nan
                if not args.crop_img:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    cv2.imwrite(dst, cv2.resize(ori, (args.res, args.res)))
                manifest[key] = label
                n_ok += 1
            except Exception:
                n_skip += 1
                continue

    with open(os.path.join(args.out, "manifest.pkl"), "wb") as f:
        pickle.dump(manifest, f)

    print(f"\n완료: manifest {n_ok:,}개 / skip {n_skip:,} / "
          f"이미지실패 {n_imgfail:,}")
    print(f"manifest: {os.path.join(args.out, 'manifest.pkl')} "
          f"({len(manifest):,} keys)")
    print(f"패치 루트: {patch_root}")
    print(f"→ 학습: python regression_train.py --patch_dir \"{patch_root}\" "
          f"--img_path ... --json_path ... --areas \"0,1,3,5,8\"")
    print(f"→ 학습: python regression_train.py --patch_dir \"{args.out}\" ...")


if __name__ == "__main__":
    main()
