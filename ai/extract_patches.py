"""
원본 이미지에서 bbox 영역 패치를 미리 추출하여 저장.

학습 시 4K 원본을 매번 디코딩하지 않고 작은 패치만 읽으면 메모리 부담이 크게 줄어듦.

생성되는 폴더 구조:
  output/
    equ_name/
      sub_fold/
        {img_basename}_area{NN}.jpg

사용:
  python extract_patches.py
  python extract_patches.py --max_size 1024            # 디스크 절약 (권장)
  python extract_patches.py --skip_existing            # 중단 후 재시작
  python extract_patches.py --img_path "...Validation\..." --json_path "..." --output "...patches\val"

추출 후 다음 단계: data_loader.py가 이 패치를 읽도록 수정 필요.
"""

import os
import cv2
import json
import gc
import argparse
import natsort
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_path",
        default=r"D:\korean_skin_data\open_data\data\Validation\train_data\VS",
        type=str,
    )
    parser.add_argument(
        "--json_path",
        default=r"D:\korean_skin_data\open_data\data\Validation\label_data\VL",
        type=str,
    )
    parser.add_argument(
        "--output",
        default=r"D:\crop_img_val",
        type=str,
        help="패치를 저장할 폴더",
    )
    parser.add_argument(
        "--num_areas",
        default=9,
        type=int,
        help="추출할 area 개수 (기본 9)",
    )
    parser.add_argument(
        "--quality",
        default=90,
        type=int,
        help="JPEG 저장 품질 (1~100). 높을수록 화질 좋고 용량 큼",
    )
    parser.add_argument(
        "--max_size",
        default=None,
        type=int,
        help="패치 최대 변 길이 (디스크 절약). 예: 1024. 없으면 원본 크기 유지",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="이미 추출된 패치는 건너뛰기 (재시작용)",
    )
    parser.add_argument(
        "--gc_every",
        default=50,
        type=int,
        help="N 이미지마다 명시적 가비지 컬렉트",
    )
    return parser.parse_args()


def process_one_image(info, args):
    """한 이미지에서 모든 area의 패치를 추출하여 저장."""
    img_path = info["img_path"]
    img_name = info["img_name"]
    angle = info["angle"]
    equ_name = info["equ_name"]
    sub_fold = info["sub_fold"]

    out_dir = os.path.join(args.output, equ_name, sub_fold)
    os.makedirs(out_dir, exist_ok=True)

    img_basename = os.path.splitext(img_name)[0]

    # 처리할 area 미리 결정 (skip_existing 적용)
    valid_areas = []
    for idx_area in range(args.num_areas):
        json_name = "_".join(img_name.split("_")[:2]) + f"_{angle}_{idx_area:02d}.json"
        json_full = os.path.join(
            args.json_path,
            equ_name,
            json_name.split("_")[0],
            json_name,
        )
        if not os.path.isfile(json_full):
            continue

        out_name = f"{img_basename}_area{idx_area:02d}.jpg"
        out_full = os.path.join(out_dir, out_name)

        if args.skip_existing and os.path.isfile(out_full):
            continue

        valid_areas.append((idx_area, json_full, out_full))

    if not valid_areas:
        return 0  # 처리할 area 없음 (skip_existing 또는 json 모두 없음)

    # 이미지 한 번만 읽음 (4K 디코딩 1회)
    img = cv2.imread(img_path)
    if img is None:
        return 0

    extracted = 0
    h, w = img.shape[:2]

    for idx_area, json_full, out_full in valid_areas:
        try:
            with open(json_full, "r", encoding="utf8") as f:
                meta = json.load(f)

            bbox_data = meta.get("images", {}).get("bbox")
            if bbox_data is None:
                continue

            bbox = [int(item) for item in bbox_data]
            bbox_x = [min(bbox[0], bbox[2]), max(bbox[0], bbox[2])]
            bbox_y = [min(bbox[1], bbox[3]), max(bbox[1], bbox[3])]

            # bbox 유효성 + 이미지 경계 클램프
            x1, x2 = max(0, bbox_x[0]), min(w, bbox_x[1])
            y1, y2 = max(0, bbox_y[0]), min(h, bbox_y[1])
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue

            patch = img[y1:y2, x1:x2]
            if patch.size == 0:
                continue

            # max_size 제한 (디스크 절약)
            if args.max_size is not None:
                ph, pw = patch.shape[:2]
                if max(ph, pw) > args.max_size:
                    scale = args.max_size / max(ph, pw)
                    new_w = max(1, int(pw * scale))
                    new_h = max(1, int(ph * scale))
                    patch = cv2.resize(patch, (new_w, new_h))

            cv2.imwrite(
                out_full,
                patch,
                [int(cv2.IMWRITE_JPEG_QUALITY), args.quality],
            )
            extracted += 1
        except Exception:
            continue

    del img
    return extracted


def main():
    args = parse_args()

    if not os.path.isdir(args.img_path):
        print(f"[오류] img_path 없음: {args.img_path}")
        return
    if not os.path.isdir(args.json_path):
        print(f"[오류] json_path 없음: {args.json_path}")
        return

    os.makedirs(args.output, exist_ok=True)

    print(f"입력 이미지: {args.img_path}")
    print(f"입력 JSON:   {args.json_path}")
    print(f"출력:         {args.output}")
    print(
        f"옵션: quality={args.quality}, max_size={args.max_size}, "
        f"skip_existing={args.skip_existing}"
    )
    print()

    # 이미지 목록 수집
    print("이미지 목록 수집 중...")
    image_list = []

    equ_names = [
        d for d in natsort.natsorted(os.listdir(args.img_path))
        if not d.startswith(".")
        and os.path.isdir(os.path.join(args.img_path, d))
    ]

    for equ_name in equ_names:
        equ_dir = os.path.join(args.img_path, equ_name)
        for sub_fold in natsort.natsorted(os.listdir(equ_dir)):
            if sub_fold.startswith("."):
                continue
            json_sub = os.path.join(args.json_path, equ_name, sub_fold)
            if not os.path.isdir(json_sub):
                continue
            folder = os.path.join(equ_dir, sub_fold)
            if not os.path.isdir(folder):
                continue
            for img_name in natsort.natsorted(os.listdir(folder)):
                if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                image_list.append({
                    "img_path": os.path.join(folder, img_name),
                    "img_name": img_name,
                    "angle": img_name.split(".")[0].split("_")[-1],
                    "equ_name": equ_name,
                    "sub_fold": sub_fold,
                })

    print(f"총 이미지: {len(image_list):,}장\n")

    total_patches = 0
    failed = 0

    for i, info in enumerate(tqdm(image_list, desc="패치 추출 중")):
        try:
            n = process_one_image(info, args)
            total_patches += n
        except Exception:
            failed += 1

        # 명시적 메모리 회수 (메모리 누적 방지)
        if (i + 1) % args.gc_every == 0:
            gc.collect()

    print()
    print("=" * 60)
    print("완료!")
    print(f"  추출된 패치: {total_patches:,}개")
    print(f"  실패한 이미지: {failed}장")
    print(f"  저장 위치: {args.output}")
    print("=" * 60)
    print("\n다음 단계: data_loader.py가 이 패치를 읽도록 수정")


if __name__ == "__main__":
    main()
