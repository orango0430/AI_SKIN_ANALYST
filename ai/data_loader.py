from PIL import Image
import natsort
import torch
from torchvision import transforms
import os
import numpy as np
import cv2
import json
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import random_split
from torch.utils.data import Dataset, ConcatDataset


def multi_area_collate(batch):
    """
    배치(list of area_list dict)를 area별로 묶음.
    샘플마다 가진 area 키가 달라서 default_collate를 못 씀.

    Returns: dict {area_key: [imgs_tensor[N], labels, descs_list]}
      class 모드: labels는 dict (라벨키 -> long tensor[N])
      regression 모드: labels는 stacked tensor[N, D]
    """
    result = {}
    all_areas = set()
    for sample in batch:
        if isinstance(sample, dict):
            all_areas.update(sample.keys())

    for area in all_areas:
        imgs, labels_buf, descs = [], [], []
        for sample in batch:
            if area not in sample:
                continue
            imgs.append(sample[area][0])
            labels_buf.append(sample[area][1])
            descs.append(sample[area][2])

        if not imgs:
            continue

        try:
            imgs_t = torch.stack(imgs, dim=0)
        except RuntimeError:
            continue

        first = labels_buf[0]
        if isinstance(first, dict):
            # 라벨 키의 union을 수집 후 누락/invalid 샘플은 -1 sentinel
            # (intersection으로 자르면 batch 단위로 라벨이 통째 사라짐)
            all_keys = set()
            for l in labels_buf:
                all_keys.update(l.keys())
            label_dict = {}
            for k in all_keys:
                vals = []
                for l in labels_buf:
                    if k not in l:
                        vals.append(-1)
                        continue
                    try:
                        vals.append(int(l[k]))
                    except (ValueError, TypeError):
                        vals.append(-1)
                label_dict[k] = torch.tensor(vals, dtype=torch.long)
            labels_out = label_dict
        elif isinstance(first, torch.Tensor):
            labels_out = torch.stack(labels_buf, dim=0)
        else:
            labels_out = labels_buf

        result[area] = [imgs_t, labels_out, descs]

    return result

folder_name = {
    "F": "01",
    "Fb": "07",
    "Ft": "06",
    "L15": "02",
    "L30": "03",
    "R15": "04",
    "R30": "05",
    "L": "02",
    "R": "03",
}

class_num_list = {
    "pigmentation": 6,       # 등급 0~5
    "wrinkle": 7,            # 등급 0~6 (perocular, glabellus)
    "pore": 6,               # 등급 0~5
    "dryness": 5,            # 등급 0~4
    "sagging": 6,            # 등급 0~5 (원본 0~6 중 등급6 13장은 등급5로 머지)
    "forehead_wrinkle": 7,   # 실제 분포 0~6 (원본 사양 9 미사용)
}


area_naming = {
    "0": "all",
    "1": "forehead",
    "2": "glabellus",
    "3": "l_peroucular",
    "4": "r_peroucular",
    "5": "l_cheek",
    "6": "r_cheek",
    "7": "lip",
    "8": "chin",
}

img_num = {
    "01": 7,
    "02": 3,
    "03": 3,
}


class CustomDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.load_list(args)
        self.train_list, self.val_list, self.test_list = random_split(
            self.dataset, [0.8, 0.1, 0.1]
        )
        self.remove_list = defaultdict(int)
        self.sub_path = []

    def __len__(self):
        return len(self.sub_path)

    def __getitem__(self, idx):
        # 미리 추출된 패치를 직접 읽음 (4K 원본 디코딩 안 함 -> 메모리 부담 매우 적음)
        # 패치 경로: {patch_path}/{equ_name}/{sub_fold}/{img_basename}_area{NN}.jpg
        value = self.sub_path[idx]
        equ_name = value["equ_name"]
        folder_path = value["folder_path"]
        sub_fold = folder_path.replace("\\", "/").split("/")[-1]
        img_name = value["img_name"]
        angle = img_name.split(".")[0].split("_")[-1]
        img_basename = os.path.splitext(img_name)[0]

        area_list = dict()
        start_idx = 1 if self.args.mode == "class" else 0

        for idx_area in range(start_idx, 9):
            try:
                # 1. 미리 추출된 패치 직접 읽기
                patch_path = os.path.join(
                    self.args.patch_path,
                    equ_name,
                    sub_fold,
                    f"{img_basename}_area{idx_area:02d}.jpg",
                )
                if not os.path.isfile(patch_path):
                    continue

                ori_patch_img = cv2.imread(patch_path)
                if ori_patch_img is None:
                    continue

                # 2. JSON에서 라벨 정보 읽기
                json_name = (
                    "_".join(img_name.split("_")[:2])
                    + f"_{angle}_{idx_area:02d}.json"
                )
                json_full = os.path.join(
                    self.json_path,
                    equ_name,
                    json_name.split("_")[0],
                    json_name,
                )
                if not os.path.isfile(json_full):
                    continue

                with open(json_full, "r", encoding="utf8") as f:
                    meta = json.load(f)

                area_name = str(idx_area)

                # 3. 기존 resize/make_double 로직 그대로
                reduction_value = max(ori_patch_img.shape[:2]) / self.args.res
                if reduction_value <= 0:
                    continue

                if idx_area != 0:
                    n_patch_img = cv2.resize(
                        ori_patch_img,
                        (
                            max(1, int(ori_patch_img.shape[1] / reduction_value)),
                            max(1, int(ori_patch_img.shape[0] / reduction_value)),
                        ),
                    )
                    patch_img = self.make_double(n_patch_img)
                    if not isinstance(patch_img, np.ndarray):
                        continue
                else:
                    patch_img = cv2.resize(
                        ori_patch_img, (self.args.res, self.args.res)
                    )

                pil_img = Image.fromarray(patch_img)
                patch_img = self.transform(pil_img)

                label_data = (
                    meta["annotations"] if self.args.mode == "class"
                    else meta["equipment"]
                )
                if type(label_data) != dict:
                    continue

                if self.args.mode != "class":
                    label_data = torch.tensor(self.norm_reg(meta, idx_area))

                desc_area = (
                    "Sub_" + sub_fold
                    + "_Equ_" + equ_name
                    + "_Angle_" + angle
                    + "_Area_" + area_name
                )
                area_list[f"{idx_area}"] = [
                    patch_img,
                    label_data,
                    desc_area,
                ]
            except Exception:
                continue

        return area_list

    def load_list(self, args):
        self.img_path = args.img_path
        self.dataset = list()
        self.json_path = args.json_path
        sub_path_list = [
            item for item in natsort.natsorted(os.listdir(self.img_path)) if not item.startswith(".")
        ]
        self.transform = transforms.ToTensor()

        for equ_name in sub_path_list:
            if equ_name.startswith("."):
                continue

            for sub_fold in natsort.natsorted(os.listdir(os.path.join(self.img_path, equ_name))):
                if sub_fold.startswith(".") or not os.path.exists(
                    os.path.join(self.json_path, equ_name, sub_fold)
                ):
                    continue

                folder_path = os.path.join(self.img_path, equ_name, sub_fold)
                for img_name in natsort.natsorted(os.listdir(folder_path)):
                    if not img_name.endswith((".png", ".jpg", ".jpeg")):
                        continue

                    self.dataset.append(
                        {
                            "equ_name": equ_name,
                            "folder_path": folder_path,
                            "img_name": img_name,
                        }
                    )

    def load_dataset(self, args, mode):
        # ★ 핵심 수정: 이미지를 읽지 않고 경로 정보만 sub_path에 저장
        self.args = args
        self.sub_path = list()

        # 학습 모드 + --use_aug 옵션 켜졌을 때만 augmentation 적용
        # 일반 사진(폰/조명/색감 변동)에 대한 도메인 일반화를 위함
        use_aug = mode == "train" and getattr(args, "use_aug", False)
        if use_aug:
            self.transform = transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02
                ),
                transforms.RandomAffine(
                    degrees=3, translate=(0.02, 0.02), scale=(0.97, 1.03), fill=0
                ),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.ToTensor()

        data_list = (
            self.train_list if mode == "train"
            else self.val_list if mode == "val"
            else self.test_list
        )
        for value in tqdm(data_list, desc=f"[{mode}] 경로 로딩 중"):
            self.sub_path.append(value)

    def make_double(self, n_patch_img):
        """모드별 padding 분기.

        - class:      letterbox 중앙 정렬 (기존 분류 체크포인트 호환)
        - regression: corner 정렬 + 64 미만 차원은 2x 확대
                      (regression_data_loader.py와 동일 = 기존 회귀 체크포인트 호환)

        __getitem__에서 reduction_value로 max차원=res로 줄인 후 호출됨.
        """
        if getattr(self.args, "mode", "class") == "regression":
            return self._make_double_corner(n_patch_img)
        return self._make_double_letterbox(n_patch_img)

    def _make_double_letterbox(self, n_patch_img):
        h, w = n_patch_img.shape[:2]
        if h > 128 or w > 128:
            scale = 128 / max(h, w)
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))
            n_patch_img = cv2.resize(n_patch_img, (new_w, new_h))
            h, w = n_patch_img.shape[:2]
        patch_img = np.zeros((128, 128, 3), dtype=np.uint8)
        y0 = (128 - h) // 2
        x0 = (128 - w) // 2
        patch_img[y0:y0 + h, x0:x0 + w] = n_patch_img
        return patch_img

    def _make_double_corner(self, n_patch_img):
        # regression_data_loader.py make_double과 동일 로직
        h, w = n_patch_img.shape[:2]
        if h < 64:
            patch_img = np.zeros((128, 256, 3), dtype=np.uint8)
            doubled = cv2.resize(n_patch_img, (w * 2, h * 2))
            patch_img[:h * 2, :w * 2] = doubled
        elif w < 64:
            patch_img = np.zeros((256, 128, 3), dtype=np.uint8)
            doubled = cv2.resize(n_patch_img, (w * 2, h * 2))
            patch_img[:h * 2, :w * 2] = doubled
        else:
            patch_img = np.zeros((128, 128, 3), dtype=np.uint8)
            patch_img[:h, :w] = n_patch_img
        return patch_img

    def load_img(self, img_name, angle, idx_area, equ_name, img, args, img_scale=1.0):
        json_name = "_".join(img_name.split("_")[:2]) + f"_{angle}_{idx_area:02d}.json"
        with open(
            os.path.join(
                self.json_path,
                equ_name,
                json_name.split("_")[0],
                json_name,
            ),
            "r",
            encoding="utf8",
        ) as f:
            meta = json.load(f)

        if meta["images"]["bbox"] == None:
            return 1

        # 디코딩 단계에서 줄어든 만큼 bbox 좌표도 비례 조정
        bbox_point = [int(item * img_scale) for item in meta["images"]["bbox"]]
        bbox_x = [
            min(bbox_point[0], bbox_point[2]),
            max(bbox_point[0], bbox_point[2]),
        ]
        bbox_y = [
            min(bbox_point[1], bbox_point[3]),
            max(bbox_point[1], bbox_point[3]),
        ]

        if (bbox_x[1] - bbox_x[0]) < 90 or (bbox_y[1] - bbox_y[0]) < 90:
            self.remove_list[str(idx_area)] += 1

        area_name = str(int(json_name.split("_")[-1].split(".")[0]))
        patch_img = img[bbox_y[0] : bbox_y[1], bbox_x[0] : bbox_x[1]]

        reduction_value = max(patch_img.shape) / args.res

        return reduction_value, json_name, area_name, meta, patch_img

    def norm_reg(self, meta, idx_area):
        item_list = list()
        type_class = {
            "0": ["pigmentation_count"],
            "1": ["forehead_moisture", "forehead_elasticity_R2"],
            "3": ["l_perocular_wrinkle_Ra"],
            "4": ["r_perocular_wrinkle_Ra"],
            "5": ["l_cheek_moisture", "l_cheek_elasticity_R2", "l_cheek_pore"],
            "6": ["r_cheek_moisture", "r_cheek_elasticity_R2", "r_cheek_pore"],
            "8": ["chin_moisture", "chin_elasticity_R2"],
        }

        for item in type_class[f"{idx_area}"]:
            item_class = item.split("_")[-1]
            if meta["equipment"][item] == "Er":
                item_list.append(np.nan)
            else:
                if item_class == "R2":
                    item_list.append(meta["equipment"][item])

                elif item_class in ["moisture", "Ra"]:
                    item_list.append(meta["equipment"][item] / 100)

                elif item_class == "count":
                    item_list.append(meta["equipment"][item] / 350)

                elif item_class == "pore":
                    item_list.append(meta["equipment"][item] / 3000)

                else:
                    assert 0, "item_class is not here"

        return item_list