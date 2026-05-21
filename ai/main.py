import os

import shutil
import numpy as np
import torch
from torchvision import models

from tensorboardX import SummaryWriter

import copy
from torch.utils.data import random_split
from matplotlib import pyplot as plt
from logger import setup_logger
from data_loader import CustomDataset, multi_area_collate
from model import resume_checkpoint, mkdir, Model
from torchvision.models import ResNet50_Weights
import torch.nn as nn

import argparse
from torch.utils import data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name",
        default="100%/1,2,3",
        type=str,
    )

    parser.add_argument(
        "--img_path",
        default=r"D:\korean_skin_data\open_data\data\Training\train_data",
        type=str,
    )

    parser.add_argument(
        "--patch_path",
        default=r"C:\skin_data\crop_img",
        type=str,
        help="미리 추출된 패치 폴더 (extract_patches.py 결과)",
    )

    parser.add_argument(
        "--loss_dir",
        default="tensorboard",
        type=str,
    )

    parser.add_argument("--stop_early", type=int, default=30)

    parser.add_argument(
        "--mode",
        default="class",
        choices=["regression", "class"],
        type=str,
    )

    parser.add_argument(
        "--json_path",
        default=r"C:\skin_data\label data",
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        default="checkpoint2",
        type=str,
    )

    parser.add_argument(
        "--epoch",
        default=300,
        type=int,
    )

    parser.add_argument(
        "--res",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--load_epoch",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--lr",
        default=3e-4,
        type=float,
    )

    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
    )

    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
    )

    parser.add_argument("--reset", action="store_true")

    parser.add_argument(
        "--use_aug", action="store_true",
        help="학습 시 augmentation 적용 (ColorJitter + 소량 affine). "
             "도메인 일반화(외부 사진 성능)를 위함. 기본 off로 기존 학습 재현 가능.",
    )

    parser.add_argument(
        "--areas", default="", type=str,
        help="재학습할 area만 지정 (콤마구분, 예: '1,3'). 비우면 전체. "
             "area1=forehead, area3=perocular.",
    )

    args = parser.parse_args()

    args.areas = (
        {int(x) for x in args.areas.split(",") if x.strip() != ""}
        if args.areas else set()
    )

    return args



def main(args):
    log_path = os.path.join(args.loss_dir, args.mode, args.name)
    check_path = os.path.join(args.output_dir, args.mode, args.name)

    writer = SummaryWriter(log_path)
    mkdir(log_path)
    mkdir(check_path)

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Multi-class: 각 진단 항목의 원본 등급 수를 그대로 사용
    # area 1 forehead:  forehead_wrinkle(7) + forehead_pigmentation(6) = 13
    # area 2 glabellus: glabellus_wrinkle(7) = 7
    # area 3 l_perocular: l_perocular_wrinkle(7) = 7
    # area 5 l_cheek:   l_cheek_pigmentation(6) + l_cheek_pore(6) = 12
    # area 7 lip:       lip_dryness(5) = 5
    # area 8 chin:      chin_sagging(6) = 6  (등급6 13장은 5로 머지)
    # area 4, 6: mirror로 처리되므로 학습 모델 없음(0)
    model_num_class = (
        [np.nan, 13, 7, 7, 0, 12, 0, 5, 6]
        if args.mode == "class"
        else [1, 2, np.nan, 1, 0, 3, 0, np.nan, 2]
    )

    # best 기준이 ±1 accuracy로 변경됨 (높을수록 좋음) → -inf로 초기화
    args.best_loss = [-np.inf for _ in range(len(model_num_class))]
    model_list = [copy.deepcopy(model) for _ in range(len(model_num_class))]
    # Define 9 resnet models for each region
    resume_list = list()
    for idx, item in enumerate(model_num_class):
        if not np.isnan(item):
            model_list[idx].fc = nn.Linear(
                model_list[idx].fc.in_features, model_num_class[idx]
            )
            resume_list.append(idx)

    ## Adjust the number of output in model for each region image
    model_dict_path = os.path.join(check_path, "1", "state_dict.bin")

    if args.reset:
        print(f"\033[90mReseting......{model_dict_path}\033[0m")
        if os.path.isdir(check_path):
            shutil.rmtree(check_path)
            mkdir(check_path)
    # If there is check-point, load that

    if os.path.isfile(model_dict_path):
        print(f"\033[92mResuming......{model_dict_path}\033[0m")

        for idx in resume_list:
            if idx in [4, 6]:
                continue
            model_list[idx] = resume_checkpoint(
                args,
                model_list[idx],
                os.path.join(check_path, f"{idx}", "state_dict.bin"),
            )
        # 이전 학습은 best_loss 기준이라 스케일이 다름. 새 ±1 acc 기준으로 강제 초기화.
        args.best_loss = [-np.inf for _ in range(len(model_num_class))]

    logger = setup_logger(args.name, args.mode)
    logger.info(args)

    # train / val을 별도 dataset 인스턴스로 분리 (sub_path 덮어쓰기 버그 방지)
    # shallow copy로 train_list/val_list/test_list (random_split 결과)를 공유
    train_dataset = CustomDataset(args)
    val_dataset = copy.copy(train_dataset)
    val_dataset.sub_path = []
    val_dataset.remove_list = train_dataset.remove_list

    train_dataset.load_dataset(args, "train")
    trainset_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        collate_fn=multi_area_collate,
        pin_memory=True,
    )

    val_dataset.load_dataset(args, "val")
    valset_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=multi_area_collate,
        pin_memory=True,
    )

    resnet_model = Model(
        args, model_list, trainset_loader, valset_loader, logger, writer
    )

    for epoch in range(args.load_epoch, args.epoch):
        resnet_model.update_e(epoch + 1) if args.load_epoch else None

        for model_idx in range(len(model_num_class)):
            if np.isnan(model_num_class[model_idx]):
                continue
            # --areas로 특정 부위만 타깃 재학습 (예: "1,3"). 미지정 시 전체.
            if args.areas and model_idx not in args.areas:
                continue
            # In regression task, there are no images for 미간, 입술, 턱
            resnet_model.choice(model_idx)
            # Change the model for each region
            resnet_model.run(phase="train")
            resnet_model.run(phase="valid")
            
        resnet_model.update_m(model_num_class)

        # Valid 정확도 / Macro-F1 요약 (reset_log 전에 출력)
        resnet_model.epoch_report()

        # Show the result for each value, such as pigmentation and pore, by averaging all of them
        resnet_model.update_e(epoch + 1)
        resnet_model.reset_log(mode=args.mode)

        if resnet_model.stop_early():
            break
        
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)
