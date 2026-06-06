"""분류 학습 루프 A/B 테스트 (old vs new).

목적: model.py 분류 학습 루프의 두 버그가 정확도에 실제로 영향을 주는지
      '같은 데이터·같은 split(seed 523)·같은 class weight/aug/metric'에서
      플래그만 바꿔 통제 비교한다.

  --loop old : 현재 model.py 동작 재현
      · 에폭마다 Adam 재생성 (momentum 리셋)
      · 매 에폭 'best-on-val' 가중치에서 deepcopy 후 재시작 (학습 누적 X)
      · adjust_learning_rate 스텝감쇠 (epoch/2 에서 0.1배)
  --loop new : 표준 학습 (제안하는 수정)
      · Adam 1회 생성, 전 에폭 유지 (momentum 보존)
      · 최신 가중치 계속 누적 (best는 저장용으로만 추적)
      · ReduceLROnPlateau (val ±1 acc 기준)

공통(공정 비교를 위해 양쪽 동일):
  · split = 시드 523 random_split (분류 기본 코드의 무시드 버그 회피)
  · fc 슬라이스 = 추론(inference.AREA_LABELS)과 동일한 '고정 순서'
    (model.py는 label dict set 순서라 비결정적 — 본 스크립트는 양쪽 모두 고정)
  · 항목별 class weight CE (model.CLASS_WEIGHTS), 패치 128×128 단일 forward
  · 모델 선정 기준 = val ±1 accuracy 최대 (model.py와 동일)
  · 최종 비교 = best 체크포인트의 test split strict / ±1 / Macro-F1

사용:
  set PYTHONIOENCODING=utf-8
  # 패치 먼저 추출 (extract_patches.py --limit N)
  python train_clean.py --patch_path "C:\\skin_data\\crop_img" \
      --img_path "D:\\...\\Training\\train_data" \
      --json_path "D:\\...\\Training\\label data" \
      --areas 5 --loop old --epoch 40 --use_aug
  python train_clean.py ... --areas 5 --loop new --epoch 40 --use_aug
  # 두 출력의 [TEST] 표를 비교
"""
import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
from torch.utils import data
from torch.utils.data import random_split

from data_loader import CustomDataset, multi_area_collate, class_num_list
from model import CLASS_WEIGHTS, diag_name

try:
    from sklearn.metrics import f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 추론(inference.py)과 동일한 고정 슬라이스 순서. area 4/6은 3/5 공유라 미학습.
AREA_LABELS = {
    1: ["forehead_wrinkle", "forehead_pigmentation"],
    2: ["glabellus_wrinkle"],
    3: ["l_perocular_wrinkle"],
    5: ["l_cheek_pigmentation", "l_cheek_pore"],
    7: ["lip_dryness"],
    8: ["chin_sagging"],
}
AREA_OUTDIM = {
    a: sum(class_num_list[diag_name(l)] for l in labs)
    for a, labs in AREA_LABELS.items()
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="abtest", type=str,
                   help="checkpoint2/class/<name>/<loop>/<area> 에 저장")
    p.add_argument("--img_path",
                   default=r"D:\korean_skin_data\open_data\data\Training\train_data")
    p.add_argument("--json_path",
                   default=r"D:\korean_skin_data\open_data\data\Training\label data")
    p.add_argument("--patch_path", default=r"C:\skin_data\crop_img",
                   help="extract_patches.py 출력 (미리 추출된 분류 패치)")
    p.add_argument("--output_dir", default="checkpoint2")
    p.add_argument("--mode", default="class")  # data_loader 호환용
    p.add_argument("--loop", choices=["old", "new"], required=True,
                   help="old=현재 버그 재현, new=표준 학습")
    p.add_argument("--areas", default="5", type=str,
                   help="학습할 area (쉼표). 권장 테스트: 5(볼) 또는 3(눈가)/8(턱)")
    p.add_argument("--epoch", default=40, type=int)
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--lr", default=3e-4, type=float)
    p.add_argument("--res", default=128, type=int)
    p.add_argument("--seed", default=523, type=int)
    p.add_argument("--stop_early", default=12, type=int,
                   help="val ±1 acc 정체 epoch 수")
    p.add_argument("--use_aug", action="store_true",
                   help="train split에 ColorJitter+소량 affine (val/test 미적용)")
    args = p.parse_args()
    args.areas = [int(x) for x in args.areas.split(",") if x.strip() != ""]
    return args


# ───────────────────────── 데이터 ─────────────────────────
def build_views(args):
    """시드 고정 split → train/val/test 세 뷰. 분류 무시드 split 버그 회피."""
    base = CustomDataset(args)
    g = torch.Generator().manual_seed(args.seed)
    base.train_list, base.val_list, base.test_list = random_split(
        base.dataset, [0.8, 0.1, 0.1], generator=g
    )

    def view(split):
        v = copy.copy(base)
        v.sub_path = []
        v.remove_list = base.remove_list
        v.load_dataset(args, split)   # split별 transform(use_aug) 적용
        return v

    return view("train"), view("val"), view("test")


def make_loader(dataset, args, shuffle):
    return data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=shuffle, collate_fn=multi_area_collate, pin_memory=True,
    )


# ───────────────────────── 손실 / 메트릭 ─────────────────────────
def build_criterions():
    return {
        name: nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float, device=device))
        for name, w in CLASS_WEIGHTS.items()
    }


def _slice_targets(logits, labels, area):
    """고정 순서로 (diag, logit_slice, target) 산출. -1 sentinel/누락 제외."""
    num = 0
    for lbl in AREA_LABELS[area]:
        diag = diag_name(lbl)
        cn = class_num_list[diag]
        sl = logits[:, num:num + cn]
        num += cn
        if lbl not in labels:
            continue
        target = labels[lbl].to(device)
        mask = target >= 0
        if mask.sum() == 0:
            continue
        tv = target[mask]
        if diag == "sagging":
            tv = torch.clamp(tv, max=cn - 1)
        tv = torch.clamp(tv, min=0, max=cn - 1)
        yield diag, sl[mask], tv


def compute_loss(logits, labels, area, criterions):
    total, n = 0.0, 0
    for diag, sl, tv in _slice_targets(logits, labels, area):
        total = total + criterions[diag](sl, tv)
        n += 1
    return total if n > 0 else None


@torch.no_grad()
def evaluate(model, loader, area):
    """best 선정/리포트용. 항목별 strict/±1/Macro-F1 + 전체 mean ±1."""
    model.eval()
    bucket = {}  # diag -> (preds[], targets[])
    akey = str(area)
    for batch in loader:
        if akey not in batch:
            continue
        imgs = batch[akey][0].to(device)
        labels = batch[akey][1]
        logits = model(imgs)
        for diag, sl, tv in _slice_targets(logits, labels, area):
            pred = torch.argmax(sl, dim=1)
            b = bucket.setdefault(diag, ([], []))
            b[0].extend(pred.cpu().tolist())
            b[1].extend(tv.cpu().tolist())

    res = {}
    off1_all = []
    for diag, (preds, targets) in bucket.items():
        if not preds:
            continue
        p = np.array(preds)
        t = np.array(targets)
        strict = float((p == t).mean())
        off1 = float((np.abs(p - t) <= 1).mean())
        if HAS_SKLEARN:
            f1 = float(f1_score(t, p, average="macro",
                                labels=list(range(class_num_list[diag])),
                                zero_division=0))
        else:
            f1 = float("nan")
        res[diag] = dict(n=len(p), strict=strict, off1=off1, f1=f1)
        off1_all.append(off1)
    res["_mean_off1"] = float(np.mean(off1_all)) if off1_all else 0.0
    return res


# ───────────────────────── 학습 루프 ─────────────────────────
def new_model(area):
    m = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, AREA_OUTDIM[area])
    return m.to(device)


def train_one_epoch(model, loader, area, criterions, opt):
    model.train()
    akey = str(area)
    run, nb = 0.0, 0
    for batch in loader:
        if akey not in batch:
            continue
        imgs = batch[akey][0].to(device)
        labels = batch[akey][1]
        logits = model(imgs)
        loss = compute_loss(logits, labels, area, criterions)
        if loss is None:
            continue
        opt.zero_grad()
        loss.backward()
        opt.step()
        run += float(loss.detach())
        nb += 1
    return run / max(1, nb)


def train_area(area, args, tr_loader, va_loader, te_loader):
    crit = build_criterions()
    save_dir = os.path.join(args.output_dir, "class", args.name,
                            args.loop, str(area))
    os.makedirs(save_dir, exist_ok=True)

    best_metric = -1.0
    best_state = None
    stale = 0

    print(f"\n{'='*72}\n[area {area}] {AREA_LABELS[area]}  "
          f"out_dim={AREA_OUTDIM[area]}  loop={args.loop}\n{'='*72}")

    if args.loop == "new":
        # ── 표준: 영속 모델 + 영속 Adam + plateau, 최신 가중치 누적 ──
        model = new_model(area)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=4)
        for ep in range(1, args.epoch + 1):
            trloss = train_one_epoch(model, tr_loader, area, crit, opt)
            res = evaluate(model, va_loader, area)
            m = res["_mean_off1"]
            sched.step(m)
            print(f"  ep{ep:3d}  trloss={trloss:.4f}  val±1={m*100:6.2f}%  "
                  f"lr={opt.param_groups[0]['lr']:.1e}")
            if m > best_metric + 1e-4:
                best_metric = m
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= args.stop_early:
                print(f"  early stop (val±1 {args.stop_early}ep 정체)")
                break

    else:
        # ── 버그 재현: 매 에폭 best에서 재시작 + fresh Adam + 스텝감쇠 ──
        best_state = copy.deepcopy(new_model(area).state_dict())  # 초기 가중치
        for ep in range(1, args.epoch + 1):
            model = new_model(area)
            model.load_state_dict(best_state)            # best에서 재시작
            lr = args.lr * (0.1 ** (ep // max(1, args.epoch // 2)))  # 스텝감쇠
            opt = torch.optim.Adam(model.parameters(), lr=lr,
                                   betas=(0.9, 0.999), weight_decay=0)  # 매 에폭 재생성
            trloss = train_one_epoch(model, tr_loader, area, crit, opt)
            res = evaluate(model, va_loader, area)
            m = res["_mean_off1"]
            print(f"  ep{ep:3d}  trloss={trloss:.4f}  val±1={m*100:6.2f}%  "
                  f"lr={lr:.1e}")
            if m > best_metric + 1e-4:
                best_metric = m
                best_state = copy.deepcopy(model.state_dict())  # 개선 시에만 누적
                stale = 0
            else:
                stale += 1                                  # 미개선 → 다음 epoch 같은 best서 재시작
            if stale >= args.stop_early:
                print(f"  early stop (val±1 {args.stop_early}ep 정체)")
                break

    # best 체크포인트 저장 + test 평가
    torch.save({"model_state": best_state, "best_loss": best_metric},
               os.path.join(save_dir, "state_dict.bin"))
    final = new_model(area)
    final.load_state_dict(best_state)
    te = evaluate(final, te_loader, area)
    print(f"\n[TEST] area {area}  loop={args.loop}  "
          f"(best val±1={best_metric*100:.2f}%)")
    for diag, d in te.items():
        if diag.startswith("_"):
            continue
        print(f"  {diag:18s} n={d['n']:5d}  strict={d['strict']*100:6.2f}%  "
              f"±1={d['off1']*100:6.2f}%  MacroF1={d['f1']*100:6.2f}%")
    print(f"  {'mean':18s}         strict=  -     "
          f"±1={te['_mean_off1']*100:6.2f}%")
    return te


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"device={device}  loop={args.loop}  seed={args.seed}  "
          f"areas={args.areas}  aug={args.use_aug}")

    if not os.path.isdir(args.patch_path):
        print(f"[오류] patch_path 없음: {args.patch_path}  "
              f"(extract_patches.py 먼저 실행)")
        return

    tr, va, te = build_views(args)
    tr_loader = make_loader(tr, args, shuffle=True)
    va_loader = make_loader(va, args, shuffle=False)
    te_loader = make_loader(te, args, shuffle=False)

    for a in args.areas:
        if a not in AREA_LABELS:
            print(f"[skip] area {a} 는 분류 학습 대상 아님 (4/6은 3/5 공유)")
            continue
        train_area(a, args, tr_loader, va_loader, te_loader)


if __name__ == "__main__":
    main()
