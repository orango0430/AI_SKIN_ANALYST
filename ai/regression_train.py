"""회귀 모델 재학습 (clean rewrite).

기존 model.py 학습 루프의 치명적 문제(에폭마다 Adam 재생성 → momentum 리셋,
deepcopy 남발, 평균예측으로 수렴)를 버리고 표준 학습으로 재작성.

설계 (사용자 목표: '예측이 실제를 따라감' = 항목별 Pearson r↑, 점수 안 쏠림):
  - 손실 = (1 - CCC) + 0.5 * L1
      CCC(일치상관계수)는 상관 + bias + 분산을 한 번에 평가 →
      'r≈0/평균찍기/bias 쏠림'(이전 실패 3종)을 직접 벌점.
  - 옵티마이저(Adam) area당 1개 생성, 전 에폭 유지 (momentum 보존).
  - ReduceLROnPlateau (val 1-CCC 기준).
  - 스트리밍 Dataset: 패치를 미리 다 적재하지 않음(OOM 근본 해결).
    regression_data_loader의 crop/정규화/transform 재사용 → 추론과 동일 전처리.
  - split 시드 523 고정 → regression_eval.py와 동일 held-out 보장.
  - per-area ResNet50 (기존 추론/평가 체크포인트 규약 그대로 호환).
  - --overfit N: 소수표본 과적합 sanity. r→1 안 되면 그 항목은 구조적
    (사진에 신호 없음/라벨문제) → descope 근거. 본학습 전 필수.

사용:
  set PYTHONIOENCODING=utf-8
  # 1) sanity (먼저!): 항목별 학습 가능성 30분 판별
  python regression_train.py --img_path "D:\\...\\Training\\train_data" \
      --json_path "<라벨>" --areas "1,5" --overfit 32 --epoch 300
  # 2) 본학습 (sanity 통과 항목만)
  python regression_train.py --img_path "..." --json_path "..." \
      --areas "0,1,3,5,8" --epoch 60 --batch_size 32
"""
import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights

import pickle

from regression_data_loader import CustomDataset
from regression_extract_patches import sub_fold_of, patch_key, patch_path


# area -> (출력 차원, norm_reg 항목 순서). inference.REG_AREA_LABELS와 일치.
REG_AREAS = {
    0: ["pigmentation_count"],
    1: ["forehead_moisture", "forehead_elasticity_R2"],
    3: ["l_perocular_wrinkle_Ra"],
    5: ["l_cheek_moisture", "l_cheek_elasticity_R2", "l_cheek_pore"],
    8: ["chin_moisture", "chin_elasticity_R2"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="100%/1,2,3", type=str)
    p.add_argument("--img_path", default="dataset/img", type=str)
    p.add_argument("--json_path", default="dataset/label", type=str)
    p.add_argument("--output_dir", default="checkpoint2", type=str)
    p.add_argument("--areas", default="0,1,3,5,8", type=str,
                   help="학습할 회귀 area (쉼표). sanity는 1~2개 권장")
    p.add_argument("--epoch", default=60, type=int)
    p.add_argument("--batch_size", default=32, type=int,
                   help="CCC는 배치 통계 필요 → 32 이상 권장")
    p.add_argument("--lr", default=1e-4, type=float)
    p.add_argument("--num_workers", default=4, type=int)
    p.add_argument("--res", default=128, type=int)
    p.add_argument("--stop_early", default=12, type=int,
                   help="val 1-CCC 정체 epoch 수 (overfit 모드는 무시)")
    p.add_argument("--overfit", default=0, type=int,
                   help=">0이면 그 수만큼만 써서 과적합 sanity (train=val). "
                        "r→~1 안 되면 그 항목은 구조적 → descope 근거")
    p.add_argument("--mode", default="regression", type=str)  # loader 호환용
    p.add_argument("--seed", default=523, type=int)
    p.add_argument("--patch_dir", default="", type=str,
                   help="regression_extract_patches.py 출력 폴더. 지정 시 "
                        "원본 대신 사전추출 128패치 사용(에폭 수배↑, OOM해결, "
                        "전처리 완전 동일).")
    p.add_argument("--eval_only", action="store_true",
                   help="학습 안 함. 저장된 체크포인트로 test split만 평가 "
                        "(항목별 r/R²/CCC/MAE/bias). regression_eval.py의 "
                        "패치 버전.")
    return p.parse_args()


# ───────────────────────── 스트리밍 데이터셋 ─────────────────────────
class RegAreaStream(Dataset):
    """한 area의 (패치, 라벨)을 매 접근마다 디스크에서 생성 (사전적재 X).

    regression_data_loader.CustomDataset의 load_img/make_double/norm_reg/
    transform을 그대로 재사용 → 추론(inference.py)과 동일 전처리 보장.
    실패 샘플은 None 반환 → collate에서 제외.
    """

    def __init__(self, base, split_list, area_idx, args, cache=False):
        self.base = base
        self.items = list(split_list)
        self.area_idx = area_idx
        self.args = args
        # overfit sanity: 소수표본을 매 에폭 4K 재디코딩하면 느림 →
        # 첫 접근에 (패치,라벨) 캐시하고 이후 재사용. 본학습은 OOM이라 off.
        self.cache_on = cache
        self.cache = {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self.cache_on and i in self.cache:
            return self.cache[i]
        r = self._load(i)
        if self.cache_on:
            self.cache[i] = r
        return r

    def _load(self, i):
        v = self.items[i]
        equ_name = v["equ_name"]
        folder_path = v["folder_path"]
        img_name = v["img_name"]
        angle = img_name.split(".")[0].split("_")[-1]
        img = cv2.imread(os.path.join(folder_path, img_name))
        if img is None:
            return None
        try:
            red, _, _, meta, ori = self.base.load_img(
                img_name, angle, self.area_idx, equ_name, img, self.args
            )
        except Exception:
            return None  # bbox None 등 → load_img가 1 반환 → 언팩 실패

        # 128×128 고정 통일 (regression_data_loader/inference와 동일).
        # make_double 폐기 → BN train/eval 일치.
        patch = cv2.resize(ori, (self.args.res, self.args.res))

        if not isinstance(meta.get("equipment"), dict):
            return None
        label = np.asarray(self.base.norm_reg(meta, self.area_idx),
                            dtype=np.float32)  # Er → nan
        x = self.base.transform(Image.fromarray(patch))  # [3,H,W]
        return x, torch.from_numpy(label)


class PatchStream(Dataset):
    """사전추출 패치(regression_extract_patches.py 출력)에서 로드.

    원본 4K 디코딩 없음 → 에폭 수배 빠름. split은 RegAreaStream과 동일하게
    base의 시드523 random_split 결과(sample dict 리스트)를 그대로 받음 →
    train/val/test 동일 보장. 전처리도 추출 시 동일(128 resize) → 일관.
    """

    def __init__(self, base, split_list, area_idx, args, manifest):
        self.base = base
        self.items = list(split_list)
        self.area_idx = area_idx
        self.args = args
        self.manifest = manifest

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        v = self.items[i]
        equ_name = v["equ_name"]
        sub_fold = sub_fold_of(v["folder_path"])
        img_name = v["img_name"]
        key = patch_key(equ_name, sub_fold, img_name, self.area_idx)
        label = self.manifest.get(key)
        if label is None:
            return None
        pp = patch_path(self.args.patch_dir, equ_name, sub_fold,
                        img_name, self.area_idx)
        patch = cv2.imread(pp)
        if patch is None:
            return None
        # crop_img는 raw crop(가변크기) → 학습/추론과 동일하게 128 고정
        patch = cv2.resize(patch, (self.args.res, self.args.res))
        x = self.base.transform(Image.fromarray(patch))
        return x, torch.tensor(label, dtype=torch.float32)


def make_ds(base, items, area_idx, args, manifest, cache=False):
    """patch_dir 있으면 PatchStream, 없으면 RegAreaStream."""
    if args.patch_dir:
        return PatchStream(base, items, area_idx, args, manifest)
    return RegAreaStream(base, items, area_idx, args, cache=cache)


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    # 패치 shape이 샘플마다 다를 수 있어 stack 불가 → 리스트로 전달
    xs = [b[0] for b in batch]
    ys = torch.stack([b[1] for b in batch], dim=0)  # [B, out_dim]
    return xs, ys


# ───────────────────────── forward / 손실 ─────────────────────────
def forward_many(model, xs, device):
    """패치 리스트 → [N,out]. 전 패치 128×128 고정이라 단순 배치 forward
    (BatchNorm이 정상 배치를 봄, train/eval 일치).
    """
    xb = torch.stack(xs, 0).to(device)  # [N,3,128,128]
    return model(xb)


def ccc(pred, gt, eps=1e-6):
    """Concordance Correlation Coefficient (미분가능). 1에 가까울수록 좋음."""
    vp = pred - pred.mean()
    vg = gt - gt.mean()
    cov = (vp * vg).mean()
    ccc_val = (2 * cov) / (
        pred.var(unbiased=False) + gt.var(unbiased=False)
        + (pred.mean() - gt.mean()) ** 2 + eps
    )
    return ccc_val


def batch_loss(preds, gts):
    """preds/gts: [B, out_dim]. 차원별 (1-CCC)+0.5L1, nan 타깃 마스킹.

    유효표본 2개 미만 차원은 CCC 불가 → L1만.
    """
    out_dim = preds.shape[1]
    total, n_terms = 0.0, 0
    logs = []
    for d in range(out_dim):
        p, g = preds[:, d], gts[:, d]
        m = torch.isfinite(g)
        if m.sum() == 0:
            logs.append(None)
            continue
        pm, gm = p[m], g[m]
        l1 = torch.abs(pm - gm).mean()
        if m.sum() >= 2 and gm.var(unbiased=False) > 1e-8:
            c = ccc(pm, gm)
            term = (1 - c) + 0.5 * l1
            logs.append(float(c.detach()))
        else:
            term = l1
            logs.append(None)
        total = total + term
        n_terms += 1
    if n_terms == 0:
        return None, logs
    return total / n_terms, logs


@torch.no_grad()
def evaluate(model, loader, device, out_dim):
    model.eval()
    P, G = [], []
    for batch in loader:
        if batch is None:
            continue
        xs, ys = batch
        ob = forward_many(model, xs, device).cpu().numpy()  # [B,out]
        P.extend(ob)
        G.extend(ys.numpy())
    if not P:
        return None
    P = np.array(P, dtype=np.float64)
    G = np.array(G, dtype=np.float64)
    res = []
    for d in range(out_dim):
        g = G[:, d]
        p = P[:, d]
        msk = np.isfinite(g)
        if msk.sum() < 2:
            res.append(dict(n=int(msk.sum()), r=np.nan, ccc=np.nan, mae=np.nan))
            continue
        gg, pp = g[msk], p[msk]
        mae = float(np.mean(np.abs(pp - gg)))
        if pp.std() < 1e-9 or gg.std() < 1e-9:
            r = np.nan
        else:
            r = float(np.corrcoef(pp, gg)[0, 1])
        cov = np.mean((pp - pp.mean()) * (gg - gg.mean()))
        cccv = float(2 * cov / (pp.var() + gg.var()
                                + (pp.mean() - gg.mean()) ** 2 + 1e-6))
        res.append(dict(n=int(msk.sum()), r=r, ccc=cccv, mae=mae,
                        bias=float(pp.mean() - gg.mean()),
                        pstd=float(pp.std()), gstd=float(gg.std())))
    return res


def train_area(area_idx, base, args, device, manifest):
    out_dim = len(REG_AREAS[area_idx])
    labels = REG_AREAS[area_idx]
    print(f"\n{'='*70}\n[area {area_idx}] {labels}  (out_dim={out_dim})\n{'='*70}")

    # RegAreaStream 캐시는 단일프로세스에서만 유지 → overfit+원본일 때만 nw=0.
    # patch_dir면 이미 빠르고 캐시 불필요 → 워커 정상 사용.
    use_cache = (args.overfit > 0) and (not args.patch_dir)
    nw = 0 if use_cache else args.num_workers

    if args.overfit > 0:
        sub = list(base.train_list)[: args.overfit]
        tr_items, va_items = sub, sub               # 일부러 같은 표본
        print(f"  [overfit sanity] N={len(sub)} (train=val), 목표 r→~1")
        tr_ds = make_ds(base, tr_items, area_idx, args, manifest, cache=True)
        va_ds = tr_ds
    else:
        tr_items, va_items = list(base.train_list), list(base.val_list)
        print(f"  train={len(tr_items)}  val={len(va_items)}"
              f"{'  [patch]' if args.patch_dir else '  [원본디코딩]'}")
        tr_ds = make_ds(base, tr_items, area_idx, args, manifest, cache=False)
        va_ds = make_ds(base, va_items, area_idx, args, manifest, cache=False)

    tr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=nw, collate_fn=collate_skip_none)
    va = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=nw, collate_fn=collate_skip_none)

    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, out_dim)
    model = model.to(device)

    # 옵티마이저 1회 생성 후 전 에폭 유지 (기존 버그의 핵심 수정점)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=4)

    save_dir = os.path.join(args.output_dir, args.mode, args.name, str(area_idx))
    os.makedirs(save_dir, exist_ok=True)
    best = -1e9          # best mean CCC (높을수록 좋음)
    stale = 0

    for ep in range(1, args.epoch + 1):
        model.train()
        run, nb = 0.0, 0
        for batch in tr:
            if batch is None:
                continue
            xs, ys = batch
            ys = ys.to(device)
            preds = forward_many(model, xs, device)
            loss, _ = batch_loss(preds, ys)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += float(loss.detach())
            nb += 1

        res = evaluate(model, va, device, out_dim)
        if res is None:
            print(f"  ep{ep:3d} val 표본 없음 — area/경로 확인 필요")
            break
        mccc = np.nanmean([d["ccc"] for d in res])
        sched.step(1 - mccc)
        msg = "  ".join(
            f"{labels[i].split('_')[-1]}: r={d['r']:+.3f} ccc={d['ccc']:+.3f} "
            f"mae={d['mae']:.3f}" for i, d in enumerate(res)
        )
        print(f"  ep{ep:3d}  trloss={run/max(1,nb):.4f}  meanCCC={mccc:+.3f} | {msg}")

        if mccc > best + 1e-4:
            best = mccc
            stale = 0
            torch.save({"model_state": model.state_dict(),
                        "epoch": ep, "best_loss": float(best)},
                       os.path.join(save_dir, "state_dict.bin"))
        else:
            stale += 1
        if args.overfit == 0 and stale >= args.stop_early:
            print(f"  early stop (val CCC {args.stop_early}ep 정체). best={best:+.3f}")
            break

    print(f"[area {area_idx}] 종료. best meanCCC={best:+.3f} → {save_dir}/state_dict.bin")
    return best


def eval_area(area_idx, base, args, device, manifest):
    """저장된 체크포인트로 test split 평가 (regression_eval.py 패치판)."""
    out_dim = len(REG_AREAS[area_idx])
    labels = REG_AREAS[area_idx]
    ckpt = os.path.join(args.output_dir, args.mode, args.name,
                        str(area_idx), "state_dict.bin")
    if not os.path.isfile(ckpt):
        print(f"[area {area_idx}] 체크포인트 없음 skip: {ckpt}")
        return
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, out_dim)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"],
                          strict=False)
    model = model.to(device)

    te = list(base.test_list)
    ds = make_ds(base, te, area_idx, args, manifest, cache=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        collate_fn=collate_skip_none)
    res = evaluate(model, loader, device, out_dim)
    print(f"\n[area {area_idx}] {labels}  (test split)")
    if res is None:
        print("  표본 없음")
        return
    for i, d in enumerate(res):
        flag = ""
        if not np.isnan(d.get("r", np.nan)):
            if d["r"] < 0.4:
                flag = "  [!] r<0.4 (사용자 납득 미달)"
        print(f"  {labels[i]:24s} n={d['n']:5d}  r={d.get('r', float('nan')):+.3f}"
              f"  CCC={d['ccc']:+.3f}  MAE={d['mae']:.4f}"
              f"  bias={d.get('bias', float('nan')):+.3f}"
              f"  pσ={d.get('pstd', float('nan')):.3f}"
              f"  gσ={d.get('gstd', float('nan')):.3f}{flag}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  seed={args.seed}  overfit={args.overfit}"
          f"  patch_dir={args.patch_dir or '(원본)'}  eval_only={args.eval_only}")

    manifest = {}
    if args.patch_dir:
        mf = os.path.join(args.patch_dir, "manifest.pkl")
        assert os.path.isfile(mf), f"manifest 없음: {mf} (추출 먼저 실행)"
        with open(mf, "rb") as f:
            manifest = pickle.load(f)
        print(f"manifest 로드: {len(manifest):,} keys")

    base = CustomDataset(args)  # load_list + 시드 고정 random_split

    areas = [int(a) for a in args.areas.split(",") if a.strip() != ""]

    if args.eval_only:
        for a in areas:
            if a in REG_AREAS:
                eval_area(a, base, args, device, manifest)
        return

    summary = {}
    for a in areas:
        if a not in REG_AREAS:
            print(f"[skip] area {a} 는 회귀 대상 아님")
            continue
        summary[a] = train_area(a, base, args, device, manifest)

    print("\n" + "=" * 70)
    print("요약 (best meanCCC; overfit 모드면 r→~1 / CCC→~1 나와야 학습가능)")
    for a, v in summary.items():
        verdict = ""
        if args.overfit > 0:
            verdict = "  → 학습가능(파이프라인 OK)" if v > 0.8 else \
                      "  → [!] 소수표본도 과적합 실패: 구조적/데이터 문제 → descope 검토"
        print(f"  area {a} {REG_AREAS[a]}: meanCCC={v:+.3f}{verdict}")
    print("=" * 70)


if __name__ == "__main__":
    main()
