from collections import defaultdict
import datetime
import errno
import os
import cv2
import torch
import cv2
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
from tqdm import tqdm
from data_loader import class_num_list, area_naming

try:
    from sklearn.metrics import f1_score, confusion_matrix as sk_cm
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# 진단 항목 목록 (per-item 메트릭/CM/F1 집계)
DIAG_KEYS = ["pigmentation", "wrinkle", "pore", "dryness", "sagging", "forehead_wrinkle"]


def diag_name(label_key):
    """라벨 키에서 진단 항목명 추출. 'l_cheek_pore' -> 'pore', 'forehead_wrinkle' -> 'forehead_wrinkle'."""
    parts = label_key.split("_")
    last_two = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    return last_two if last_two in class_num_list else parts[-1]


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, batch_size=32):
        self.val = val
        self.sum += val * batch_size
        self.count += batch_size
        self.avg = self.sum / self.count

    def update_acc(self, val, num=1):
        self.val = val
        self.sum += val
        self.count += num
        self.avg = self.sum / self.count


def softmax(x):
    e_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)

    return e_x / torch.sum(e_x, dim=1).unsqueeze(dim=1)


def mkdir(path):
    # if it is the current folder, skip.
    # otherwise the original code will raise FileNotFoundError
    if path == "":
        return
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def adjust_learning_rate(optimizer, epoch, args):
    """
    Sets the learning rate to the initial LR decayed by x every y epochs
    x = 0.1, y = args.num_train_epochs/2.0 = 100
    """
    lr = args.lr * (0.1 ** (epoch // (args.epoch / 2.0)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_checkpoint(model, args, epoch, m_idx, best_loss):
    checkpoint_dir = os.path.join(args.output_dir, args.mode, args.name, str(m_idx))
    mkdir(checkpoint_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model_state": model_to_save.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
        },
        os.path.join(checkpoint_dir, "temp_file.bin"),
    )

    os.rename(
        os.path.join(checkpoint_dir, "temp_file.bin"),
        os.path.join(checkpoint_dir, "state_dict.bin"),
    )
    return checkpoint_dir


def resume_checkpoint(args, model, path):
    state_dict = torch.load(path)
    best_loss = state_dict["best_loss"]
    epoch = state_dict["epoch"]
    model.load_state_dict(state_dict["model_state"], strict=False)
    del state_dict
    args.load_epoch = epoch

    args.best_loss = best_loss

    return model


class Model_test(object):
    def __init__(self, args, model_list, testset_loader, logger):
        super(Model_test, self).__init__()
        self.args = args
        self.model_list = model_list
        self.logger = logger
        self.test_loader = testset_loader
        self.count = defaultdict(int)

        self.test_class_acc = {
            "sagging": AverageMeter(),
            "wrinkle": AverageMeter(),
            "pore": AverageMeter(),
            "pigmentation": AverageMeter(),
            "dryness": AverageMeter(),
        }
        # 진단 항목별 strict/±1 정확도 + macro-F1 + confusion matrix용 누적 버퍼
        self.metrics = {
            dig: {
                "strict": AverageMeter(),
                "off1": AverageMeter(),
                "preds": [],
                "targets": [],
            }
            for dig in DIAG_KEYS
        }
        # Expected-value(softmax 기대등급 반올림) 추론 누적 — argmax 대비
        # ±1/소수등급 recall 변화를 같은 모델로 비교(재학습 0).
        self.metrics_ev = {
            dig: {
                "strict": AverageMeter(),
                "off1": AverageMeter(),
                "preds": [],
                "targets": [],
            }
            for dig in DIAG_KEYS
        }
        # 부위별(라벨키 단위) 누적 — wrinkle처럼 미간/눈가가 합산돼 가려지는
        # 항목을 분리 측정. lazy 생성.
        self.area_metrics = {}
        self.test_regresion_mae = {
            "moisture": AverageMeter(),
            "wrinkle": AverageMeter(),
            "elasticity": AverageMeter(),
            "pore": AverageMeter(),
            "count": AverageMeter(),
        }
        # 회귀 부위별(area+항목) raw 정규화 pred/gt 누적 → MAE 외에 예측 std,
        # R²/상관, bias 계산용. MAE만으로 안 보이는 '평균값 회귀 붕괴' 진단.
        self.reg_metrics = {}

        self.equip_loss = {
            "0": {"count": 1},
            "1": {"moisture": 1, "elasticity": 1},
            "3": {"wrinkle": 1},
            "4": {"wrinkle": 1},
            "5": {"moisture": 1, "elasticity": 1, "pore": 1},
            "6": {"moisture": 1, "elasticity": 1, "pore": 1},
            "8": {"moisture": 1, "elasticity": 1},
        }
        self.criterion = (
            nn.CrossEntropyLoss() if self.args.mode == "class" else nn.L1Loss()
        )

        self.phase = None
        self.m_idx = 0
        self.model = None
        self.update_c = 0

    def choice(self, m_idx):
        if m_idx in [4, 6]:
            m_idx -= 1
            self.flag = True
        else:
            self.flag = False

        self.model = copy.deepcopy(self.model_list[m_idx])
        self.m_idx = m_idx

    def acc_avg(self, name):
        return round(self.test_class_acc[name].avg * 100, 2)

    def loss_avg(self, name):
        return round(self.test_regresion_mae[name].avg, 4)

    def print_total(self, iter):
        if self.args.mode == "class":
            self.logger.info(
                f"pigmentation: {self.acc_avg('pigmentation')}%(T: {self.test_class_acc['pigmentation'].sum} / F: {self.test_class_acc['pigmentation'].count - self.test_class_acc['pigmentation'].sum}) // wrinkle: {self.acc_avg('wrinkle')}%(T: {self.test_class_acc['wrinkle'].sum} / F: {self.test_class_acc['wrinkle'].count - self.test_class_acc['wrinkle'].sum}) // sagging: {self.acc_avg('sagging')}%(T: {self.test_class_acc['sagging'].sum} / F: {self.test_class_acc['sagging'].count - self.test_class_acc['sagging'].sum}) // pore: {self.acc_avg('pore')}%(T: {self.test_class_acc['pore'].sum} / F: {self.test_class_acc['pore'].count - self.test_class_acc['pore'].sum}) // dryness: {self.acc_avg('dryness')}%(T: {self.test_class_acc['dryness'].sum} / F: {self.test_class_acc['dryness'].count - self.test_class_acc['dryness'].sum})"
            )

            self.logger.info(
                f"[{iter} / {len(self.test_loader)}]Total Average Acc => {((self.acc_avg('pigmentation') + self.acc_avg('wrinkle') + self.acc_avg('sagging') + self.acc_avg('pore') + self.acc_avg('dryness') ) / 5):.2f}%"
            )

        else:
            self.logger.info(
                f"count: {self.loss_avg('count')} // moisture: {self.loss_avg('moisture')} // wrinkle: {self.loss_avg('wrinkle')} // elasticity: {self.loss_avg('elasticity')} // pore: {self.loss_avg('pore')}"
            )
            self.logger.info(
                f"[{iter} / {len(self.test_loader)}] Total Average MAE => {((self.loss_avg('count') + self.loss_avg('moisture') + self.loss_avg('wrinkle') +self.loss_avg('elasticity') + self.loss_avg('pore')) / 5):.3f}"
            )

        self.logger.info("============" * 15)

    def match_img(self, vis_img, img):
        col = self.num % self.col
        row = self.num // self.col
        vis_img[row * 256 : (row + 1) * 256, col * 256 : (col + 1) * 256] = img

        return vis_img

    def nan_detect(self, label):
        nan_list = list()
        for batch_idx, batch_data in enumerate(label):
            for value in batch_data:
                if not torch.isfinite(value):
                    nan_list.append(batch_idx)
        return nan_list

    def get_test_loss(self, pred_p, label, area_num, patch_list):
        patch_list[area_num].append(dict())
        for idx, name in enumerate(self.equip_loss[area_num]):
            dig = name.split("_")[-1]
            gt = label[:, idx: idx + 1]
            if torch.isnan(gt): 
                continue
            
            pred = pred_p[:, idx: idx + 1]
            self.test_regresion_mae[name].update(
                self.criterion(pred, gt).item(), batch_size=pred.shape[0]
            )
            self.logger.info(
                patch_list[area_num][2][0]
                + f"({dig})"
                + f"==> Pred: {pred.item():.3f}  /  Gt: {gt.item():.3f}  ==> MAE: {self.criterion(pred, gt).item():.3f}"
            )
            
            self.count[f"{area_num}_{dig}"] += 1

            # 부위+항목 단위 raw(정규화) pred/gt 누적 (denorm 전에 캡처)
            rkey = f"{area_num}_{name}"
            rm = self.reg_metrics.setdefault(rkey, {"preds": [], "gts": []})
            rm["preds"].append(float(pred.item()))
            rm["gts"].append(float(gt.item()))

            if dig == "moisture":
                gt, pred = gt * 100, pred * 100
            elif dig == "count":
                gt, pred = gt * 350, pred * 350
            elif dig == "pore":
                gt, pred = gt * 3000, pred * 3000
            patch_list[area_num][3][dig] = [round(gt.item(), 3), round(pred.item(), 3)]

        return patch_list

    def get_test_acc(self, pred, label, patch_list, area_num):
        gt = (
            torch.tensor(
                np.array([label[value].detach().cpu().numpy() for value in label])
            )
            .permute(1, 0)
            .to(device)
        )
        num = 0
        patch_list[area_num].append(dict())
        for idx, area_name in enumerate(label):
            dig = diag_name(area_name)              # 'wrinkle' or 'forehead_wrinkle' 등 분리
            class_num = class_num_list[dig]
            legacy_key = "wrinkle" if dig == "forehead_wrinkle" else dig  # 기존 5개 키 호환

            logits_slice = pred[:, num : num + class_num]
            pred_l = torch.argmax(logits_slice, dim=1)
            # Expected-value: 등급 기대값 반올림 (멀리 튀는 오차 완화 → ±1↑)
            probs = torch.softmax(logits_slice, dim=1)
            grades = torch.arange(class_num, device=probs.device, dtype=probs.dtype)
            pred_ev = torch.clamp(
                (probs * grades).sum(dim=1).round().long(), min=0, max=class_num - 1
            )
            num += class_num

            # GT 클램프 (sagging 등급6->5 머지, 그 외도 안전 클램프)
            gt_l = gt[:, idx].long()
            if dig == "sagging":
                gt_l = torch.clamp(gt_l, max=class_num - 1)
            gt_l = torch.clamp(gt_l, min=0, max=class_num - 1)

            diff = (pred_l - gt_l).abs()
            strict = (diff == 0).sum().item()
            off1 = (diff <= 1).sum().item()
            n = pred_l.shape[0]

            # 기존 ±1 호환 (5개 키로 합산: forehead_wrinkle -> wrinkle)
            self.test_class_acc[legacy_key].update_acc(off1, n)

            # 신규: 6개 항목별 strict / ±1 / macro-F1 / CM 누적
            self.metrics[dig]["strict"].update_acc(strict, n)
            self.metrics[dig]["off1"].update_acc(off1, n)
            self.metrics[dig]["preds"].extend(pred_l.cpu().numpy().tolist())
            self.metrics[dig]["targets"].extend(gt_l.cpu().numpy().tolist())

            # Expected-value 누적 (동일 모델, 추론 방식만 다름)
            diff_ev = (pred_ev - gt_l).abs()
            self.metrics_ev[dig]["strict"].update_acc(
                (diff_ev == 0).sum().item(), n
            )
            self.metrics_ev[dig]["off1"].update_acc(
                (diff_ev <= 1).sum().item(), n
            )
            self.metrics_ev[dig]["preds"].extend(pred_ev.cpu().numpy().tolist())
            self.metrics_ev[dig]["targets"].extend(gt_l.cpu().numpy().tolist())

            # 부위별(라벨키) 분리 누적 — 예: l_perocular_wrinkle 단독 측정
            am = self.area_metrics.setdefault(
                area_name,
                {"strict": AverageMeter(), "off1": AverageMeter(),
                 "preds": [], "targets": [], "ncls": class_num},
            )
            am["strict"].update_acc(strict, n)
            am["off1"].update_acc(off1, n)
            am["preds"].extend(pred_l.cpu().numpy().tolist())
            am["targets"].extend(gt_l.cpu().numpy().tolist())

            patch_list[area_num][3][dig] = [int(gt_l.item()), int(pred_l.item())]
            self.logger.info(
                patch_list[area_num][2][0]
                + f"({dig})"
                + f"==> Pred: {pred_l.item()}  /  Gt: {gt_l.item()}  ==> off1={off1==n} strict={strict==n}"
            )

            self.count[f"{area_num}_{dig}"] += 1

        return patch_list

    # ----- 평가 종합 보고 -----

    def _compute_macro_f1(self, preds, targets, num_classes):
        f1s = []
        for c in range(num_classes):
            tp = sum(1 for p, t in zip(preds, targets) if p == c and t == c)
            fp = sum(1 for p, t in zip(preds, targets) if p == c and t != c)
            fn = sum(1 for p, t in zip(preds, targets) if p != c and t == c)
            if tp == 0:
                f1s.append(0.0)
                continue
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1s.append(2 * prec * rec / (prec + rec))
        return sum(f1s) / max(1, len(f1s))

    def _compute_cm(self, preds, targets, num_classes):
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for p, t in zip(preds, targets):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                cm[t, p] += 1
        return cm

    def _format_cm(self, cm, dig):
        n = cm.shape[0]
        lines = [f"  Confusion matrix ({dig}, rows=GT, cols=Pred):"]
        lines.append("         " + " ".join(f"P{c:<5}" for c in range(n)))
        for r in range(n):
            row = f"  GT{r:<3}: " + " ".join(f"{cm[r, c]:>5,}" for c in range(n))
            lines.append(row)
        return "\n".join(lines)

    def final_report(self):
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("Final Evaluation Report")
        self.logger.info("=" * 80)

        strict_avgs, off1_avgs, macro_f1s = [], [], []
        ev_strict_avgs, ev_off1_avgs, ev_f1s = [], [], []

        for dig in DIAG_KEYS:
            m = self.metrics[dig]
            if m["off1"].count == 0:
                continue

            strict_pct = m["strict"].avg * 100
            off1_pct = m["off1"].avg * 100
            num_cls = class_num_list[dig]

            if HAS_SKLEARN:
                f1 = f1_score(m["targets"], m["preds"], average="macro",
                              labels=list(range(num_cls)), zero_division=0) * 100
                cm = sk_cm(m["targets"], m["preds"], labels=list(range(num_cls)))
            else:
                f1 = self._compute_macro_f1(m["preds"], m["targets"], num_cls) * 100
                cm = self._compute_cm(m["preds"], m["targets"], num_cls)

            self.logger.info("")
            self.logger.info(f"[{dig}]  N = {m['off1'].count:,}  (classes = {num_cls})")
            self.logger.info(
                f"  Strict Top-1: {strict_pct:6.2f}%   "
                f"±1 Top-1: {off1_pct:6.2f}%   "
                f"Macro-F1: {f1:6.2f}%"
            )

            # per-class recall
            per_class_recall = []
            for c in range(num_cls):
                row_sum = cm[c].sum()
                per_class_recall.append((cm[c, c] / row_sum * 100) if row_sum > 0 else 0.0)
            recall_str = "  per-class recall: " + ", ".join(
                f"등급{c}={r:5.1f}%" for c, r in enumerate(per_class_recall)
            )
            self.logger.info(recall_str)
            self.logger.info(self._format_cm(cm, dig))

            strict_avgs.append(strict_pct)
            off1_avgs.append(off1_pct)
            macro_f1s.append(f1)

            # ── Expected-value 추론 비교 (동일 모델, argmax → 기대등급 반올림) ──
            mev = self.metrics_ev[dig]
            if mev["off1"].count > 0:
                ev_strict = mev["strict"].avg * 100
                ev_off1 = mev["off1"].avg * 100
                if HAS_SKLEARN:
                    ev_f1 = f1_score(mev["targets"], mev["preds"], average="macro",
                                     labels=list(range(num_cls)), zero_division=0) * 100
                    ev_cm = sk_cm(mev["targets"], mev["preds"],
                                  labels=list(range(num_cls)))
                else:
                    ev_f1 = self._compute_macro_f1(
                        mev["preds"], mev["targets"], num_cls) * 100
                    ev_cm = self._compute_cm(mev["preds"], mev["targets"], num_cls)
                ev_recall = []
                for c in range(num_cls):
                    rs = ev_cm[c].sum()
                    ev_recall.append((ev_cm[c, c] / rs * 100) if rs > 0 else 0.0)
                self.logger.info(
                    f"  [E-value] Strict: {ev_strict:6.2f}%   "
                    f"±1: {ev_off1:6.2f}% (Δ{ev_off1 - off1_pct:+5.2f})   "
                    f"Macro-F1: {ev_f1:6.2f}% (Δ{ev_f1 - f1:+5.2f})"
                )
                self.logger.info(
                    "  [E-value] per-class recall: "
                    + ", ".join(f"등급{c}={r:5.1f}%"
                                for c, r in enumerate(ev_recall))
                )
                ev_strict_avgs.append(ev_strict)
                ev_off1_avgs.append(ev_off1)
                ev_f1s.append(ev_f1)

        self.logger.info("")
        self.logger.info("-" * 80)
        if strict_avgs:
            self.logger.info(
                f"Average over {len(strict_avgs)} items   "
                f"Strict: {np.mean(strict_avgs):6.2f}%   "
                f"±1: {np.mean(off1_avgs):6.2f}%   "
                f"Macro-F1: {np.mean(macro_f1s):6.2f}%"
            )
        if ev_off1_avgs:
            self.logger.info(
                f"Average [E-value]            "
                f"Strict: {np.mean(ev_strict_avgs):6.2f}%   "
                f"±1: {np.mean(ev_off1_avgs):6.2f}%   "
                f"Macro-F1: {np.mean(ev_f1s):6.2f}%"
            )
        # ── 부위별(라벨키) 분리 리포트 — 합산 지표에 가려지는 부위 측정 ──
        # wrinkle 계열(glabellus/l_perocular/r_perocular)만 분리 출력.
        wrinkle_keys = sorted(
            k for k in self.area_metrics if diag_name(k) == "wrinkle"
        )
        if wrinkle_keys:
            self.logger.info("")
            self.logger.info("-" * 80)
            self.logger.info("Per-area breakdown (wrinkle 계열)")
            for key in wrinkle_keys:
                am = self.area_metrics[key]
                if am["off1"].count == 0:
                    continue
                ncls = am["ncls"]
                if HAS_SKLEARN:
                    f1 = f1_score(am["targets"], am["preds"], average="macro",
                                  labels=list(range(ncls)), zero_division=0) * 100
                    cm = sk_cm(am["targets"], am["preds"], labels=list(range(ncls)))
                else:
                    f1 = self._compute_macro_f1(am["preds"], am["targets"], ncls) * 100
                    cm = self._compute_cm(am["preds"], am["targets"], ncls)
                self.logger.info("")
                self.logger.info(f"[{key}]  N = {am['off1'].count:,}  (classes = {ncls})")
                self.logger.info(
                    f"  Strict Top-1: {am['strict'].avg * 100:6.2f}%   "
                    f"±1 Top-1: {am['off1'].avg * 100:6.2f}%   "
                    f"Macro-F1: {f1:6.2f}%"
                )
                pcr = []
                for c in range(ncls):
                    rs = cm[c].sum()
                    pcr.append((cm[c, c] / rs * 100) if rs > 0 else 0.0)
                self.logger.info(
                    "  per-class recall: "
                    + ", ".join(f"등급{c}={r:5.1f}%" for c, r in enumerate(pcr))
                )
                self.logger.info(self._format_cm(cm, key))

        if not HAS_SKLEARN:
            self.logger.info("(sklearn 미설치 — fallback 사용. `pip install scikit-learn` 권장)")
        self.logger.info("=" * 80)

    def reg_final_report(self):
        """회귀 부위별 진단 — MAE만으로 안 보이는 '평균값 회귀 붕괴' 노출.

        per-item: N / MAE / RMSE / 예측·정답 std / Pearson r / R² / bias.
        - 예측 std ≪ 정답 std  → 모델이 거의 상수만 뱉음(붕괴)
        - R² ≤ 0, |r| 낮음     → 예측력 없음(평균 찍기)
        - bias(예측평균-정답평균) 큰 음/양 → 점수가 한쪽으로 쏠림
          (사용자 점수가 터무니없이 낮/높게 나오는 직접 원인)
        """
        area_kr = {"0": "전체", "1": "forehead", "3": "l_perocular",
                   "4": "r_perocular", "5": "l_cheek", "6": "r_cheek",
                   "8": "chin"}
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("Regression Final Report  (raw 정규화 0~1 기준)")
        self.logger.info("=" * 80)

        for rkey in sorted(self.reg_metrics):
            rm = self.reg_metrics[rkey]
            p = np.array(rm["preds"], dtype=np.float64)
            g = np.array(rm["gts"], dtype=np.float64)
            n = len(p)
            if n == 0:
                continue
            anum, name = rkey.split("_", 1)
            label = f"{area_kr.get(anum, anum)}_{name}"

            mae = float(np.mean(np.abs(p - g)))
            rmse = float(np.sqrt(np.mean((p - g) ** 2)))
            p_std, g_std = float(p.std()), float(g.std())
            p_mean, g_mean = float(p.mean()), float(g.mean())
            bias = p_mean - g_mean

            ss_res = float(np.sum((g - p) ** 2))
            ss_tot = float(np.sum((g - g_mean) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
            if p_std > 1e-9 and g_std > 1e-9:
                r = float(np.corrcoef(p, g)[0, 1])
            else:
                r = float("nan")

            flags = []
            if g_std > 1e-9 and p_std < 0.5 * g_std:
                flags.append("예측분산붕괴")
            if not np.isnan(r2) and r2 < 0.1:
                flags.append("R²≈0(평균찍기)")
            if abs(bias) > 0.10:
                flags.append(f"bias{'↓' if bias < 0 else '↑'}(점수쏠림)")
            flag_str = ("  [!] " + ", ".join(flags)) if flags else ""

            self.logger.info("")
            self.logger.info(f"[{label}]  N = {n:,}{flag_str}")
            self.logger.info(
                f"  MAE: {mae:.4f}  RMSE: {rmse:.4f}   "
                f"pred(μ={p_mean:.3f}, σ={p_std:.3f})  "
                f"gt(μ={g_mean:.3f}, σ={g_std:.3f})"
            )
            self.logger.info(
                f"  Pearson r: {r:6.3f}   R²: {r2:6.3f}   "
                f"bias(pred-gt): {bias:+.3f}"
            )

        self.logger.info("")
        self.logger.info("-" * 80)
        self.logger.info(
            "해석: 예측σ가 정답σ보다 많이 작거나 R²≤0이면 그 항목은 "
            "사진과 무관하게 평균만 출력 → 캘리브레이션해도 사용자 점수가 "
            "중앙에 뭉치거나 bias 방향으로 쏠림. bias 음수 = 점수 과소(낮게)."
        )
        self.logger.info("=" * 80)

    def test(self, model_num_class, data_loader):
        for iter, patch_list in enumerate(data_loader):
            for model_idx in range(len(model_num_class)):
                if np.isnan(model_num_class[model_idx]):
                    continue

                self.choice(model_idx)
                self.model = self.model_list[self.m_idx]
                self.model.eval()

                data_loader = self.test_loader
                area_num = str(self.m_idx + 1) if self.flag else str(self.m_idx)

                # 이 sample이 해당 area를 안 가지는 경우 skip (angle 따라 perocular 등 빠짐)
                if area_num not in patch_list:
                    continue

                if type(patch_list[area_num][1]) == torch.Tensor:
                    label = patch_list[area_num][1].to(device)
                else:
                    for name in patch_list[area_num][1]:
                        patch_list[area_num][1][name] = patch_list[area_num][1][
                            name
                        ].to(device)
                    label = patch_list[area_num][1]

                if label == {}:
                    continue        ## 눈가/볼 영역이 없는 경우
                
                img = patch_list[area_num][0].to(device)

                if area_num in [4, 6]:
                    img = torch.flip(img, dims=[3])

                if img.shape[-1] > 128:
                    img_l = img[:, :, :, :128]
                    img_r = img[:, :, :, 128:]
                    pred = self.model.to(device)(img_l)
                    pred = self.model.to(device)(img_r) + pred

                elif img.shape[-2] > 128:
                    img_l = img[:, :, :128, :]
                    img_r = img[:, :, 128:, :]
                    pred = self.model.to(device)(img_l)
                    pred = self.model.to(device)(img_r) + pred

                else:
                    pred = self.model.to(device)(img)

                if self.args.mode == "class":
                    _ = self.get_test_acc(pred, label, patch_list, area_num)

                else:
                    _ = self.get_test_loss(pred, label.to(device), area_num, patch_list)
            self.print_total(iter)

        if self.args.mode == "class":
            self.final_report()
        else:
            self.reg_final_report()

        if self.args.log: [self.logger.info(f"{key} => {self.count[key]} 장") for key in self.count]
