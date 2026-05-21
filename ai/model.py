import errno
import gc
import os
import cv2
import torch
import cv2
import torch.nn as nn
import numpy as np
import copy
import torch.nn.functional as F
from data_loader import class_num_list, area_naming

try:
    from sklearn.metrics import f1_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


DIAG_KEYS = ["pigmentation", "wrinkle", "pore", "dryness", "sagging", "forehead_wrinkle"]


# label_distribution.txt의 'balanced' 가중치에 sqrt를 적용해 극단치 완화
# 원본: pore 등급5=14.3, dryness 양극단=8.58 등 너무 극단적이라 학습 진동 발생
# sqrt 적용 후: pore 등급5=3.78, dryness 양극단=2.93 → 안정성 ↑
# (sagging은 등급5+6 머지 후 재계산 기준)
CLASS_WEIGHTS = {
    "dryness":          [2.929, 1.127, 0.572, 1.039, 2.929],
    # forehead_wrinkle / wrinkle: 모델이 고등급(4~6) 과소예측 + 저등급(0~1)
    # 과다예측 → 다수 등급(1,2) 가중치 ↓, 고등급(3~6) 가중치 ↑로 재조정.
    # (학습 분포: fw 4.2/37.2/23.4/16.7/7.6/6.4/4.6%)
    "forehead_wrinkle": [1.20, 0.40, 0.70, 1.20, 2.20, 2.80, 3.50],
    "pigmentation":     [1.384, 0.682, 0.837, 0.885, 1.503, 2.330],
    "pore":             [2.441, 0.965, 0.524, 1.153, 1.824, 3.781],
    "sagging":          [0.606, 0.909, 1.135, 1.151, 1.744, 2.183],
    # wrinkle 분포: 11.7/36.8/13.8/13.3/7.6/9.1/7.8%
    "wrinkle":          [1.00, 0.45, 0.95, 1.10, 1.90, 1.90, 2.10],
}


def diag_name(label_key):
    """라벨 키(예: 'l_cheek_pore', 'forehead_wrinkle')에서 진단 항목명 추출."""
    parts = label_key.split("_")
    last_two = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    return last_two if last_two in class_num_list else parts[-1]
    
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
    temp_path = os.path.join(checkpoint_dir, "temp_file.bin")
    final_path = os.path.join(checkpoint_dir, "state_dict.bin")
    
    torch.save(
        {
            "model_state": model_to_save.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
        },
        temp_path,
    )

    # 이미 파일이 존재하면 삭제
    if os.path.exists(final_path):
        os.remove(final_path)
    
    os.rename(temp_path, final_path)
    return checkpoint_dir



def resume_checkpoint(args, model, path):
    state_dict = torch.load(path, map_location=device)
    best_loss = state_dict["best_loss"]
    epoch = state_dict["epoch"]
    model.load_state_dict(state_dict["model_state"], strict=False)
    del state_dict
    args.load_epoch = epoch

    args.best_loss = best_loss

    return model


class Model(object):
    def __init__(
        self,
        args,
        model_list,
        train_loader,
        valid_loader,
        logger,
        writer,
    ):
        super(Model, self).__init__()
        self.args = args
        self.model_list = model_list
        self.temp_model_list = [None for _ in range(9)]
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.best_loss = args.best_loss

        self.writer = writer
        self.logger = logger

        self.train_loss = [AverageMeter() for _ in range(9)]
        self.val_loss = [AverageMeter() for _ in range(9)]
        # best 모델 선정 기준: idx별 valid ±1 accuracy (높을수록 좋음)
        self.val_acc = [AverageMeter() for _ in range(9)]

        self.keep_acc = {
            "sagging": 0,
            "wrinkle": 0,
            "pore": 0,
            "pigmentation": 0,
            "dryness": 0,
        }

        self.keep_mae = {
            "moisture": 0,
            "wrinkle": 0,
            "elasticity": 0,
            "pore": 0,
            "count": 0,
        }

        self.equip_loss = {
            "0": {"count": 1},
            "1": {"moisture": 1, "elasticity": 1},
            "3": {"wrinkle": 1},
            "4": {"wrinkle": 1},
            "5": {"moisture": 1, "elasticity": 1, "pore": 1},
            "6": {"moisture": 1, "elasticity": 1, "pore": 1},
            "8": {"moisture": 1, "elasticity": 1},
        }

        self.epoch = 0
        # Valid 메트릭 누적 (epoch별 reset)
        self.metrics = {
            dig: {
                "strict": AverageMeter(),
                "off1": AverageMeter(),
                "preds": [],
                "targets": [],
            }
            for dig in DIAG_KEYS
        }
        # Multi-class: 진단 항목별로 다른 class_weight를 가진 CE
        if self.args.mode == "class":
            self.criterion_dict = {
                name: nn.CrossEntropyLoss(
                    weight=torch.tensor(w, dtype=torch.float, device=device)
                )
                for name, w in CLASS_WEIGHTS.items()
            }
            self.criterion = self.criterion_dict  # regression 모드와 호환을 위해
        else:
            self.criterion = nn.L1Loss()

        self.phase = None
        self.m_idx = 0
        self.model = None
        self.update_c = 0

        # 첫 epoch valid 시 test_value가 아직 없으면 AttributeError 나므로 한 번 초기화
        self.reset_log(mode=args.mode)

    def choice(self, m_idx):
        if m_idx in [4, 6]:
            m_idx -= 1
            self.flag = True
        else:
            self.flag = False

        self.model = copy.deepcopy(self.model_list[m_idx])
        self.m_idx = m_idx

    def update_m(self, model_num_class):
        count = 0

        is_reg = self.args.mode != "class"

        for idx, value in enumerate(model_num_class):
            print("update_m : " + str(idx) + "\n")
            if np.isnan(value) or idx in [4, 6]:
                continue

            if is_reg:
                # 회귀: valid L1 loss 최소화 기준 (낮을수록 좋음).
                # regression 분기는 get_test_acc 미호출 → val_acc=0이라 못 씀.
                if self.val_loss[idx].count == 0:
                    continue  # 이 idx는 이번 epoch valid가 안 돎
                current = float(self.val_loss[idx].avg)
                improved = current < self.best_loss[idx]
            else:
                # 분류: valid ±1 accuracy 기준 (높을수록 좋음).
                # 이번 epoch 학습/검증 안 한 부위(--areas 필터)는 temp_model이
                # None이라 저장 시 크래시 → val_acc.count==0이면 skip.
                if self.val_acc[idx].count == 0 or self.temp_model_list[idx] is None:
                    continue
                current = float(self.val_acc[idx].avg)
                improved = current > self.best_loss[idx]

            if improved:
                self.best_loss[idx] = round(current, 4)
                self.model_list[idx] = copy.deepcopy(self.temp_model_list[idx])
                save_checkpoint(
                    self.model_list[idx], self.args, self.epoch, idx, self.best_loss
                )
                count += 1

        if count == 0:
            self.update_c += 1

        else:
            self.update_c = 0

    def update_e(self, epoch):
        self.epoch = epoch

    def acc_avg(self, name):
        return round(self.test_value[name].avg * 100, 2)

    def loss_avg(self, name):
        return round(self.test_value[name].avg, 4)

    def up_and_down(self, name, color="\033[95m", c_color="\033[0m"):
        if self.args.mode == "class":
            sub = (self.test_value[name].avg * 100) - self.keep_acc[name]
            value = round(sub, 2)
            result = (
                f"{color}+{value}{c_color}%"
                if value > 0
                else "No change"
                if value == 0
                else f"{color}{value}{c_color}%"
            )
        else:
            sub = (self.test_value[name].avg) - self.keep_mae[name]
            value = round(sub, 4)
            result = (
                f"{color}+{value}{c_color}"
                if value > 0
                else "No change"
                if value == 0
                else f"{color}{value}{c_color}"
            )

        return result


    def print_loss(self, iteration):
        dataloader_len = (
            len(self.train_loader) if self.phase == "train" else len(self.valid_loader)
        )

        if iteration == dataloader_len - 1:
            print(
                f"\rEpoch: {self.epoch} [{self.phase}][{area_naming[f'{self.area_num}']}][{iteration}/{dataloader_len}] ---- >  loss: {(self.train_loss[self.m_idx].avg if self.phase == 'train' else self.val_loss[self.m_idx].avg):.04f}"
            )

            self.writer.add_scalar(
                f"{self.phase}/{area_naming[f'{self.m_idx}']}",
                self.train_loss[self.m_idx].avg
                if self.args.mode == "class"
                else self.val_loss[self.m_idx].avg,
                self.epoch,
            )
            self.temp_model_list[self.m_idx] = self.model

        else:
            print(
                f"\rEpoch: {self.epoch} [{self.phase}][{area_naming[f'{self.area_num}']}][{iteration}/{dataloader_len}] ---- >  loss: {self.train_loss[self.m_idx].avg if self.phase == 'train' else self.val_loss[self.m_idx].avg:.04f}",
                end="",
            )

    def stop_early(self):
        if self.update_c > self.args.stop_early:
            return True

    def reset_log(self, mode):
        self.train_loss = [AverageMeter() for _ in range(9)]
        self.val_loss = [AverageMeter() for _ in range(9)]
        self.val_acc = [AverageMeter() for _ in range(9)]
        self.test_value = (
            {
                "sagging": AverageMeter(),
                "wrinkle": AverageMeter(),
                "pore": AverageMeter(),
                "pigmentation": AverageMeter(),
                "dryness": AverageMeter(),
            }
            if mode == "class"
            else {
                "moisture": AverageMeter(),
                "wrinkle": AverageMeter(),
                "elasticity": AverageMeter(),
                "pore": AverageMeter(),
                "count": AverageMeter(),
            }
        )
        # epoch 단위 valid 메트릭 reset
        self.metrics = {
            dig: {
                "strict": AverageMeter(),
                "off1": AverageMeter(),
                "preds": [],
                "targets": [],
            }
            for dig in DIAG_KEYS
        }

    def class_loss(self, pred, label):
        # Multi-class CE: 진단 항목별 출력 슬라이스 + 항목별 weighted CE
        # - sagging 등급6은 단 13장이라 등급5로 클램프 (6-class)
        # - 라벨이 dict인데 각 키가 진단 부위(예: 'l_cheek_pore') → diag_name으로 항목명 추출
        # - collate_fn이 누락/invalid를 -1 sentinel로 마킹 → mask로 제외
        num = 0
        loss = 0
        batch_size = 0

        for name in label:
            diag = diag_name(name)
            class_num = class_num_list[diag]
            criterion = self.criterion_dict[diag]

            target = label[name].long().to(device)
            mask = target >= 0
            if mask.sum() == 0:
                num += class_num
                continue

            target_v = target[mask]
            if diag == "sagging":
                target_v = torch.clamp(target_v, max=class_num - 1)
            target_v = torch.clamp(target_v, min=0, max=class_num - 1)

            logits = pred[mask, num : num + class_num]
            num += class_num
            loss = loss + criterion(logits, target_v)
            batch_size = target_v.shape[0]

        if batch_size == 0:
            return loss

        self.train_loss[self.m_idx].update(
            loss, batch_size=batch_size
        ) if self.phase == "train" else self.val_loss[self.m_idx].update(
            loss, batch_size=batch_size
        )

        return loss

    def regression(self, pred, label):
        loss = self.criterion(pred, label.to(device))
        self.train_loss[self.m_idx].update(
            loss, batch_size=pred.shape[0]
        ) if self.phase == "train" else self.val_loss[self.m_idx].update(
            loss, batch_size=pred.shape[0]
        )

        return loss

    def match_img(self, vis_img, img):
        col = self.num % 3
        row = self.num // 3
        vis_img[row * 256 : (row + 1) * 256, col * 256 : (col + 1) * 256] = img

        return vis_img

    def save_img(self, iteration, patch_list):
        if self.epoch == 0 and self.m_idx == 1:
            if self.args.mode == "class":
                vis_img = np.zeros([256 * 5, 256 * 3, 3])
                self.num_patch = len(patch_list) + 7
            else:
                vis_img = np.zeros([256 * 4, 256 * 3, 3])
                self.num_patch = len(patch_list) + 4

            self.num = 0
            for area_num in patch_list:
                img = patch_list[area_num][0][0].permute(1, 2, 0).numpy().copy()

                if img.shape[1] > 128:
                    l_img = img[:, :128]
                    l_img = cv2.resize(l_img, (256, 256))
                    cv2.putText(
                        l_img,
                        f"{area_naming[str(int(area_num))]}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 244, 0),
                        2,
                    )

                    vis_img = self.match_img(vis_img, l_img)
                    self.num += 1
                    r_img = img[:, 128:]
                    r_img = cv2.resize(r_img, (256, 256))
                    cv2.putText(
                        r_img,
                        f"{area_naming[str(int(area_num))]}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 244, 0),
                        2,
                    )
                    vis_img = self.match_img(vis_img, r_img)
                    self.num += 1

                elif img.shape[0] > 128:
                    l_img = img[:128]
                    l_img = cv2.resize(l_img, (256, 256))
                    cv2.putText(
                        l_img,
                        f"{area_naming[str(int(area_num))]}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 244, 0),
                        2,
                    )

                    vis_img = self.match_img(vis_img, l_img)
                    self.num += 1
                    r_img = img[128:]
                    r_img = cv2.resize(r_img, (256, 256))
                    cv2.putText(
                        r_img,
                        f"{area_naming[str(int(area_num))]}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 244, 0),
                        2,
                    )
                    vis_img = self.match_img(vis_img, r_img)
                    self.num += 1

                else:
                    img = img[:, :128]
                    img = cv2.resize(img, (256, 256))
                    cv2.putText(
                        img,
                        f"{area_naming[str(int(area_num))]}",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 244, 0),
                        2,
                    )

                    vis_img = self.match_img(vis_img, img)
                    self.num += 1

            mkdir(f"vis/{self.args.mode}/{self.args.name}")
            cv2.imwrite(
                f"vis/{self.args.mode}/{self.args.name}/{iteration}.jpg",
                # f"vis/{self.args.mode}/{self.args.name}/Input.jpg",
                vis_img * 255,
            )

    def nan_detect(self, label):
        nan_list = list()
        for batch_idx, batch_data in enumerate(label):
            # NaN이 하나라도 있는 경우 추가
            if torch.isnan(batch_data).any():
                nan_list.append(batch_idx)
        return nan_list

    def get_test_acc(self, pred, label):
        gt = (
            torch.tensor(
                np.array([label[value].detach().cpu().numpy() for value in label])
            )
            .permute(1, 0)
            .to(device)
        )
        num = 0

        for idx, area_name in enumerate(label):
            dig = diag_name(area_name)
            class_num = class_num_list[dig]

            gt_l = gt[:, idx].long()
            mask = gt_l >= 0
            if mask.sum() == 0:
                num += class_num
                continue

            pred_l = torch.argmax(pred[mask, num : num + class_num], dim=1)
            num += class_num

            gt_l = gt_l[mask]
            if dig == "sagging":
                gt_l = torch.clamp(gt_l, max=class_num - 1)
            gt_l = torch.clamp(gt_l, min=0, max=class_num - 1)

            diff = (pred_l - gt_l).abs()
            strict = (diff == 0).sum().item()
            off1 = (diff <= 1).sum().item()
            n = pred_l.shape[0]

            # 기존 5개 키 호환 (forehead_wrinkle은 wrinkle에 합산) — ±1 기준
            legacy_key = "wrinkle" if dig == "forehead_wrinkle" else dig
            if legacy_key in self.test_value:
                self.test_value[legacy_key].update_acc(off1, n)

            # idx별 best 체크포인트 선정용 ±1 accuracy 누적
            self.val_acc[self.m_idx].update_acc(off1, n)

            # 신규: 6개 항목 분리 메트릭
            self.metrics[dig]["strict"].update_acc(strict, n)
            self.metrics[dig]["off1"].update_acc(off1, n)
            self.metrics[dig]["preds"].extend(pred_l.detach().cpu().numpy().tolist())
            self.metrics[dig]["targets"].extend(gt_l.detach().cpu().numpy().tolist())

    def epoch_report(self):
        """매 epoch 끝날 때 valid 메트릭 요약 (CM 없이 1줄/항목)."""
        if self.args.mode != "class":
            return
        lines = [f"\n[Epoch {self.epoch}] Valid metrics"]
        strict_avgs, off1_avgs, f1s = [], [], []
        for dig in DIAG_KEYS:
            m = self.metrics[dig]
            if m["off1"].count == 0:
                continue
            num_cls = class_num_list[dig]
            strict_pct = m["strict"].avg * 100
            off1_pct = m["off1"].avg * 100
            if HAS_SKLEARN:
                f1 = f1_score(m["targets"], m["preds"], average="macro",
                              labels=list(range(num_cls)), zero_division=0) * 100
            else:
                f1 = 0.0  # sklearn 없으면 생략
            lines.append(
                f"  {dig:<20}  N={m['off1'].count:>6,}   "
                f"Strict={strict_pct:6.2f}%   "
                f"±1={off1_pct:6.2f}%   "
                f"F1={f1:6.2f}%"
            )
            strict_avgs.append(strict_pct)
            off1_avgs.append(off1_pct)
            f1s.append(f1)
        if strict_avgs:
            lines.append(
                f"  {'AVG':<20}  ----     "
                f"Strict={np.mean(strict_avgs):6.2f}%   "
                f"±1={np.mean(off1_avgs):6.2f}%   "
                f"F1={np.mean(f1s):6.2f}%"
            )
        msg = "\n".join(lines)
        self.logger.info(msg)
        # tensorboard
        for dig in DIAG_KEYS:
            m = self.metrics[dig]
            if m["off1"].count == 0:
                continue
            self.writer.add_scalar(f"valid_strict/{dig}", m["strict"].avg, self.epoch)
            self.writer.add_scalar(f"valid_off1/{dig}", m["off1"].avg, self.epoch)

    def get_test_loss(self, pred_p, label, area_num):
        count = 0
        for name in self.equip_loss[area_num]:
            gt = label[:, count : count + self.equip_loss[area_num][name]]
            pred = pred_p[:, count : count + self.equip_loss[area_num][name]]
            count += self.equip_loss[area_num][name]
            self.test_value[name].update(
                self.criterion(pred, gt).item(), batch_size=pred.shape[0]
            )

    def run(self, phase="train"):
        
        self.model = (
            copy.deepcopy(self.model_list[self.m_idx])
            if phase == 'train'
            else self.temp_model_list[self.m_idx]
        )
        self.model.train() if phase == "train" else self.model.eval()
        
        optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=0,
        )
        adjust_learning_rate(optimizer, self.epoch, self.args)

        data_loader = (
            self.train_loader if phase == "train"
            else self.valid_loader
        )
        
        self.phase = phase
        self.area_num = str(self.m_idx + 1) if self.flag else str(self.m_idx)
        self.img_count = 0
        
        def run_iter():
            self.img_count = 0
            for iteration, patch_list in enumerate(data_loader):
                if not self.area_num in list(patch_list.keys()):
                    continue

                if type(patch_list[self.area_num][1]) == torch.Tensor:
                    label = patch_list[self.area_num][1].to(device)
                    
                else:
                    for name in patch_list[self.area_num][1]:
                        patch_list[self.area_num][1][name] = patch_list[self.area_num][1][
                            name
                        ].to(device)
                    label = patch_list[self.area_num][1]

                if label == {}:
                    continue
                
                img = patch_list[self.area_num][0].to(device)
                if self.area_num in [4, 6]:
                    img = torch.flip(img, dims=[3])

                if img.shape[-1] > 128:
                    img_l = img[:, :, :, :128]
                    img_r = torch.flip(img[:, :, :, 128:], dims=[3])
                    pred = self.model.to(device)(img_l)
                    pred = self.model.to(device)(img_r) + pred

                elif img.shape[-2] > 128:
                    img_l = img[:, :, :128, :]
                    img_r = torch.flip(img[:, :, 128:, :], dims=[2])
                    pred = self.model.to(device)(img_l)
                    pred = self.model.to(device)(img_r) + pred

                else:
                    pred = self.model.to(device)(img)

                if self.args.mode == "class":
                    loss = self.class_loss(pred, label)
                    if self.phase == "valid":
                        self.get_test_acc(pred, label)

                else:
                    idx_list = set([idx for idx in range(label.size(0))])
                    nan_list = set(self.nan_detect(label))
                    idx_list = list(idx_list - nan_list)
                    if len(idx_list) > 0:
                        loss = self.regression(pred[idx_list], label[idx_list])
                    else:
                        continue

                self.print_loss(iteration)

                if self.phase == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if iteration == len(data_loader) - 1:
                        self.temp_model_list[self.m_idx] = self.model
                        
                self.img_count += 1
                self.temp_model_list[self.m_idx] = self.model
        
        if self.phase == 'train':
            run_iter()
        else:
            with torch.no_grad():
                run_iter()
                
        print(f"{self.phase}_{self.area_num}_{self.img_count}장")
                
                    
