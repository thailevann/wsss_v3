import argparse
import datetime
import logging
import os
import wandb
import numpy as np
import cv2 as cv
from omegaconf import OmegaConf
from tqdm import tqdm
import ttach as tta_lib
from skimage import morphology

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.trainutils import get_cls_dataset, all_reduced
from utils.optimizer import PolyWarmupAdamW
from utils.pyutils import str2bool, set_seed, setup_logger, AverageMeter
from utils.evaluate import ConfusionMatrixAllClass
from utils.cam_utils import get_seg_label
from utils.lir_utils import localization_informed_regularization
from utils.pseudo_refine import refine_pseudo_by_shortest_path, refined_to_soft_mask
from cls_network.model import ClsNetwork


start_time = datetime.datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None)
parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
parser.add_argument('--backend', default='nccl')
parser.add_argument("--wandb_log", type=str2bool, default=False)
args = parser.parse_args()


def _get_seg_label_with_cfg(cams, inputs, label, cfg):
    return get_seg_label(cams, inputs, label)


def _fuse_cam234(cam2, cam3, cam4, cfg, current_epoch=None):
    """Fuse cam2, cam3, cam4. If fuse_reweight_epoch: (1-alpha)*(0.5*cam2+0.5*cam3)+alpha*cam4, alpha=min(fuse_alpha_max, epoch/T). Else: 0.3*cam2+0.3*cam3+0.4*cam4."""
    if not cfg.train.get("fuse_reweight_epoch", False):
        return 0.3 * cam2 + 0.3 * cam3 + 0.4 * cam4
    T = float(cfg.train.get("fuse_ramp_epochs", 5))
    alpha_max = float(cfg.train.get("fuse_alpha_max", 1.0))
    epoch = current_epoch if current_epoch is not None else 1e9
    alpha = min(alpha_max, epoch / T)
    return (1.0 - alpha) * (0.5 * cam2 + 0.5 * cam3) + alpha * cam4


def _weak_aug_for_consist(x, device):
    x = x.to(device)
    b = 0.9 + 0.2 * torch.rand(1, device=device, dtype=x.dtype).view(1, 1, 1, 1)
    out = x * b
    mean = out.view(out.size(0), -1).mean(dim=1, keepdim=True).view(-1, 1, 1, 1)
    c = 0.9 + 0.2 * torch.rand(1, device=device, dtype=x.dtype).view(1, 1, 1, 1)
    out = (out - mean) * c + mean
    out = out + 0.02 * torch.randn_like(out, device=device)
    return out


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now().replace(microsecond=0)
    scale = (total_iter - cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = delta * scale
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def validate(model=None, data_loader=None, cfg=None, cls_loss_func=None, distributed=False, n_gpus=1, current_epoch=None):
    model.eval()
    avg_meter = AverageMeter()
    fuse234_matrix = ConfusionMatrixAllClass(num_classes=cfg.dataset.cls_num_classes + 1)
    tta_transform = tta_lib.Compose([
        tta_lib.HorizontalFlip(),
        tta_lib.Multiply(factors=[0.9, 1.0, 1.1])
    ])

    if len(data_loader) == 0:
        logging.warning("Validation data_loader is empty.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        all_cls_acc4 = torch.tensor(0.0).cuda()
        avg_cls_acc4 = torch.tensor(0.0).cuda()
        cls_loss = torch.tensor(0.0).cuda()
        fuse234_score = torch.zeros(cfg.dataset.cls_num_classes + 1, device=device)
        cam_area = torch.tensor(0.0).cuda()
        if distributed:
            all_reduced(all_cls_acc4, n_gpus)
            all_reduced(avg_cls_acc4, n_gpus)
            all_reduced(cls_loss, n_gpus)
        model.train()
        return all_cls_acc4, avg_cls_acc4, fuse234_score, cls_loss, cam_area

    with torch.no_grad():
        for data in tqdm(data_loader, total=len(data_loader), ncols=100, ascii=" >="):
            name, inputs, cls_label, labels = data
            inputs = inputs.cuda().float()
            b, c, h, w = inputs.shape
            labels = labels.cuda()
            cls_label = cls_label.cuda().float()

            cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, attns, _, _ = model(inputs, ema_update_enabled=False)

            cls_loss1 = cls_loss_func(cls1, cls_label)
            cls_loss2 = cls_loss_func(cls2, cls_label)
            cls_loss3 = cls_loss_func(cls3, cls_label)
            cls_loss4 = cls_loss_func(cls4, cls_label)
            cls_loss = cfg.train.l1 * cls_loss1 + cfg.train.l2 * cls_loss2 + cfg.train.l3 * cls_loss3 + cfg.train.l4 * cls_loss4

            cls4 = (torch.sigmoid(cls4) > 0.5).float()
            all_cls_acc4 = (cls4 == cls_label).all(dim=1).float().sum() / cls4.shape[0] * 100
            avg_cls_acc4 = ((cls4 == cls_label).sum(dim=0) / cls4.shape[0]).mean() * 100
            cam_area = (cam4 > 0.3).float().mean()
            avg_meter.add({"all_cls_acc4": all_cls_acc4, "avg_cls_acc4": avg_cls_acc4, "cls_loss": cls_loss, "cam_area": cam_area})

            pseudo_refine_enabled = cfg.train.get("pseudo_refine_enabled", False)
            if pseudo_refine_enabled:
                lam_low = cfg.train.get("pseudo_refine_lam_low", 0.2)
                lam_high = cfg.train.get("pseudo_refine_lam_high", 0.7)
                max_cost = cfg.train.get("pseudo_refine_max_cost", 10.0)
                neighbor = int(cfg.train.get("pseudo_refine_neighbor", 4))
                num_cls = cfg.dataset.cls_num_classes
                segs_refined_list = []
                for tta_trans in tta_transform:
                    augmented_tensor = tta_trans.augment_image(inputs)
                    if augmented_tensor.dim() == 5:
                        augmented_tensor = augmented_tensor.view(-1, *augmented_tensor.shape[2:])
                    _, _, cam2_t, _, cam3_t, cam4_t, _, _, _, _, _ = model(augmented_tensor, ema_update_enabled=False)
                    if (cam2_t.dim() != 4 or cam2_t.shape[2] < 1 or cam2_t.shape[3] < 1 or
                        cam3_t.dim() != 4 or cam4_t.dim() != 4):
                        logging.warning("Validation TTA: CAM spatial dims invalid, skipping this TTA transform.")
                        continue
                    if cam2_t.shape[0] != inputs.shape[0]:
                        cam2_t = cam2_t[:inputs.shape[0]]
                        cam3_t = cam3_t[:inputs.shape[0]]
                        cam4_t = cam4_t[:inputs.shape[0]]
                    cam2 = F.interpolate(cam2_t.float(), size=(h, w), mode="bilinear", align_corners=False)
                    cam3 = F.interpolate(cam3_t.float(), size=(h, w), mode="bilinear", align_corners=False)
                    cam4 = F.interpolate(cam4_t.float(), size=(h, w), mode="bilinear", align_corners=False)
                    cam_fuse = _fuse_cam234(cam2, cam3, cam4, cfg, current_epoch)
                    refined = refine_pseudo_by_shortest_path(
                        cam_fuse, cls_label, lam_low, lam_high, max_cost, neighbor, num_cls
                    )
                    soft = refined_to_soft_mask(refined, num_cls).cuda()
                    soft = tta_trans.deaugment_mask(soft).unsqueeze(dim=0)
                    segs_refined_list.append(soft)
                if not segs_refined_list:
                    cam2 = F.interpolate(cam2.float(), size=(h, w), mode="bilinear", align_corners=False)
                    cam3 = F.interpolate(cam3.float(), size=(h, w), mode="bilinear", align_corners=False)
                    cam4 = F.interpolate(cam4.float(), size=(h, w), mode="bilinear", align_corners=False)
                    fuse234 = _fuse_cam234(cam2, cam3, cam4, cfg, current_epoch)
                    fuse_label234 = torch.argmax(fuse234, dim=1).cuda()
                else:
                    fuse234_soft = torch.cat(segs_refined_list, dim=0).mean(dim=0).unsqueeze(dim=0)
                    if cfg.dataset.name == "luad":
                        img = cv.imread(os.path.join(cfg.dataset.val_root, "test", "img", name[0]), cv.IMREAD_UNCHANGED)
                        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                        ret, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
                        binary = np.uint8(binary)
                        dst = morphology.remove_small_objects(binary == 255, min_size=80, connectivity=1).astype(np.uint8)
                        priori_bg_mask = (1 - dst).reshape(1, 1, h, w)
                        priori_bg_mask = torch.from_numpy(priori_bg_mask).float().cuda()
                        fuse234_soft[:, 1:, :, :] *= priori_bg_mask
                    fuse_label234 = torch.argmax(fuse234_soft, dim=1).cuda()
            else:
                # Plan: fuse cam2, cam3, cam4 → pseudo mask (sim_only = get_seg_label) → argmax
                segs_list = []
                for tta_trans in tta_transform:
                    augmented_tensor = tta_trans.augment_image(inputs)
                    if augmented_tensor.dim() == 5:
                        augmented_tensor = augmented_tensor.view(-1, *augmented_tensor.shape[2:])
                    _, _, cam2_t, _, cam3_t, cam4_t, _, _, _, _, _ = model(augmented_tensor, ema_update_enabled=False)
                    if cam2_t.dim() != 4 or cam2_t.shape[2] < 1 or cam2_t.shape[3] < 1:
                        continue
                    if cam2_t.shape[0] != inputs.shape[0]:
                        cam2_t = cam2_t[:inputs.shape[0]]
                        cam3_t = cam3_t[:inputs.shape[0]]
                        cam4_t = cam4_t[:inputs.shape[0]]
                    cam2 = F.interpolate(cam2_t.float(), size=(h, w), mode="bilinear", align_corners=False)
                    cam3 = F.interpolate(cam3_t.float(), size=(h, w), mode="bilinear", align_corners=False)
                    cam4 = F.interpolate(cam4_t.float(), size=(h, w), mode="bilinear", align_corners=False)
                    fuse234 = _fuse_cam234(cam2, cam3, cam4, cfg, current_epoch)
                    soft = _get_seg_label_with_cfg(fuse234, augmented_tensor, cls_label, cfg).cuda()
                    soft = tta_trans.deaugment_mask(soft).unsqueeze(dim=0)
                    segs_list.append(soft)
                if not segs_list:
                    cam2 = F.interpolate(cam2.float(), size=(h, w), mode="bilinear", align_corners=False)
                    cam3 = F.interpolate(cam3.float(), size=(h, w), mode="bilinear", align_corners=False)
                    cam4 = F.interpolate(cam4.float(), size=(h, w), mode="bilinear", align_corners=False)
                    fuse234 = _fuse_cam234(cam2, cam3, cam4, cfg, current_epoch)
                    soft = _get_seg_label_with_cfg(fuse234, inputs, cls_label, cfg).cuda()
                    fuse_label234 = torch.argmax(soft, dim=1).cuda()
                else:
                    fuse234_soft = torch.cat(segs_list, dim=0).mean(dim=0).unsqueeze(dim=0)
                    if cfg.dataset.name == "luad":
                        img = cv.imread(os.path.join(cfg.dataset.val_root, "test", "img", name[0]), cv.IMREAD_UNCHANGED)
                        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                        ret, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
                        binary = np.uint8(binary)
                        dst = morphology.remove_small_objects(binary == 255, min_size=80, connectivity=1).astype(np.uint8)
                        priori_bg_mask = (1 - dst).reshape(1, 1, h, w)
                        priori_bg_mask = torch.from_numpy(priori_bg_mask).float().cuda()
                        fuse234_soft[:, 1:, :, :] *= priori_bg_mask
                    fuse_label234 = torch.argmax(fuse234_soft, dim=1).cuda()

            fuse234_matrix.update(labels.detach().clone(), fuse_label234.clone())

    try:
        all_cls_acc4, avg_cls_acc4, cls_loss = avg_meter.pop('all_cls_acc4'), avg_meter.pop("avg_cls_acc4"), avg_meter.pop("cls_loss")
        cam_area = avg_meter.pop("cam_area")
    except KeyError:
        all_cls_acc4 = torch.tensor(0.0).cuda()
        avg_cls_acc4 = torch.tensor(0.0).cuda()
        cls_loss = torch.tensor(0.0).cuda()
        cam_area = torch.tensor(0.0).cuda()

    if fuse234_matrix.mat1 is not None:
        if distributed:
            all_reduced(all_cls_acc4, n_gpus)
            all_reduced(avg_cls_acc4, n_gpus)
            all_reduced(cls_loss, n_gpus)
            all_reduced(cam_area, n_gpus)
            fuse234_matrix.reduce_from_all_processes()
        fuse234_score = fuse234_matrix.compute()[2]
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        fuse234_score = torch.zeros(cfg.dataset.cls_num_classes + 1, device=device)
        if distributed:
            all_reduced(all_cls_acc4, n_gpus)
            all_reduced(avg_cls_acc4, n_gpus)
            all_reduced(cls_loss, n_gpus)
            all_reduced(cam_area, n_gpus)

    model.train()
    return all_cls_acc4, avg_cls_acc4, fuse234_score, cls_loss, cam_area


def train(cfg):
    num_workers = 10
    import torch.distributed as dist

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend=args.backend)
        distributed = True
    else:
        args.local_rank = 0
        torch.cuda.set_device(0)
        distributed = False

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        gpu_ids = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        n_gpus = len(gpu_ids)
    else:
        n_gpus = 1
        gpu_ids = [0]

    time0 = datetime.datetime.now().replace(microsecond=0)

    train_dataset, val_dataset = get_cls_dataset(cfg)
    logging.info("use {} images for training, {} for validation".format(len(train_dataset), len(val_dataset)))
    logging.info("use {} GPUs: {}".format(n_gpus, gpu_ids))

    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['train']['samples_per_gpu'],
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    iters_per_epoch = len(train_loader)
    cfg.train.max_iters = cfg.train.epoch * iters_per_epoch
    cfg.train.eval_iters = iters_per_epoch
    cfg.scheduler.warmup_iter = cfg.scheduler.warmup_iter * iters_per_epoch
    val_sampler = None
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg['train']['samples_per_gpu'],
        sampler=val_sampler,
        shuffle=False,
        num_workers=4
    )

    device = torch.device(args.local_rank)

    wetr = ClsNetwork(
        backbone=cfg.model.backbone.config,
        stride=cfg.model.backbone.stride,
        cls_num_classes=cfg.dataset.cls_num_classes,
        n_ratio=cfg.model.n_ratio,
        pretrained=cfg.train.pretrained,
        k_fea_path=cfg.model.knowledge_feature_path,
        l_fea_path=cfg.model.label_feature_path,
        knowledge_encoder=cfg.model.get("knowledge_encoder", "clinical_bert"),
        dataset_name=cfg.dataset.get("name"),
        knowledge_features_base_dir=cfg.model.get("knowledge_features_base_dir"),
        bio_model_name=cfg.model.get("bio_model_name", "emilyalsentzer/Bio_ClinicalBERT"),
        engram_confidence_mode=cfg.model.get("engram_confidence_mode", "norm"),
        engram_reweight_pseudo=cfg.model.get("engram_reweight_pseudo", True),
    )

    logging.info('\nNetwork config: \n%s' % (wetr))
    param_groups = wetr.get_param_groups()
    wetr.to(device)
    wetr.train()

    optimizer = PolyWarmupAdamW(
        params=param_groups,
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power
    )
    train_loader_iter = iter(train_loader)
    if train_sampler is not None:
        train_sampler.set_epoch(np.random.randint(cfg.train.max_iters))
    if distributed:
        wetr = DistributedDataParallel(wetr, device_ids=[args.local_rank], find_unused_parameters=True)

    avg_meter = AverageMeter()
    loss_function = nn.BCEWithLogitsLoss()
    best_fuse234_dice = 0.0
    warmup_iters = int(cfg.train.get("warmup_iters", 0))
    ramp_iters = int(cfg.train.get("ramp_iters", 0))

    for n_iter in range(cfg.train.max_iters):
        wetr.train()
        if warmup_iters <= 0 and ramp_iters <= 0:
            ema_update_enabled = True
            lambda_lir = 1.0
            lambda_consist = 1.0
        elif n_iter < warmup_iters:
            ema_update_enabled = False
            lambda_lir = 0.0
            lambda_consist = 0.0
        elif n_iter < warmup_iters + ramp_iters:
            ema_update_enabled = True
            ratio = min(1.0, float((n_iter - warmup_iters) / max(1, ramp_iters)))
            lambda_lir = ratio
            lambda_consist = ratio
        else:
            ema_update_enabled = True
            lambda_lir = 1.0
            lambda_consist = 1.0

        try:
            img_name, inputs, cls_labels, gt_label = next(train_loader_iter)
        except StopIteration:
            if train_sampler is not None:
                train_sampler.set_epoch(int((n_iter + 1) / iters_per_epoch))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, gt_label = next(train_loader_iter)

        inputs = inputs.to(device).float()
        cls_labels = cls_labels.to(device).float()

        cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, attns, patch_feat_4, class_tokens_4 = wetr(
            inputs, ema_update_enabled=ema_update_enabled
        )

        loss1 = loss_function(cls1, cls_labels)
        loss2 = loss_function(cls2, cls_labels)
        loss3 = loss_function(cls3, cls_labels)
        loss4 = loss_function(cls4, cls_labels)
        loss = cfg.train.l1 * loss1 + cfg.train.l2 * loss2 + cfg.train.l3 * loss3 + cfg.train.l4 * loss4

        lir_weight = cfg.train.get("lir_weight", 0.0)
        loss_lir, lir_dict = None, None
        if lir_weight > 0 and lambda_lir > 0:
            loss_lir, lir_dict = localization_informed_regularization(
                patch_feat_4, class_tokens_4, cam4, cls_labels,
                lam_l=cfg.train.get("lir_lam_l", 0.2),
                lam_h=cfg.train.get("lir_lam_h", 0.7),
                tau=cfg.train.get("lir_tau", 0.07),
                weight_cre=cfg.train.get("lir_weight_cre", 1.0),
            )
            loss = loss + lir_weight * lambda_lir * loss_lir

        consist_weight = cfg.train.get("consist_weight", 0.0)
        loss_consist = None
        if consist_weight > 0 and lambda_consist > 0:
            view2 = _weak_aug_for_consist(inputs, device)
            with torch.no_grad():
                label_for_mask = (torch.sigmoid(cls4) > 0.15).float()
            _, _, _, _, _, _, _, cam4_v2, _, _, _ = wetr(view2, ema_update_enabled=ema_update_enabled)
            cam4_for_mask1 = cam4.detach() if cfg.train.get("consist_detach_cam", False) else cam4
            mask1 = _get_seg_label_with_cfg(cam4_for_mask1, inputs, label_for_mask, cfg)
            mask2 = _get_seg_label_with_cfg(cam4_v2, view2, label_for_mask, cfg)
            loss_consist = F.mse_loss(mask1, mask2)
            loss = loss + consist_weight * lambda_consist * loss_consist

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cls_pred4 = (torch.sigmoid(cls4) > 0.5).float()
        all_cls_acc4 = (cls_pred4 == cls_labels).all(dim=1).float().sum() / cls_labels.shape[0] * 100
        avg_cls_acc4 = ((cls_pred4 == cls_labels).sum(dim=0) / cls_labels.shape[0]).mean() * 100

        if distributed:
            all_reduced(loss, n_gpus)
            all_reduced(all_cls_acc4, n_gpus)
            all_reduced(avg_cls_acc4, n_gpus)

        avg_meter.add({'cls_loss': loss, "all_cls_acc4": all_cls_acc4, "avg_cls_acc4": avg_cls_acc4})
        if loss_lir is not None:
            avg_meter.add({'loss_lir': loss_lir, 'l_cre': lir_dict['l_cre']})
        if loss_consist is not None:
            avg_meter.add({'loss_consist': loss_consist})

        if args.local_rank == 0:
            if (n_iter + 1) % 100 == 0:
                delta, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
                cur_lr = optimizer.param_groups[0]['lr']
                phase = "warmup" if n_iter < warmup_iters else ("ramp" if n_iter < warmup_iters + ramp_iters else "full")
                log_msg = "Iter: %d / %d [%s]; Elapsed: %s; ETA: %s; LR: %.3e; cls_loss: %.4f; all_acc4: %.2f; avg_acc4: %.2f" % (
                    n_iter + 1, cfg.train.max_iters, phase, delta, eta, cur_lr, loss, all_cls_acc4, avg_cls_acc4)
                if loss_lir is not None:
                    log_msg += "; loss_lir: %.4f (l_cre: %.4f)" % (loss_lir.item(), lir_dict["l_cre"].item())
                if loss_consist is not None:
                    log_msg += "; loss_consist: %.4f" % loss_consist.item()
                logging.info(log_msg)

            if args.wandb_log:
                iter_wandb_log = {"iter_log/lr%d" % i: x["lr"] for i, x in enumerate(optimizer.param_groups)}
                iter_wandb_log.update({"iter_log/iter_train_loss": loss.item()})
                if loss_lir is not None:
                    iter_wandb_log.update({"iter_log/loss_lir": loss_lir.item()})
                if loss_consist is not None:
                    iter_wandb_log.update({"iter_log/loss_consist": loss_consist.item()})
                wandb.log(iter_wandb_log, step=n_iter)

            if (n_iter + 1) % cfg.train.eval_iters == 0 or (n_iter + 1) == cfg.train.max_iters:
                cls_loss, all_cls_acc4, avg_cls_acc4 = avg_meter.pop('cls_loss'), avg_meter.pop('all_cls_acc4'), avg_meter.pop("avg_cls_acc4")
                if distributed:
                    if args.local_rank == 0:
                        current_epoch_1based = (n_iter + 1) // iters_per_epoch
                        val_all_acc4, val_avg_acc4, fuse234_score, val_cls_loss, cam_area = validate(
                            model=wetr, data_loader=val_loader, cfg=cfg, cls_loss_func=loss_function,
                            distributed=False, n_gpus=1, current_epoch=current_epoch_1based)
                    else:
                        val_all_acc4 = torch.tensor(0.0).cuda()
                        val_avg_acc4 = torch.tensor(0.0).cuda()
                        val_cls_loss = torch.tensor(0.0).cuda()
                        fuse234_score = torch.zeros(cfg.dataset.cls_num_classes + 1).cuda()
                        cam_area = torch.tensor(0.0).cuda()
                else:
                    current_epoch_1based = (n_iter + 1) // iters_per_epoch
                    val_all_acc4, val_avg_acc4, fuse234_score, val_cls_loss, cam_area = validate(
                        model=wetr, data_loader=val_loader, cfg=cfg, cls_loss_func=loss_function,
                        distributed=False, n_gpus=1, current_epoch=current_epoch_1based)
                logging.info("val all acc4: %.6f" % val_all_acc4)
                logging.info("val avg acc4: %.6f" % val_avg_acc4)
                logging.info("fuse234 score: {}, mIOU: {:.4f}, cam_area: {:.4f}".format(fuse234_score, fuse234_score[:-1].mean(), cam_area.item()))
                if args.wandb_log:
                    wandb.log({
                        "loss/train_loss": cls_loss, "acc/all_cls_acc4": all_cls_acc4, "acc/avg_cls_acc4": avg_cls_acc4,
                        "acc/val_all_acc4": val_all_acc4, "acc/val_avg_acc4": val_avg_acc4,
                        "loss/val_loss": val_cls_loss, "miou/val_mIOU_fuse234": fuse234_score[:-1].mean(),
                        "val/cam_area": cam_area.item()
                    }, step=n_iter)
                state_dict = {
                    "cfg": cfg, "iter": n_iter, "optimizer": optimizer.state_dict(),
                    "model": wetr.module.state_dict() if distributed else wetr.state_dict()
                }
                if fuse234_score[:-1].mean() > best_fuse234_dice:
                    best_fuse234_dice = fuse234_score[:-1].mean()
                    torch.save(state_dict, os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth"))

    if args.local_rank == 0:
        torch.cuda.empty_cache()
        data_root = getattr(cfg.dataset, "val_root", None)
        data_root_arg = " --data_root '%s'" % data_root if (data_root and os.path.isdir(data_root)) else ""
        knowledge_encoder = cfg.model.get("knowledge_encoder", "clinical_bert")
        bio_model_name = cfg.model.get("bio_model_name", "emilyalsentzer/Bio_ClinicalBERT")
        knowledge_encoder_arg = " --knowledge_encoder %s --bio_model_name '%s'" % (knowledge_encoder, bio_model_name)
        logging.info("start test seg......")
        os.system(
            "python evaluate_cls.py --model_path '%s' --gpu 0 --backbone %s --dataset %s --label_feature_path '%s' --knowledge_feature_path '%s' --n_ratio %s %s%s"
            % (os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth"), cfg.model.backbone.config, cfg.dataset.name,
               cfg.model.label_feature_path, cfg.model.knowledge_feature_path, cfg.model.n_ratio, knowledge_encoder_arg, data_root_arg))
        logging.info("test seg finished.......")
        logging.info("start val seg......")
        os.system(
            "python evaluate_cls.py --model_path '%s' --gpu 0 --backbone %s --split valid --dataset %s --label_feature_path '%s' --knowledge_feature_path '%s' --n_ratio %s %s%s"
            % (os.path.join(cfg.work_dir.ckpt_dir, "best_cam.pth"), cfg.model.backbone.config, cfg.dataset.name,
               cfg.model.label_feature_path, cfg.model.knowledge_feature_path, cfg.model.n_ratio, knowledge_encoder_arg, data_root_arg))
        logging.info("val seg finished.......")

    if distributed:
        dist.barrier()
    end_time = datetime.datetime.now()
    logging.info('cost %s' % (end_time - start_time))


if __name__ == "__main__":
    cfg = OmegaConf.load(args.config)
    cfg.work_dir.dir = os.path.dirname(args.config)
    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())
    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.train_log_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.train_log_dir)

    os.makedirs(cfg.work_dir.dir, exist_ok=True)
    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.train_log_dir, exist_ok=True)

    if args.local_rank == 0:
        if args.wandb_log:
            wandb.init(project='TPRO-%s-cls' % cfg.dataset.name)
        log_file = getattr(cfg.work_dir, "log_file", None) or ""
        if log_file.strip():
            log_path = log_file if os.path.isabs(log_file) else os.path.join(cfg.work_dir.train_log_dir, log_file.strip())
        else:
            log_path = os.path.join(cfg.work_dir.train_log_dir, timestamp + '.log')
        setup_logger(filename=log_path)
        logging.info('\nargs: %s' % args)
        logging.info('\nconfigs: %s' % cfg)

    set_seed(0)
    train(cfg=cfg)
