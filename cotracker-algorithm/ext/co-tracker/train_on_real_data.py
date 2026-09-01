# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import copy
import torch
import signal
import socket
import sys
import json

import numpy as np
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

from pytorch_lightning.lite import LightningLite

from cotracker.models.core.cotracker.cotracker import CoTracker2
from cotracker.models.core.cotracker.cotracker3_offline import CoTrackerThreeOffline
from cotracker.models.core.cotracker.cotracker3_online import CoTrackerThreeOnline

from cotracker.utils.visualizer import Visualizer

from cotracker.evaluation.core.evaluator import Evaluator
from cotracker.datasets.utils import collate_fn, collate_fn_train, dataclass_to_cuda_
from cotracker.models.core.model_utils import (
    get_uniformly_sampled_pts,
    get_points_on_a_grid,
    get_sift_sampled_pts,
    get_superpoint_sampled_pts,
)
from cotracker.models.core.cotracker.losses import sequence_loss, sequence_prob_loss
from cotracker.models.build_cotracker import build_cotracker
from cotracker.utils.teacher_sampling import (
    TeacherPairSampler,
    blend_teacher_losses,
)
from cotracker.utils.confidence_head_tuning import (
    configure_confidence_head_only,
    configure_confidence_updateformer,
    trainable_parameter_count,
    trainable_parameter_names,
)
from cotracker.utils.train_utils import (
    Logger,
    get_eval_dataloader,
    sig_handler,
    term_handler,
    run_test_eval,
)


def fetch_optimizer(args, model):
    """Create the optimizer and learning rate scheduler"""
    if args.confidence_head_only:
        parameters = configure_confidence_head_only(model)
        weight_decay = 0.0
        print(
            "Confidence-head-only trainable tensors: "
            f"{trainable_parameter_names(model)}"
        )
        print(
            "Confidence-head-only optimizer elements (including masked visibility "
            f"row): {trainable_parameter_count(parameters)}"
        )
    elif args.confidence_updateformer:
        shared_parameters, head_parameters = configure_confidence_updateformer(model)
        parameters = [
            {"params": shared_parameters, "weight_decay": args.wdecay},
            {"params": head_parameters, "weight_decay": 0.0},
        ]
        weight_decay = 0.0
        print(
            "Confidence-UpdateFormer trainable tensors: "
            f"{trainable_parameter_names(model)}"
        )
        print(
            "Confidence-UpdateFormer trainable elements: "
            f"{trainable_parameter_count(shared_parameters + head_parameters)}"
        )
    else:
        for name, param in model.named_parameters():
            if "vis_conf_head" in name:
                param.requires_grad = False
        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        weight_decay = args.wdecay

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameter elements: {total_params}")

    optimizer = optim.AdamW(
        parameters, lr=args.lr, weight_decay=weight_decay, eps=1e-8
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.0,
        cycle_momentum=False,
        anneal_strategy="cos",
    )
    return optimizer, scheduler


def _forward_batch_single_teacher(
    batch, model, args, teacher_models, teacher_model_ind, queries=None
):
    video = batch.video
    trajs_g = batch.trajectory
    vis_g = batch.visibility
    valids = batch.valid
    B, T, C, H, W = video.shape
    assert C == 3
    B, T, N, D = trajs_g.shape
    device = video.device
    failed_sample = False
    if queries is not None:
        queries = queries.clone()
    elif args.real_data_filter_sift:
        queries = get_sift_sampled_pts(video, N, T, [H, W], device=device)

        if queries.shape[1] < N:
            logging.warning(
                f"SIFT wasn't able to extract enough features: {queries.shape[1]}"
            )
            failed_sample = True
            queries = get_uniformly_sampled_pts(N, T, [H, W], device=device)
    elif args.real_data_filter_superpoint:
        queries = get_superpoint_sampled_pts(video, N, T, [H, W], device=device)

        if queries.shape[1] < N:
            logging.warning("SuperPoint wasn't able to extract enough features")
            failed_sample = True
            queries = get_uniformly_sampled_pts(N, T, [H, W], device=device)
    else:
        queries = get_uniformly_sampled_pts(N, T, [H, W], device=device)
    # Inference with additional points sampled on a regular grid usually makes predictions better.
    # So we sample these points and discard them thereafter

    teacher_model_type, teacher_model = teacher_models[teacher_model_ind]
    uniform_size = grid_size = sift_size = 0
    queries_cat = queries.clone()
    if "online" in teacher_model_type:
        grid_size = args.train_grid_size
        sift_size = args.train_sift_size
        if grid_size > 0:
            xy = get_points_on_a_grid(grid_size, [H, W], device=device)
            xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2)  #
            queries_cat = torch.cat([queries_cat, xy], dim=1)  #

        if sift_size > 0:
            xy = get_sift_sampled_pts(video, sift_size, T, [H, W], device=device)
            if xy.shape[1] == sift_size:
                queries_cat = torch.cat([queries_cat, xy], dim=1)  #
            else:
                sift_size = 0
    elif "offline" in teacher_model_type:
        uniform_size = 100
        if uniform_size > 0:
            xy = get_uniformly_sampled_pts(uniform_size, T, [H, W], device=device)
            queries_cat = torch.cat([queries_cat, xy], dim=1)  #
    elif teacher_model_type == "tapir":
        pass
    else:
        raise ValueError(f"Model type {teacher_model_type} doesn't exist")

    if "cotracker_three" in teacher_model_type:
        with torch.no_grad():
            (
                trajs_g,
                vis_g,
                confidence,
                __,
            ) = teacher_model(video, queries_cat)
    else:
        with torch.no_grad():
            trajs_g, vis_g, *_ = teacher_model(video, queries_cat)
            confidence = torch.ones_like(vis_g)

    # discarding additional points
    if sift_size > 0 or grid_size > 0 or uniform_size > 0:
        trajs_g = trajs_g[:, :, : -(grid_size**2) - sift_size - uniform_size]
        vis_g = vis_g[:, :, : -(grid_size**2) - sift_size - uniform_size]
        confidence = confidence[:, :, : -(grid_size**2) - sift_size - uniform_size]

    vis_g = vis_g > 0.9

    batch.trajectory = trajs_g
    batch.visibility = vis_g

    if args.model_name == "cotracker_three":
        if (
            torch.isnan(queries).any()
            or torch.isnan(trajs_g).any()
            or queries.abs().max() > 1500
        ):
            logging.warning("failed_sample")
            queries = torch.ones_like(queries).to(queries.device).float()
            valids = torch.zeros_like(valids).to(valids.device).float()

        tracks, visibility, confidence, train_data = model(
            video=video, queries=queries, iters=args.train_iters, is_train=True
        )
        coord_predictions, vis_predictions, confidence_predicitons, valid_mask = (
            train_data
        )

        if failed_sample:
            valid_mask = torch.zeros_like(vis_g)
            logging.warning("Making mask zero for failed sample")

        vis_gts = []
        invis_gts = []
        traj_gts = []
        valids_gts = []
        if args.offline_model:
            S = T
            seq_len = (S // 2) + 1
        else:
            S = args.sliding_window_len
            seq_len = T
        for ind in range(0, seq_len - S // 2, S // 2):
            vis_gts.append(vis_g[:, ind : ind + S].float())
            invis_gts.append(1 - vis_g[:, ind : ind + S].float())
            traj_gts.append(trajs_g[:, ind : ind + S, :, :2])
            valids_gts.append(valids[:, ind : ind + S] * valid_mask[:, ind : ind + S])

        output = {
            "flow": {"predictions": (tracks[0].detach() * valid_mask[..., None])[0]}
        }
        confidence_only = args.confidence_head_only or args.confidence_updateformer
        if not confidence_only:
            seq_loss = sequence_loss(
                coord_predictions,
                traj_gts,
                valids_gts,
                vis=vis_gts,
                gamma=0.8,
                add_huber_loss=True,
                loss_only_for_visible=True,
            )
            output["flow"]["loss"] = seq_loss.mean() * 0.05
        output["flow"]["queries"] = queries.clone()
        output["flow"]["query_frame"] = queries[0, :, 0].cpu().int()

        output["visibility"] = {
            "predictions": visibility[0].detach(),
        }
        if args.supervise_confidence:
            confidence_loss = sequence_prob_loss(
                coord_predictions,
                confidence_predicitons,
                traj_gts,
                vis_gts,
                expected_dist_thresh=args.confidence_distance_threshold,
                target_mode=args.confidence_target_mode,
                soft_inner_radius=args.confidence_inner_radius,
                soft_outer_radius=args.confidence_outer_radius,
            )
            output["confidence"] = {
                "loss": confidence_loss.mean() * args.confidence_loss_weight,
            }
        if (
            not confidence_only
            and not (teacher_model_type == "tapir" or args.train_only_visible_points)
        ):
            seq_loss_invisible = sequence_loss(
                coord_predictions,
                traj_gts,
                valids_gts,
                vis=invis_gts,
                gamma=0.8,
                add_huber_loss=False,
                loss_only_for_visible=True,
            )
            output["flow_invisible"] = {"loss": seq_loss_invisible.mean() * 0.01}

        return output
    else:
        predictions, visibility, train_data = model(
            video=video, queries=queries, iters=args.train_iters, is_train=True
        )
        coord_predictions, vis_predictions, valid_mask = train_data

        if failed_sample:
            valid_mask = torch.zeros_like(valid_mask)
            logging.warning("Making mask zero for failed sample")

        vis_gts = []
        traj_gts = []
        valids_gts = []
        delta = 6
        S = args.sliding_window_len
        pred_ind = 0

        for ind in range(0, args.sequence_len - S // 2, S // 2):
            vis_gts.append(vis_g[:, ind : ind + S])
            traj_gts.append(trajs_g[:, ind : ind + S])
            if (
                teacher_model_type == "tapir"
                or teacher_model_type == "online_cotracker_three"
                or args.train_only_visible_points
            ):
                valids_gts.append(
                    valids[:, ind : ind + S]
                    * valid_mask[:, ind : ind + S]
                    * vis_g[:, ind : ind + S]
                    > 0.9
                )
            else:
                valids_gts.append(
                    valids[:, ind : ind + S] * valid_mask[:, ind : ind + S]
                )
            pred_ind += 1

        seq_loss = sequence_loss(
            coord_predictions,
            traj_gts,
            vis_gts,
            valids_gts,
            gamma=0.8,
            loss_only_for_visible_pts=False,
        )

        batch.trajectory = batch.trajectory * valid_mask[..., None]
        output = {"flow": {}}

        output["flow"]["predictions"] = (predictions.detach() * valid_mask[..., None])[
            0
        ]
        output["flow"]["loss"] = seq_loss.mean()
        output["flow"]["query_frame"] = queries[0, :, 0].cpu().int()
        output["visibility"] = {
            "predictions": visibility[0].detach(),
        }
        return output


def _clone_teacher_batch(batch):
    """Copy pseudo-label fields while sharing the immutable input video."""

    cloned = copy.copy(batch)
    for field in ("trajectory", "visibility", "valid"):
        value = getattr(batch, field, None)
        if value is not None:
            setattr(cloned, field, value.clone())
    return cloned


def _sum_output_losses(output):
    total = None
    for value in output.values():
        if isinstance(value, dict) and "loss" in value:
            total = value["loss"] if total is None else total + value["loss"]
    if total is None:
        raise RuntimeError("teacher forward pass produced no loss")
    return total


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _capture_rng_state():
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["state"].cpu().numpy(),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def forward_batch(batch, model, args, teacher_models, teacher_sampler):
    """Run the baseline teacher and an optional random low-weight tutor.

    The two student forward passes are deliberately kept independent.  Their
    losses are normalized to ``(1-alpha) * primary + alpha * auxiliary``, so
    changing ``alpha`` does not change the total distillation-loss scale.
    """

    selection = teacher_sampler.sample()
    primary_batch = _clone_teacher_batch(batch)
    primary_output = _forward_batch_single_teacher(
        primary_batch,
        model,
        args,
        teacher_models,
        selection.primary,
    )
    primary_loss = _sum_output_losses(primary_output)
    auxiliary_loss = None

    if selection.auxiliary is not None:
        if selection.auxiliary == selection.primary:
            # Exact implementation control: no second forward pass or extra
            # model-state update is allowed to distinguish it from baseline.
            auxiliary_loss = primary_loss
        else:
            auxiliary_batch = _clone_teacher_batch(batch)
            # The auxiliary branch is the experimental intervention. Preserve
            # every global RNG stream around it so the next batch, data
            # augmentation, and primary branch remain paired with baseline.
            rng_state_before_auxiliary = _capture_rng_state()
            try:
                auxiliary_output = _forward_batch_single_teacher(
                    auxiliary_batch,
                    model,
                    args,
                    teacher_models,
                    selection.auxiliary,
                    queries=primary_output["flow"]["queries"],
                )
            finally:
                _restore_rng_state(rng_state_before_auxiliary)
            auxiliary_loss = _sum_output_losses(auxiliary_output)

    # Keep primary pseudo labels for visualization and diagnostics.
    batch.trajectory = primary_batch.trajectory
    batch.visibility = primary_batch.visibility

    blended_loss = blend_teacher_losses(
        primary_loss,
        auxiliary_loss,
        args.auxiliary_teacher_weight,
        same_teacher_control=args.same_teacher_control,
    )
    # The training loop sums every nested ``loss`` field. Remove the original
    # components and expose exactly one normalized objective.
    for value in primary_output.values():
        if isinstance(value, dict):
            value.pop("loss", None)
    primary_output["flow"]["loss"] = blended_loss
    primary_output["teacher_sampling"] = {
        "primary_index": selection.primary,
        "auxiliary_index": selection.auxiliary,
        "primary_loss_value": float(primary_loss.detach().cpu()),
        "auxiliary_loss_value": (
            None
            if auxiliary_loss is None
            else float(auxiliary_loss.detach().cpu())
        ),
    }
    return primary_output


class Lite(LightningLite):
    def run(self, args):
        def seed_everything(seed: int):
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        seed_everything(args.seed)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)

        if self.global_rank == 0:
            eval_dataloaders = []
            final_dataloaders = []
            evaluator = None
            if not args.skip_evaluation:
                for ds_name in args.eval_datasets:
                    eval_dataloaders.append(
                        (ds_name, get_eval_dataloader(args.dataset_root, ds_name))
                    )
                final_dataloaders = [dl for dl in eval_dataloaders]
                if not args.debug:
                    for ds_name in (
                        "tapvid_kinetics_first",
                        "tapvid_robotap",
                        "dynamic_replica",
                    ):
                        final_dataloaders.append(
                            (ds_name, get_eval_dataloader(args.dataset_root, ds_name))
                        )
                evaluator = Evaluator(args.ckpt_path)

            visualizer = Visualizer(
                save_dir=args.ckpt_path,
                pad_value=180,
                fps=1,
                show_first_frame=0,
                tracks_leave_trace=0,
            )

        if args.model_name == "cotracker":
            model = CoTracker2(
                stride=args.model_stride,
                window_len=args.sliding_window_len,
                num_virtual_tracks=args.num_virtual_tracks,
                model_resolution=args.crop_size,
            )
        elif args.model_name == "cotracker_three":
            if args.offline_model:
                model = CoTrackerThreeOffline(
                    stride=4,
                    corr_radius=3,
                    window_len=60,
                    model_resolution=(384, 512),
                    linear_layer_for_vis_conf=True,
                )
            else:
                model = CoTrackerThreeOnline(
                    stride=4,
                    corr_radius=3,
                    window_len=16,
                    model_resolution=(384, 512),
                    linear_layer_for_vis_conf=True,
                )
        else:
            raise ValueError(f"Model {args.model_name} doesn't exist")

        with open(args.ckpt_path + "/meta.json", "w") as file:
            json.dump(vars(args), file, sort_keys=True, indent=4)

        model.cuda()
        teacher_models = []
        if args.model_name != "cotracker_three":
            raise ValueError("random tutor training currently requires cotracker_three")

        for teacher_type in args.teacher_types:
            if teacher_type == "cotracker2v1":
                checkpoint = args.teacher_cotracker2_ckpt
                teacher_model = (
                    build_cotracker(
                        window_len=16,
                        offline=False,
                        checkpoint=checkpoint,
                        v2=True,
                    )
                    .cuda()
                    .eval()
                )
                teacher_models.append(("online", teacher_model))
            elif teacher_type == "online_cotracker_three":
                checkpoint = args.teacher_online_ckpt
                teacher_model = (
                    build_cotracker(
                        checkpoint=checkpoint,
                        offline=False,
                        window_len=16,
                    )
                    .cuda()
                    .eval()
                )
                teacher_models.append((teacher_type, teacher_model))
            elif teacher_type == "offline_cotracker_three":
                checkpoint = args.teacher_offline_ckpt
                teacher_model = (
                    build_cotracker(
                        checkpoint=checkpoint,
                        offline=True,
                        window_len=60,
                    )
                    .cuda()
                    .eval()
                )
                teacher_models.append((teacher_type, teacher_model))
            elif teacher_type == "tapir":
                from cotracker.models.bootstap_predictor import TAPIRPredictor

                teacher_models.append(("tapir", TAPIRPredictor()))
            else:
                raise ValueError(f"Unsupported teacher type: {teacher_type}")

        if args.trackrad_data_dir:
            from cotracker.datasets.trackrad_video_dataset import TrackRADVideoDataset

            train_dataset = TrackRADVideoDataset(
                root=args.trackrad_data_dir,
                crop_size=args.crop_size,
                seq_len=args.sequence_len,
                traj_per_sample=args.traj_per_sample,
                random_frame_rate=args.random_frame_rate,
                limit_samples=args.limit_samples,
            )
        else:
            from cotracker.datasets import real_dataset

            train_dataset = real_dataset.RealDataset(
                crop_size=args.crop_size,
                seq_len=args.sequence_len,
                traj_per_sample=args.traj_per_sample,
                random_frame_rate=args.random_frame_rate,
                random_seq_len=args.offline_model,
                data_splits=args.real_data_splits,
                random_resize=False,
                limit_samples=args.limit_samples,
            )

        teacher_sampler = TeacherPairSampler(
            teacher_count=len(teacher_models),
            auxiliary_weight=args.auxiliary_teacher_weight,
            seed=args.teacher_seed + self.global_rank,
            auxiliary_seed=args.auxiliary_teacher_seed + self.global_rank,
            same_teacher_control=args.same_teacher_control,
        )
        if self.global_rank == 0:
            logging.info(
                "Experiment %s; teacher pool: %s; auxiliary weight: %.4f; same-teacher control: %s; primary seed: %d; auxiliary seed: %d",
                args.experiment_name,
                [name for name, _ in teacher_models],
                args.auxiliary_teacher_weight,
                args.same_teacher_control,
                args.teacher_seed,
                args.auxiliary_teacher_seed,
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=True,
            collate_fn=collate_fn_train,
            drop_last=True,
        )

        train_loader = self.setup_dataloaders(train_loader, move_to_device=False)
        print("LEN TRAIN LOADER", len(train_loader))
        optimizer, scheduler = fetch_optimizer(args, model)

        total_steps = 0
        resume_epoch = 0
        resume_batch = 0
        if self.global_rank == 0:
            logger = Logger(model, scheduler, args.ckpt_path)

        folder_ckpts = [
            f
            for f in os.listdir(args.ckpt_path)
            if not os.path.isdir(os.path.join(args.ckpt_path, f))
            and f.endswith(".pth")
            and not "final" in f
        ]
        if len(folder_ckpts) > 0:
            ckpt_path = sorted(folder_ckpts)[-1]
            ckpt = self.load(os.path.join(args.ckpt_path, ckpt_path))
            logging.info(f"Loading checkpoint {ckpt_path}")
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            else:
                model.load_state_dict(ckpt)
            if "optimizer" in ckpt:
                logging.info("Load optimizer")
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                logging.info("Load scheduler")
                scheduler.load_state_dict(ckpt["scheduler"])
            if "total_steps" in ckpt:
                total_steps = ckpt["total_steps"]
                logging.info(f"Load total_steps {total_steps}")
            if "teacher_sampler" in ckpt:
                teacher_sampler.load_state_dict(ckpt["teacher_sampler"])
                logging.info("Load teacher sampler state")
            if "rng_state" in ckpt:
                _restore_rng_state(ckpt["rng_state"])
                logging.info("Load random-number generator state")
            resume_epoch = ckpt.get("epoch", 0)
            resume_batch = ckpt.get("next_batch", 0)
            logging.info(
                "Resume data position at epoch=%s batch=%s",
                resume_epoch,
                resume_batch,
            )

        elif args.restore_ckpt is not None:
            assert args.restore_ckpt.endswith(".pth") or args.restore_ckpt.endswith(
                ".pt"
            )
            logging.info("Loading checkpoint...")

            state_dict = self.load(args.restore_ckpt)
            if "model" in state_dict:
                state_dict = state_dict["model"]

            if list(state_dict.keys())[0].startswith("module."):
                state_dict = {
                    k.replace("module.", ""): v for k, v in state_dict.items()
                }
            model.load_state_dict(state_dict, strict=True)

            logging.info(f"Done loading checkpoint")
        model, optimizer = self.setup(model, optimizer, move_to_device=False)
        # model.cuda()
        model.train()

        def save_training_checkpoint(epoch_index, next_batch_index):
            ckpt_iter = str(total_steps).zfill(8)
            save_path = Path(
                f"{args.ckpt_path}/model_{args.model_name}_{ckpt_iter}.pth"
            )
            temporary_path = save_path.with_suffix(".pth.tmp")
            save_dict = {
                "model": _unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "total_steps": total_steps,
                "epoch": epoch_index,
                "next_batch": next_batch_index,
                "teacher_sampler": teacher_sampler.state_dict(),
                "rng_state": _capture_rng_state(),
                "experiment": vars(args),
            }
            logging.info(f"Saving atomic checkpoint {save_path}")
            self.save(save_dict, temporary_path)
            os.replace(temporary_path, save_path)
            latest_path = Path(args.ckpt_path) / "latest_checkpoint.txt"
            latest_tmp = latest_path.with_suffix(".tmp")
            latest_tmp.write_text(save_path.name + "\n", encoding="utf-8")
            os.replace(latest_tmp, latest_path)
            checkpoints = sorted(
                Path(args.ckpt_path).glob(f"model_{args.model_name}_*.pth")
            )
            for obsolete in checkpoints[: -args.keep_last_checkpoints]:
                obsolete.unlink()

        save_freq = args.save_freq
        scaler = GradScaler(enabled=False)

        should_keep_training = True
        global_batch_num = 0
        epoch = resume_epoch - 1

        if (
            self.global_rank == 0
            and args.validate_at_start
            and not args.skip_evaluation
        ):
            run_test_eval(
                evaluator,
                model,
                eval_dataloaders,
                logger.writer,
                total_steps,
            )
            model.train()
            torch.cuda.empty_cache()

        if total_steps >= args.num_steps:
            should_keep_training = False

        while should_keep_training:
            epoch += 1
            g.manual_seed(args.seed + epoch)
            for i_batch, batch in enumerate(tqdm(train_loader)):
                if epoch == resume_epoch and i_batch < resume_batch:
                    continue
                batch, gotit = batch
                if not all(gotit):
                    print("batch is None")
                    continue
                dataclass_to_cuda_(batch)

                optimizer.zero_grad()

                assert model.training

                output = forward_batch(
                    batch,
                    model,
                    args,
                    teacher_models=teacher_models,
                    teacher_sampler=teacher_sampler,
                )

                loss = 0
                for k, v in output.items():
                    if "loss" in v:
                        loss += v["loss"]

                if self.global_rank == 0:
                    for k, v in output.items():
                        if "loss" in v:
                            logger.writer.add_scalar(
                                f"live_{k}_loss", v["loss"].item(), total_steps
                            )
                        if "metrics" in v:
                            logger.push(v["metrics"], k)
                    if total_steps % save_freq == save_freq - 1:
                        visualizer.visualize(
                            video=batch.video.clone(),
                            tracks=batch.trajectory.clone(),
                            visibility=batch.visibility.clone(),
                            filename="train_gt_traj",
                            query_frame=output["flow"]["query_frame"],
                            writer=logger.writer,
                            step=total_steps,
                        )

                        visualizer.visualize(
                            video=batch.video.clone(),
                            tracks=output["flow"]["predictions"][None],
                            visibility=output["visibility"]["predictions"][None] > 0.6,
                            filename="train_pred_traj",
                            query_frame=output["flow"]["query_frame"],
                            writer=logger.writer,
                            step=total_steps,
                        )

                    if len(output) > 1:
                        logger.writer.add_scalar(
                            f"live_total_loss", loss.item(), total_steps
                        )
                    logger.writer.add_scalar(
                        f"learning_rate", optimizer.param_groups[0]["lr"], total_steps
                    )
                    teacher_info = output["teacher_sampling"]
                    if total_steps % args.teacher_log_every == 0:
                        logging.info(
                            "experiment=%s step=%d primary=%s auxiliary=%s alpha=%.4f primary_loss=%.6f auxiliary_loss=%s",
                            args.experiment_name,
                            total_steps,
                            teacher_models[teacher_info["primary_index"]][0],
                            (
                                "none"
                                if teacher_info["auxiliary_index"] is None
                                else teacher_models[teacher_info["auxiliary_index"]][0]
                            ),
                            args.auxiliary_teacher_weight,
                            teacher_info["primary_loss_value"],
                            teacher_info["auxiliary_loss_value"],
                        )
                    logger.writer.add_scalar(
                        "teacher/primary_index",
                        teacher_info["primary_index"],
                        total_steps,
                    )
                    logger.writer.add_scalar(
                        "teacher/primary_loss",
                        teacher_info["primary_loss_value"],
                        total_steps,
                    )
                    if teacher_info["auxiliary_index"] is not None:
                        logger.writer.add_scalar(
                            "teacher/auxiliary_index",
                            teacher_info["auxiliary_index"],
                            total_steps,
                        )
                        logger.writer.add_scalar(
                            "teacher/auxiliary_loss",
                            teacher_info["auxiliary_loss_value"],
                            total_steps,
                        )
                    global_batch_num += 1

                self.barrier()

                self.backward(scaler.scale(loss))

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                total_steps += 1
                if self.global_rank == 0:
                    if (
                        args.save_every_n_steps > 0
                        and total_steps % args.save_every_n_steps == 0
                    ):
                        save_training_checkpoint(epoch, i_batch + 1)
                    if i_batch >= len(train_loader) - 1:
                        if (epoch + 1) % args.save_every_n_epoch == 0:
                            save_training_checkpoint(epoch + 1, 0)

                        if (
                            not args.skip_evaluation
                            and (epoch + 1) % args.evaluate_every_n_epoch == 0
                        ):
                            run_test_eval(
                                evaluator,
                                model,
                                eval_dataloaders,
                                logger.writer,
                                total_steps,
                            )
                            model.train()
                            torch.cuda.empty_cache()

                self.barrier()
                if total_steps >= args.num_steps:
                    should_keep_training = False
                    break
        if self.global_rank == 0:
            print("FINISHED TRAINING")

            PATH = f"{args.ckpt_path}/{args.model_name}_final.pth"
            torch.save(_unwrap_model(model).state_dict(), PATH)
            if not args.skip_evaluation:
                run_test_eval(
                    evaluator, model, final_dataloaders, logger.writer, total_steps
                )
            logger.close()


if __name__ == "__main__":
    signal.signal(signal.SIGUSR1, sig_handler)
    signal.signal(signal.SIGTERM, term_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="cotracker_three", help="model name")
    parser.add_argument(
        "--experiment_name", default="random_tutor", help="audit label stored in logs"
    )
    parser.add_argument("--restore_ckpt", help="path to restore a checkpoint")
    parser.add_argument("--ckpt_path", help="path to save checkpoints")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="batch size used during training."
    )
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument(
        "--num_workers", type=int, default=10, help="number of dataloader workers"
    )

    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument("--lr", type=float, default=0.0005, help="max learning rate.")
    parser.add_argument(
        "--wdecay", type=float, default=0.00001, help="Weight decay in optimizer."
    )
    parser.add_argument(
        "--num_steps", type=int, default=200000, help="length of training schedule."
    )
    parser.add_argument(
        "--evaluate_every_n_epoch",
        type=int,
        default=1,
        help="evaluate during training after every n epochs, after every epoch by default",
    )
    parser.add_argument(
        "--save_every_n_epoch",
        type=int,
        default=1,
        help="save checkpoints during training after every n epochs, after every epoch by default",
    )
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=50,
        help="atomically save optimizer and teacher-sampler state every N steps; 0 disables",
    )
    parser.add_argument(
        "--keep_last_checkpoints",
        type=int,
        default=3,
        help="retain the newest N resumable checkpoints to bound disk usage",
    )
    parser.add_argument(
        "--validate_at_start",
        action="store_true",
        help="whether to run evaluation before training starts",
    )
    parser.add_argument(
        "--save_freq",
        type=int,
        default=100,
        help="frequency of trajectory visualization during training",
    )
    parser.add_argument(
        "--traj_per_sample",
        type=int,
        default=768,
        help="the number of trajectories to sample for training",
    )
    parser.add_argument(
        "--dataset_root", type=str, help="path lo all the datasets (train and eval)"
    )
    parser.add_argument(
        "--trackrad_data_dir",
        type=str,
        default=None,
        help="TrackRAD case directory containing */images/*_frames.mha",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="skip TapVid evaluation when only TrackRAD training data is mounted",
    )
    parser.add_argument("--seed", type=int, default=0, help="global training seed")
    parser.add_argument(
        "--teacher_seed",
        type=int,
        default=20260829,
        help="independent, checkpointed teacher-sampling seed",
    )
    parser.add_argument(
        "--auxiliary_teacher_seed",
        type=int,
        default=20260830,
        help="independent, checkpointed auxiliary-teacher sampling seed",
    )
    parser.add_argument(
        "--teacher_log_every",
        type=int,
        default=1,
        help="write selected teacher names and losses every N optimizer steps",
    )
    parser.add_argument(
        "--auxiliary_teacher_weight",
        type=float,
        default=0.0,
        help="normalized tutor-teacher weight alpha in [0, 0.5]",
    )
    parser.add_argument(
        "--same_teacher_control",
        action="store_true",
        help="use the primary teacher as tutor; should reproduce the baseline objective",
    )
    parser.add_argument(
        "--supervise_confidence",
        action="store_true",
        help="supervise confidence against the selected teacher trajectory",
    )
    parser.add_argument(
        "--confidence_head_only",
        action="store_true",
        help=(
            "freeze every student parameter except the confidence output row; "
            "requires --supervise_confidence and uses confidence loss only"
        ),
    )
    parser.add_argument(
        "--confidence_updateformer",
        action="store_true",
        help=(
            "train shared UpdateFormer representations and the confidence output "
            "row using confidence loss only; coordinate and visibility heads stay "
            "frozen"
        ),
    )
    parser.add_argument(
        "--confidence_target_mode",
        choices=["hard", "linear_soft"],
        default="hard",
        help="hard radius target or piecewise-linear soft confidence target",
    )
    parser.add_argument(
        "--confidence_distance_threshold",
        type=float,
        default=12.0,
        help="positive hard-label distance threshold in pixels",
    )
    parser.add_argument(
        "--confidence_inner_radius",
        type=float,
        default=8.0,
        help="distance at or below which a soft confidence target equals one",
    )
    parser.add_argument(
        "--confidence_outer_radius",
        type=float,
        default=16.0,
        help="distance at or above which a soft confidence target equals zero",
    )
    parser.add_argument(
        "--confidence_loss_weight",
        type=float,
        default=1.0,
        help="non-negative multiplier applied to confidence BCE",
    )
    parser.add_argument(
        "--teacher_types",
        nargs="+",
        choices=[
            "cotracker2v1",
            "online_cotracker_three",
            "offline_cotracker_three",
            "tapir",
        ],
        default=[
            "cotracker2v1",
            "online_cotracker_three",
            "offline_cotracker_three",
        ],
        help="teacher pool sampled once (baseline) or twice (random tutor) per batch",
    )
    parser.add_argument(
        "--teacher_cotracker2_ckpt",
        default="./checkpoints/cotracker2v1.pth",
    )
    parser.add_argument(
        "--teacher_online_ckpt",
        default="./checkpoints/baseline_online.pth",
    )
    parser.add_argument(
        "--teacher_offline_ckpt",
        default="./checkpoints/baseline_offline.pth",
    )

    parser.add_argument(
        "--train_iters",
        type=int,
        default=4,
        help="number of updates to the disparity field in each forward pass.",
    )
    parser.add_argument(
        "--sequence_len", type=int, default=8, help="train sequence length"
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        default=["tapvid_davis_first"],
        help="what datasets to use for evaluation",
    )
    parser.add_argument(
        "--num_virtual_tracks",
        type=int,
        default=None,
        help="stride of the CoTracker feature network",
    )
    parser.add_argument(
        "--dont_use_augs",
        action="store_true",
        help="don't apply augmentations during training",
    )
    parser.add_argument(
        "--sample_vis_1st_frame",
        action="store_true",
        help="only sample trajectories with points visible on the first frame",
    )
    parser.add_argument(
        "--sliding_window_len",
        type=int,
        default=8,
        help="length of the CoTracker sliding window",
    )
    parser.add_argument(
        "--model_stride",
        type=int,
        default=8,
        help="stride of the CoTracker feature network",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        nargs="+",
        default=[384, 512],
        help="crop videos to this resolution during training",
    )
    parser.add_argument(
        "--eval_max_seq_len",
        type=int,
        default=1000,
        help="maximum length of evaluation videos",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="saves launch time for faster debug",
    )
    parser.add_argument(
        "--random_frame_rate",
        action="store_true",
        help="random_frame_rate",
    )
    parser.add_argument(
        "--real_data_splits",
        type=int,
        nargs="+",
        default=[0],
        help="real data folders",
    )
    parser.add_argument(
        "--loss_only_for_visible_pts",
        action="store_true",
        help="compute sequence loss only for visible points",
    )
    parser.add_argument(
        "--real_data_filter_sift",
        action="store_true",
        help="select point to track based on SIFT features",
    )
    parser.add_argument(
        "--train_grid_size",
        type=int,
        default=5,
        help="number of extra regular grid points that we sample at training. This number will be squared",
    )
    parser.add_argument(
        "--train_sift_size",
        type=int,
        default=0,
        help="number of extra SIFT points that we sample at training.",
    )
    parser.add_argument(
        "--real_data_filter_superpoint",
        action="store_true",
        help="select point to track based on SuperPoint features",
    )
    parser.add_argument(
        "--train_only_visible_points",
        action="store_true",
        help="Loss only for visible points",
    )
    parser.add_argument(
        "--offline_model",
        action="store_true",
        help="training the offline model",
    )
    parser.add_argument(
        "--clean_kubric",
        action="store_true",
        help="filtering out bad tracks in Kubric",
    )
    parser.add_argument(
        "--random_number_traj",
        action="store_true",
        help="when training on Kubric, sampling a random number \
            of tracks between 1 and args.traj_per_sample",
    )
    parser.add_argument(
        "--random_seq_len",
        action="store_true",
        help="when training on Kubric, cropping the sequence \
            to have a length between 10 and args.sequence_len frames",
    )
    parser.add_argument(
        "--uniform_query_sampling_method",
        action="store_true",
        help="Whether to sample points uniformly across time. Kubric training only",
    )
    parser.add_argument(
        "--limit_samples", type=int, default=10000, help="limit samples on real data"
    )

    args = parser.parse_args()
    if not 0.0 <= args.auxiliary_teacher_weight <= 0.5:
        parser.error("--auxiliary_teacher_weight must be in [0, 0.5]")
    if args.same_teacher_control and args.auxiliary_teacher_weight == 0.0:
        parser.error("--same_teacher_control requires a positive auxiliary weight")
    if args.teacher_log_every < 1:
        parser.error("--teacher_log_every must be positive")
    if args.confidence_distance_threshold <= 0:
        parser.error("--confidence_distance_threshold must be positive")
    if (
        args.confidence_inner_radius < 0
        or args.confidence_outer_radius <= args.confidence_inner_radius
    ):
        parser.error(
            "confidence radii must satisfy 0 <= inner radius < outer radius"
        )
    if args.confidence_loss_weight < 0:
        parser.error("--confidence_loss_weight must be non-negative")
    confidence_only_modes = int(args.confidence_head_only) + int(
        args.confidence_updateformer
    )
    if confidence_only_modes > 1:
        parser.error(
            "--confidence_head_only and --confidence_updateformer are mutually exclusive"
        )
    if confidence_only_modes and not args.supervise_confidence:
        parser.error("confidence-only modes require --supervise_confidence")
    if confidence_only_modes and args.auxiliary_teacher_weight != 0.0:
        parser.error(
            "confidence-only modes are isolated single-teacher ablations and "
            "require --auxiliary_teacher_weight 0"
        )
    if args.keep_last_checkpoints < 1:
        parser.error("--keep_last_checkpoints must be positive")
    if args.trackrad_data_dir and not args.skip_evaluation and not args.dataset_root:
        parser.error(
            "TrackRAD-only training requires --skip_evaluation or a TapVid --dataset_root"
        )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    Path(args.ckpt_path).mkdir(exist_ok=True, parents=True)
    from pytorch_lightning.strategies import DDPStrategy

    Lite(
        strategy=DDPStrategy(find_unused_parameters=False),
        devices="auto",
        accelerator="gpu",
        precision="bf16" if args.mixed_precision else 32,
        num_nodes=args.num_nodes,
        # precision=32,
    ).run(args)
