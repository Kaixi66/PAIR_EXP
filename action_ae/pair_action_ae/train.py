"""Train the PAIR Action Autoencoder teacher."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from .checkpoint import (
    append_jsonl,
    save_encoder_checkpoint,
    save_json,
    save_training_checkpoint,
)
from .data import ActionDataConfig, make_action_iterables
from .model import (
    ActionAEConfig,
    ActionPerceptionAEConfig,
    ActionPerceptionTransformerAE,
    ActionTransformerAE,
    corrupt_actions,
)
from .perception import build_perception_tokens

from experiments.robot.openvla_utils import check_model_logic_mismatch, model_is_on_hf_hub, update_auto_map
from huggingface_hub import snapshot_download
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models import load, load_vla
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PAIR ActionTransformerAE.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data_root_dir", type=str, default=None)
    parser.add_argument("--mixture", "--dataset_name", dest="mixture", type=str, default=None)
    parser.add_argument("--run_root_dir", type=Path, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--eval_every", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--eval_batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None)
    parser.add_argument("--ae_version", choices=("v1", "v2"), default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--encoder_layers", type=int, default=None)
    parser.add_argument("--decoder_layers", type=int, default=None)
    parser.add_argument("--perception_layers", type=int, default=None)
    parser.add_argument("--vlm_path", type=str, default=None)
    parser.add_argument("--vla_config_file_path", type=str, default=None)
    parser.add_argument("--num_images_in_input", type=int, default=None)
    parser.add_argument("--mask_mode", choices=("random", "chunk"), default=None)
    parser.add_argument("--mask_prob", type=float, default=None)
    parser.add_argument("--mask_count", type=int, default=None)
    parser.add_argument("--noise_std", type=float, default=None)
    parser.add_argument("--episode_split_file", type=str, default=None)
    parser.add_argument("--data_contract_file", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping, got {type(config).__name__}")
    return config


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}
    if args.data_root_dir is not None:
        updates.setdefault("data", {})["data_root_dir"] = args.data_root_dir
    if args.mixture is not None:
        updates.setdefault("data", {})["mixture"] = args.mixture
    if args.run_root_dir is not None:
        updates.setdefault("run", {})["run_root_dir"] = str(args.run_root_dir)
    if args.run_name is not None:
        updates.setdefault("run", {})["run_name"] = args.run_name
    if args.batch_size is not None:
        updates.setdefault("training", {})["batch_size"] = args.batch_size
    if args.max_steps is not None:
        updates.setdefault("training", {})["max_steps"] = args.max_steps
    if args.learning_rate is not None:
        updates.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        updates.setdefault("training", {})["weight_decay"] = args.weight_decay
    if args.log_every is not None:
        updates.setdefault("training", {})["log_every"] = args.log_every
    if args.eval_every is not None:
        updates.setdefault("training", {})["eval_every"] = args.eval_every
    if args.save_every is not None:
        updates.setdefault("training", {})["save_every"] = args.save_every
    if args.eval_batches is not None:
        updates.setdefault("training", {})["eval_batches"] = args.eval_batches
    if args.seed is not None:
        updates.setdefault("training", {})["seed"] = args.seed
    if args.device is not None:
        updates.setdefault("training", {})["device"] = args.device
    if args.wandb_entity is not None:
        updates.setdefault("wandb", {})["entity"] = args.wandb_entity
    if args.wandb_project is not None:
        updates.setdefault("wandb", {})["project"] = args.wandb_project
    if args.wandb_mode is not None:
        updates.setdefault("wandb", {})["mode"] = args.wandb_mode
    if args.ae_version is not None:
        updates.setdefault("model", {})["ae_version"] = args.ae_version
    if args.latent_dim is not None:
        updates.setdefault("model", {})["latent_dim"] = args.latent_dim
    if args.encoder_layers is not None:
        updates.setdefault("model", {})["encoder_layers"] = args.encoder_layers
    if args.decoder_layers is not None:
        updates.setdefault("model", {})["decoder_layers"] = args.decoder_layers
    if args.perception_layers is not None:
        updates.setdefault("model", {})["perception_layers"] = args.perception_layers
    if args.vlm_path is not None:
        updates.setdefault("vla", {})["vlm_path"] = args.vlm_path
    if args.vla_config_file_path is not None:
        updates.setdefault("vla", {})["config_file_path"] = args.vla_config_file_path
    if args.num_images_in_input is not None:
        updates.setdefault("vla", {})["num_images_in_input"] = args.num_images_in_input
    if args.mask_mode is not None:
        updates.setdefault("model", {})["mask_mode"] = args.mask_mode
    if args.mask_prob is not None:
        updates.setdefault("model", {})["mask_prob"] = args.mask_prob
    if args.mask_count is not None:
        updates.setdefault("model", {})["mask_count"] = args.mask_count
    if args.noise_std is not None:
        updates.setdefault("model", {})["noise_std"] = args.noise_std
    if args.episode_split_file is not None:
        updates.setdefault("data", {})["episode_split_file"] = args.episode_split_file
    if args.data_contract_file is not None:
        updates.setdefault("data", {})["data_contract_file"] = args.data_contract_file
    return deep_update(config, updates)


@dataclass(frozen=True)
class DistributedState:
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int
    is_main_process: bool
    device: torch.device


def init_distributed_state(device_name: Optional[str]) -> DistributedState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1

    if is_distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        rank = 0
        local_rank = 0
        if device_name is not None:
            device = torch.device(device_name)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return DistributedState(
        is_distributed=is_distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_main_process=rank == 0,
        device=device,
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def distributed_barrier(distributed_state: DistributedState) -> None:
    if distributed_state.is_distributed:
        dist.barrier()


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def wrap_model_for_training(model: torch.nn.Module, distributed_state: DistributedState) -> torch.nn.Module:
    if not distributed_state.is_distributed:
        return model
    if distributed_state.device.type == "cuda":
        return DDP(model, device_ids=[distributed_state.local_rank], output_device=distributed_state.local_rank)
    return DDP(model)


def reduce_metrics(metrics: Dict[str, float], device: torch.device, distributed_state: DistributedState) -> Dict[str, float]:
    if not distributed_state.is_distributed:
        return metrics

    reduced: Dict[str, float] = {}
    for key, value in metrics.items():
        if key == "step":
            reduced[key] = value
            continue
        if isinstance(value, (int, float)):
            tensor = torch.tensor(float(value), device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reduced[key] = (tensor / distributed_state.world_size).item()
        else:
            reduced[key] = value
    return reduced


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_scheduler(optimizer: torch.optim.Optimizer, *, warmup_steps: int, max_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        decay_steps = max(1, max_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def compute_metrics(actions: torch.Tensor, recon: torch.Tensor, latents: torch.Tensor, prefix: str) -> Dict[str, float]:
    error = (recon - actions).abs().float()
    metrics: Dict[str, float] = {
        f"{prefix}/l1": error.mean().item(),
        f"{prefix}/current_l1": error[:, 0].mean().item(),
        f"{prefix}/future_l1": error[:, 1:].mean().item(),
        f"{prefix}/latent_mean": latents.float().mean().item(),
        f"{prefix}/latent_std": latents.float().std(unbiased=False).item(),
        f"{prefix}/latent_norm": latents.float().norm(dim=-1).mean().item(),
    }
    per_dim_l1 = error.mean(dim=(0, 1))
    for dim, value in enumerate(per_dim_l1):
        metrics[f"{prefix}/dim_{dim}_l1"] = value.item()
    return metrics


@torch.no_grad()
def evaluate(
    *,
    model: ActionTransformerAE,
    eval_iterable: Iterable[Dict[str, Any]],
    device: torch.device,
    max_batches: int,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    count = 0

    for batch in eval_iterable:
        actions = batch["actions"].to(device, non_blocking=True)
        recon, latents = model(actions)
        batch_metrics = compute_metrics(actions, recon, latents, "eval")
        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
        if count >= max_batches:
            break

    if count == 0:
        raise RuntimeError("Evaluation iterable produced no batches.")

    return {key: value / count for key, value in totals.items()}


def init_wandb(config: Dict[str, Any], run_name: str):
    wandb_config = config.get("wandb", {})
    mode = os.environ.get("WANDB_MODE", wandb_config.get("mode", "online"))
    os.environ["WANDB_MODE"] = mode

    try:
        import wandb
    except ImportError:
        if mode == "disabled":
            return None
        raise

    return wandb.init(
        entity=wandb_config.get("entity", "kaixi-university-of-maryland"),
        project=wandb_config.get("project", "PAIR"),
        name=run_name,
        mode=mode,
        config=config,
    )


def add_step_suffix(path: Path, step: int) -> Path:
    return path.with_name(f"{path.stem}_{step}-steps{path.suffix}")


def rename_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    replace_map = [
        ("vision_backbone.dino_featurizer", "vision_backbone.featurizer"),
        ("vision_backbone.siglip_featurizer", "vision_backbone.fused_featurizer"),
        ("llm_backbone.llm", "language_model"),
        ("projector.projector.0", "projector.fc1"),
        ("projector.projector.2", "projector.fc2"),
        ("projector.projector.4", "projector.fc3"),
        ("gamma", "scale_factor"),
    ]
    renamed = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in replace_map:
            if old in new_key:
                new_key = new_key.replace(old, new)
        renamed[new_key] = value
    return renamed


def load_frozen_vla_for_perception(vla_cfg: Dict[str, Any], device: torch.device):
    config_file_path = str(vla_cfg.get("config_file_path", "/data/kaixi/PAIR/VLA-Adapter/pretrained_models/configs")).rstrip("/")
    vlm_path = str(vla_cfg.get("vlm_path", "/data/kaixi/PAIR/VLA-Adapter/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"))
    use_minivlm = bool(vla_cfg.get("use_minivlm", True))
    num_images_in_input = int(vla_cfg.get("num_images_in_input", 2))

    if model_is_on_hf_hub(config_file_path):
        config_file_path = snapshot_download(repo_id=config_file_path)
    else:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        update_auto_map(config_file_path)
        check_model_logic_mismatch(config_file_path)

    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    processor = AutoProcessor.from_pretrained(config_file_path, trust_remote_code=True)

    if use_minivlm:
        if "prism-qwen25-extra-dinosiglip-224px-0_5b" in vlm_path:
            vlm = load(vlm_path, hf_token="", load_for_training=True)
        else:
            vlm = load_vla(vlm_path, hf_token="", load_for_training=True)
        hf_config = AutoConfig.from_pretrained(str(Path(config_file_path) / "config.json"))
        vla = AutoModelForVision2Seq.from_config(hf_config, torch_dtype=torch.bfloat16).to(device)
        vla.load_state_dict(rename_state_dict_keys(vlm.state_dict()), strict=False)
        del vlm
    else:
        vla = AutoModelForVision2Seq.from_pretrained(
            config_file_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
            trust_remote_code=False,
        ).to(device)

    vla.vision_backbone.set_num_images_in_input(num_images_in_input)
    vla.eval()
    for param in vla.parameters():
        param.requires_grad_(False)
    return vla, processor


def make_v2_dataloader(
    *,
    data_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    vla_cfg: Dict[str, Any],
    vla,
    processor,
    train: bool,
) -> Tuple[DataLoader, Dict[str, Any]]:
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    use_wrist_image = int(vla_cfg.get("num_images_in_input", 2)) > 1
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=False,
        use_minivlm=bool(vla_cfg.get("use_minivlm", True)),
    )
    dataset_train = train or bool(data_cfg.get("eval_uses_train_split", False))
    dataset = RLDSDataset(
        Path(data_cfg.get("data_root_dir", "/data/kaixi/dataset/libero")),
        data_cfg.get("mixture", "libero_4_task_suites_no_noops"),
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=int(data_cfg.get("shuffle_buffer_size", 100_000 if train else 10_000)),
        train=dataset_train,
        image_aug=bool(data_cfg.get("image_aug", train)) if train else False,
        validation_split_percent=int(data_cfg.get("validation_split_percent", 0)),
        episode_split_file=data_cfg.get("episode_split_file"),
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_cfg.get("batch_size", 8)),
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    return loader, dataset.dataset_statistics


@torch.no_grad()
def extract_perception_tokens(
    *,
    vla,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    num_patches: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    autocast_context = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
    with autocast_context:
        output = vla(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(device=device, dtype=torch.bfloat16),
            labels=batch["labels"].to(device),
            output_hidden_states=True,
            proprio=None,
            proprio_projector=None,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=False,
        )
    return build_perception_tokens(
        hidden_state=output.hidden_states[0],
        labels=batch["labels"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        num_patches=num_patches,
    )


def compute_latent_adjacent_cosine(latents: torch.Tensor) -> float:
    if latents.shape[1] < 2:
        return 0.0
    normalized = F.normalize(latents.float(), dim=-1)
    return (normalized[:, :-1] * normalized[:, 1:]).sum(dim=-1).mean().item()


def compute_v2_metrics(
    *,
    actions: torch.Tensor,
    recon: torch.Tensor,
    latents: torch.Tensor,
    action_mask: torch.Tensor,
    prefix: str,
) -> Dict[str, float]:
    error = (recon - actions).abs().float()
    metrics: Dict[str, float] = {
        f"{prefix}/l1": error.mean().item(),
        f"{prefix}/current_l1": error[:, 0].mean().item(),
        f"{prefix}/future_l1": error[:, 1:].mean().item(),
        f"{prefix}/latent_std": latents.float().std(unbiased=False).item(),
        f"{prefix}/latent_norm": latents.float().norm(dim=-1).mean().item(),
        f"{prefix}/latent_adjacent_cosine": compute_latent_adjacent_cosine(latents),
        f"{prefix}/mask_ratio": action_mask.float().mean().item(),
    }
    if action_mask.any():
        metrics[f"{prefix}/masked_step_l1"] = error[action_mask].mean().item()
    else:
        metrics[f"{prefix}/masked_step_l1"] = 0.0
    unmasked = ~action_mask
    if unmasked.any():
        metrics[f"{prefix}/unmasked_step_l1"] = error[unmasked].mean().item()
    else:
        metrics[f"{prefix}/unmasked_step_l1"] = 0.0
    return metrics


@torch.no_grad()
def evaluate_v2(
    *,
    model: ActionPerceptionTransformerAE,
    vla,
    eval_loader: DataLoader,
    device: torch.device,
    num_patches: int,
    max_batches: int,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    count = 0
    for batch in eval_loader:
        actions = batch["actions"].to(device, non_blocking=True).float()
        perception_tokens, perception_mask = extract_perception_tokens(
            vla=vla,
            batch=batch,
            device=device,
            num_patches=num_patches,
        )
        corrupted, action_mask = corrupt_actions(
            actions,
            mask_prob=model.config.mask_prob,
            noise_std=model.config.noise_std,
            mask_mode=model.config.mask_mode,
            mask_count=model.config.mask_count,
            training=True,
        )
        latents = model.encode(corrupted, perception_tokens, perception_mask)
        recon = model.decode(latents)
        batch_metrics = compute_v2_metrics(
            actions=actions,
            recon=recon,
            latents=latents,
            action_mask=action_mask,
            prefix="eval",
        )

        zero_tokens = torch.zeros_like(perception_tokens)
        zero_latents = model.encode(corrupted, zero_tokens, perception_mask)
        zero_recon = model.decode(zero_latents)
        batch_metrics["eval/mask_perception_l1"] = F.l1_loss(zero_recon, actions).item()

        if actions.shape[0] > 1:
            perm = torch.randperm(actions.shape[0], device=device)
            shuffled_latents = model.encode(corrupted, perception_tokens[perm], perception_mask[perm])
            shuffled_recon = model.decode(shuffled_latents)
            batch_metrics["eval/shuffle_perception_l1"] = F.l1_loss(shuffled_recon, actions).item()
        else:
            batch_metrics["eval/shuffle_perception_l1"] = batch_metrics["eval/l1"]

        for key, value in batch_metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
        if count >= max_batches:
            break
    if count == 0:
        raise RuntimeError("Evaluation dataloader produced no batches.")
    return {key: value / count for key, value in totals.items()}


def train_v2(
    *,
    config: Dict[str, Any],
    run_dir: Path,
    run_name: str,
    wandb_run,
    distributed_state: DistributedState,
) -> None:
    training_cfg = config.setdefault("training", {})
    model_cfg = config.setdefault("model", {})
    data_cfg = config.setdefault("data", {})
    vla_cfg = config.setdefault("vla", {})

    device = distributed_state.device
    vla, processor = load_frozen_vla_for_perception(vla_cfg, device)
    num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    text_config = getattr(vla.config, "text_config", None)
    perception_dim = int(getattr(vla, "llm_dim", getattr(text_config, "hidden_size", 896)))
    model_cfg["perception_dim"] = perception_dim
    model_cfg.setdefault("latent_dim", 16)
    model_cfg.setdefault("mask_mode", "random")
    model_cfg.setdefault("mask_prob", 0.3)
    model_cfg.setdefault("mask_count", 4)
    model_cfg.setdefault("noise_std", 0.05)

    ae_config = ActionPerceptionAEConfig.from_dict(model_cfg)
    model = ActionPerceptionTransformerAE(ae_config).to(device)
    model = wrap_model_for_training(model, distributed_state)
    if distributed_state.is_main_process:
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    train_loader, dataset_statistics = make_v2_dataloader(
        data_cfg=data_cfg,
        training_cfg=training_cfg,
        vla_cfg=vla_cfg,
        vla=vla,
        processor=processor,
        train=True,
    )
    eval_loader, _ = make_v2_dataloader(
        data_cfg=data_cfg,
        training_cfg=training_cfg,
        vla_cfg=vla_cfg,
        vla=vla,
        processor=processor,
        train=False,
    )
    data_contract = None
    if distributed_state.is_main_process:
        save_json(run_dir / "dataset_statistics.json", dataset_statistics)
        data_contract_path = data_cfg.get("data_contract_file")
        if data_contract_path:
            with Path(data_contract_path).open("r", encoding="utf-8") as f:
                data_contract = json.load(f)
            save_json(run_dir / "data_contract.json", data_contract)

    optimizer = AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    max_steps = int(training_cfg.get("max_steps", 50_000))
    scheduler = make_scheduler(
        optimizer,
        warmup_steps=int(training_cfg.get("warmup_steps", 1000)),
        max_steps=max_steps,
    )
    log_every = int(training_cfg.get("log_every", 100))
    eval_every = int(training_cfg.get("eval_every", 2000))
    save_every = int(training_cfg.get("save_every", 10000))
    eval_batches = int(training_cfg.get("eval_batches", 20))
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 1.0))

    best_eval_l1 = float("inf")
    metrics_path = run_dir / "metrics.jsonl"
    train_iterator = iter(train_loader)
    start_time = time.time()

    if distributed_state.is_main_process:
        print(f"[action_ae_v2] run_dir: {run_dir}")
        print(f"[action_ae_v2] device: {device}")
        print(f"[action_ae_v2] distributed: world_size={distributed_state.world_size}")
        print(f"[action_ae_v2] max_steps: {max_steps}")
        print(f"[action_ae_v2] batch_size_per_rank: {training_cfg.get('batch_size', 8)}")
        print(f"[action_ae_v2] effective_batch_size: {int(training_cfg.get('batch_size', 8)) * distributed_state.world_size}")
        print(f"[action_ae_v2] latent_dim: {ae_config.latent_dim}")
        print(
            "[action_ae_v2] layers: "
            f"encoder={ae_config.encoder_layers} "
            f"perception={ae_config.perception_layers} "
            f"decoder={ae_config.decoder_layers}"
        )
        print(
            f"[action_ae_v2] mask_mode: {ae_config.mask_mode} "
            f"mask_prob: {ae_config.mask_prob} "
            f"mask_count: {ae_config.mask_count} "
            f"noise_std: {ae_config.noise_std}"
        )

    for step in range(1, max_steps + 1):
        model.train()
        batch = next(train_iterator)
        actions = batch["actions"].to(device, non_blocking=True).float()
        perception_tokens, perception_mask = extract_perception_tokens(
            vla=vla,
            batch=batch,
            device=device,
            num_patches=num_patches,
        )

        optimizer.zero_grad(set_to_none=True)
        output = model(actions, perception_tokens, perception_mask, corrupt=True)
        loss = F.l1_loss(output.recon_actions, actions)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()

        if step % log_every == 0 or step == 1:
            lr = scheduler.get_last_lr()[0]
            metrics = compute_v2_metrics(
                actions=actions.detach(),
                recon=output.recon_actions.detach(),
                latents=output.latents.detach(),
                action_mask=output.action_mask.detach(),
                prefix="train",
            )
            metrics.update(
                {
                    "step": step,
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "time/elapsed_sec": time.time() - start_time,
                }
            )
            metrics = reduce_metrics(metrics, device, distributed_state)
            if distributed_state.is_main_process:
                append_jsonl(metrics_path, metrics)
                if wandb_run is not None:
                    wandb_run.log(metrics, step=step)
                print(f"[action_ae_v2] step={step} loss={metrics['train/loss']:.6f} lr={lr:.3e}")

        if step % eval_every == 0 or step == max_steps:
            if distributed_state.is_main_process:
                module = unwrap_model(model)
                eval_metrics = evaluate_v2(
                    model=module,
                    vla=vla,
                    eval_loader=eval_loader,
                    device=device,
                    num_patches=num_patches,
                    max_batches=eval_batches,
                )
                eval_metrics["step"] = step
                append_jsonl(metrics_path, eval_metrics)
                if wandb_run is not None:
                    wandb_run.log(eval_metrics, step=step)
                eval_l1 = eval_metrics["eval/l1"]
                print(f"[action_ae_v2] eval step={step} l1={eval_l1:.6f}")
                if eval_l1 < best_eval_l1:
                    best_eval_l1 = eval_l1
                    best_checkpoint_path = add_step_suffix(run_dir / "checkpoint_best.pt", step)
                    best_encoder_path = add_step_suffix(run_dir / "encoder.pt", step)
                    save_training_checkpoint(
                        path=best_checkpoint_path,
                        model=module,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        best_eval_l1=best_eval_l1,
                        config=config,
                    )
                    save_encoder_checkpoint(
                        path=best_encoder_path,
                        encoder=module.encoder,
                        config=ae_config,
                        metadata={
                            "step": step,
                            "eval_l1": best_eval_l1,
                            "run_name": run_name,
                            "data_contract": data_contract,
                        },
                    )
                    save_encoder_checkpoint(
                        path=run_dir / "encoder_best.pt",
                        encoder=module.encoder,
                        config=ae_config,
                        metadata={
                            "step": step,
                            "eval_l1": best_eval_l1,
                            "run_name": run_name,
                            "data_contract": data_contract,
                        },
                    )
            distributed_barrier(distributed_state)

        if step % save_every == 0 or step == max_steps:
            if distributed_state.is_main_process:
                latest_checkpoint_path = add_step_suffix(run_dir / "checkpoint_latest.pt", step)
                save_training_checkpoint(
                    path=latest_checkpoint_path,
                    model=unwrap_model(model),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_eval_l1=best_eval_l1,
                    config=config,
                )
            distributed_barrier(distributed_state)

    if distributed_state.is_main_process and wandb_run is not None:
        wandb_run.finish()


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args)

    training_cfg = config.setdefault("training", {})
    run_cfg = config.setdefault("run", {})
    model_cfg = config.setdefault("model", {})
    data_cfg = config.setdefault("data", {})

    distributed_state = init_distributed_state(training_cfg.get("device"))
    seed = int(training_cfg.get("seed", 7))
    set_seed(seed + distributed_state.rank)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_cfg.get("run_name") or f"action_ae_libero_all_{timestamp}"
    run_root_dir = Path(run_cfg.get("run_root_dir", "/umd-datapool/kaixi/PAIR/action_ae_runs"))
    run_dir = run_root_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if distributed_state.is_main_process:
        with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    wandb_run = init_wandb(config, run_name) if distributed_state.is_main_process else None

    ae_version = str(model_cfg.get("ae_version", "v1")).lower()
    if ae_version == "v2":
        train_v2(
            config=config,
            run_dir=run_dir,
            run_name=run_name,
            wandb_run=wandb_run,
            distributed_state=distributed_state,
        )
        cleanup_distributed()
        return
    if ae_version != "v1":
        raise ValueError(f"Unsupported Action AE version: {ae_version!r}")

    action_ae_config = ActionAEConfig.from_dict(model_cfg)
    model = ActionTransformerAE(action_ae_config)

    device = distributed_state.device
    model.to(device)
    model = wrap_model_for_training(model, distributed_state)

    data_config = ActionDataConfig(
        data_root_dir=data_cfg.get("data_root_dir", "/data/kaixi/dataset/libero"),
        mixture=data_cfg.get("mixture", "libero_4_task_suites_no_noops"),
        train_split=data_cfg.get("train_split", "train[:95%]"),
        eval_split=data_cfg.get("eval_split", "train[95%:]"),
        batch_size=int(training_cfg.get("batch_size", data_cfg.get("batch_size", 1024))),
        shuffle_buffer_size=int(data_cfg.get("shuffle_buffer_size", 256_000)),
        traj_transform_threads=data_cfg.get("traj_transform_threads", 4),
        traj_read_threads=data_cfg.get("traj_read_threads", 4),
        balance_weights=bool(data_cfg.get("balance_weights", True)),
        include_dataset_name=bool(data_cfg.get("include_dataset_name", True)),
    )
    train_iterable, eval_iterable, dataset_statistics = make_action_iterables(data_config)
    if distributed_state.is_main_process:
        save_json(run_dir / "dataset_statistics.json", dataset_statistics)

    optimizer = AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    max_steps = int(training_cfg.get("max_steps", 100_000))
    scheduler = make_scheduler(
        optimizer,
        warmup_steps=int(training_cfg.get("warmup_steps", 1000)),
        max_steps=max_steps,
    )

    log_every = int(training_cfg.get("log_every", 100))
    eval_every = int(training_cfg.get("eval_every", 5000))
    save_every = int(training_cfg.get("save_every", 10000))
    eval_batches = int(training_cfg.get("eval_batches", 20))
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 1.0))

    best_eval_l1 = float("inf")
    metrics_path = run_dir / "metrics.jsonl"
    train_iterator = iter(train_iterable)
    start_time = time.time()

    if distributed_state.is_main_process:
        print(f"[action_ae] run_dir: {run_dir}")
        print(f"[action_ae] device: {device}")
        print(f"[action_ae] distributed: world_size={distributed_state.world_size}")
        print(f"[action_ae] max_steps: {max_steps}")
        print(f"[action_ae] batch_size_per_rank: {data_config.batch_size}")
        print(f"[action_ae] effective_batch_size: {data_config.batch_size * distributed_state.world_size}")

    for step in range(1, max_steps + 1):
        model.train()
        batch = next(train_iterator)
        actions = batch["actions"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        recon, latents = model(actions)
        loss = F.l1_loss(recon, actions)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()

        if step % log_every == 0 or step == 1:
            lr = scheduler.get_last_lr()[0]
            metrics = compute_metrics(actions.detach(), recon.detach(), latents.detach(), "train")
            metrics.update(
                {
                    "step": step,
                    "train/loss": loss.item(),
                    "train/lr": lr,
                    "time/elapsed_sec": time.time() - start_time,
                }
            )
            metrics = reduce_metrics(metrics, device, distributed_state)
            if distributed_state.is_main_process:
                append_jsonl(metrics_path, metrics)
                if wandb_run is not None:
                    wandb_run.log(metrics, step=step)
                print(f"[action_ae] step={step} loss={metrics['train/loss']:.6f} lr={lr:.3e}")

        if step % eval_every == 0 or step == max_steps:
            if distributed_state.is_main_process:
                module = unwrap_model(model)
                eval_metrics = evaluate(
                    model=module,
                    eval_iterable=eval_iterable,
                    device=device,
                    max_batches=eval_batches,
                )
                eval_metrics["step"] = step
                append_jsonl(metrics_path, eval_metrics)
                if wandb_run is not None:
                    wandb_run.log(eval_metrics, step=step)
                eval_l1 = eval_metrics["eval/l1"]
                print(f"[action_ae] eval step={step} l1={eval_l1:.6f}")

                if eval_l1 < best_eval_l1:
                    best_eval_l1 = eval_l1
                    best_checkpoint_path = add_step_suffix(run_dir / "checkpoint_best.pt", step)
                    best_encoder_path = add_step_suffix(run_dir / "encoder.pt", step)
                    save_training_checkpoint(
                        path=best_checkpoint_path,
                        model=module,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        step=step,
                        best_eval_l1=best_eval_l1,
                        config=config,
                    )
                    save_encoder_checkpoint(
                        path=best_encoder_path,
                        encoder=module.encoder,
                        config=action_ae_config,
                        metadata={"step": step, "eval_l1": best_eval_l1, "run_name": run_name},
                    )
                    save_encoder_checkpoint(
                        path=run_dir / "encoder_best.pt",
                        encoder=module.encoder,
                        config=action_ae_config,
                        metadata={"step": step, "eval_l1": best_eval_l1, "run_name": run_name},
                    )
            distributed_barrier(distributed_state)

        if step % save_every == 0 or step == max_steps:
            if distributed_state.is_main_process:
                latest_checkpoint_path = add_step_suffix(run_dir / "checkpoint_latest.pt", step)
                save_training_checkpoint(
                    path=latest_checkpoint_path,
                    model=unwrap_model(model),
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    best_eval_l1=best_eval_l1,
                    config=config,
                )
            distributed_barrier(distributed_state)

    if distributed_state.is_main_process and wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
