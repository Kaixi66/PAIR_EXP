"""Merge a saved LoRA adapter into the MiniVLA base model offline.

The output is written to a temporary directory, validated, and only then moved
into the checkpoint directory. This keeps an interrupted merge from replacing a
previously valid model file.
"""

import json
import os
import shutil
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import torch
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models import load, load_vla


@dataclass
class ConvertConfig:
    base_checkpoint: Union[str, Path] = ""
    lora_finetuned_checkpoint_dir: Union[str, Path] = ""
    vlm_path: Union[str, Path] = ""
    use_minivla: bool = False
    device: str = "cpu"
    action_queries_checkpoint: Union[str, Path] = ""


def _renamed_minivla_state_dict(state_dict):
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
        for old, new in replace_map:
            key = key.replace(old, new)
        renamed[key] = value
    return renamed


def _load_action_queries(path: Path) -> torch.Tensor:
    """Read action queries, including from a truncated safetensors payload."""
    tensor_name = "action_queries.weight"
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(tensor_name)
    except Exception:
        # Safetensors writes the complete JSON header before tensor bytes. If a
        # later tensor was truncated, an early tensor can still be recovered.
        with path.open("rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
            metadata = header[tensor_name]
            start, end = metadata["data_offsets"]
            handle.seek(8 + header_len + start)
            raw = handle.read(end - start)
        if len(raw) != end - start:
            raise ValueError(f"Action-query bytes are incomplete in {path}")
        dtype_map = {
            "BF16": torch.bfloat16,
            "F16": torch.float16,
            "F32": torch.float32,
        }
        tensor = torch.frombuffer(bytearray(raw), dtype=dtype_map[metadata["dtype"]]).clone()
        return tensor.reshape(metadata["shape"])


def _validate_merged_model(model_path: Path) -> int:
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    if "action_queries.weight" not in keys:
        raise ValueError(f"Merged model is missing action_queries.weight: {model_path}")
    return len(keys)


@draccus.wrap()
def main(cfg: ConvertConfig) -> None:
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    checkpoint_dir = Path(cfg.lora_finetuned_checkpoint_dir).resolve()
    adapter_dir = checkpoint_dir / "lora_adapter"
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing LoRA adapter: {adapter_dir}")

    print(f"Loading base model on {cfg.device}: {cfg.vlm_path or cfg.base_checkpoint}")
    if cfg.use_minivla:
        vlm_path = str(cfg.vlm_path)
        if "prism-qwen25-extra-dinosiglip-224px-0_5b" in vlm_path:
            vlm = load(vlm_path, hf_token="", load_for_training=False)
        else:
            vlm = load_vla(vlm_path, hf_token="", load_for_training=False)
        config_path = Path(cfg.base_checkpoint)
        if config_path.is_dir():
            config_path = config_path / "config.json"
        config = AutoConfig.from_pretrained(config_path)
        vla = AutoModelForVision2Seq.from_config(config, torch_dtype=torch.bfloat16)
        state_dict = _renamed_minivla_state_dict(vlm.state_dict())
        vla.load_state_dict(state_dict, strict=False)
        del state_dict, vlm
    else:
        vla = AutoModelForVision2Seq.from_pretrained(
            cfg.base_checkpoint,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    if cfg.action_queries_checkpoint:
        source = Path(cfg.action_queries_checkpoint).resolve()
        action_queries = _load_action_queries(source)
        with torch.no_grad():
            vla.action_queries.weight.copy_(action_queries.to(vla.action_queries.weight.dtype))
        print(f"Restored action queries from: {source}")

    print("Merging LoRA weights into base model...")
    start_time = time.time()
    merged_vla = PeftModel.from_pretrained(vla, adapter_dir).to(cfg.device)
    merged_vla = merged_vla.merge_and_unload(safe_merge=True)

    temp_dir = checkpoint_dir / f".merge-repair-{os.getpid()}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=False)
    try:
        merged_vla.save_pretrained(temp_dir, safe_serialization=True)
        model_path = temp_dir / "model.safetensors"
        tensor_count = _validate_merged_model(model_path)
        for name in ("model.safetensors", "config.json", "generation_config.json"):
            source = temp_dir / name
            if source.exists():
                os.replace(source, checkpoint_dir / name)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed = time.time() - start_time
    print(f"Merge complete: {checkpoint_dir}")
    print(f"Validated tensors: {tensor_count}; elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
