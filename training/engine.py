from contextlib import nullcontext
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def initialize(device="auto", seed=42):
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if distributed:
        rank = int(os.environ["LOCAL_RANK"])
        if device.startswith("cuda"):
            torch.cuda.set_device(rank)
            device = f"cuda:{rank}"
        dist.init_process_group("nccl" if device.startswith("cuda") else "gloo")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    return torch.device(device)


def primary():
    return not dist.is_initialized() or dist.get_rank() == 0


def unwrap(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def wrap(model, device):
    if not dist.is_initialized():
        return model
    return DistributedDataParallel(
        model, device_ids=[device.index] if device.type == "cuda" else None
    )


def amp_context(device, precision):
    if precision == "fp32":
        return nullcontext()
    if precision == "fp16" and device.type != "cuda":
        raise ValueError("fp16 training requires CUDA; use fp32 or bf16 on CPU")
    return torch.autocast(
        device.type, dtype=torch.float16 if precision == "fp16" else torch.bfloat16
    )


def optimize(
    model,
    optimizer,
    batches,
    objective,
    device,
    precision="fp32",
    accumulation=1,
    scaler=None,
):
    if accumulation < 1:
        raise ValueError("accumulation must be positive")
    scaler = scaler or torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and precision == "fp16"
    )
    losses = []
    gradient_norms = []
    model.train()
    for start in range(0, len(batches), accumulation):
        # Lazy batches keep only this accumulation window on the device.
        window = batches[start : start + accumulation]
        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(window):
            sync = (
                model.no_sync()
                if isinstance(model, DistributedDataParallel)
                and index < len(window) - 1
                else nullcontext()
            )
            with sync, amp_context(device, precision):
                loss = objective(model, batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss")
                # Average over this window so a short final window is not underweighted.
                scaler.scale(loss / len(window)).backward()
            losses.append(loss.item())
        scaler.unscale_(optimizer)
        trainable = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite gradient norm")
        gradient_norms.append(float(norm))
        scaler.step(optimizer)
        scaler.update()
    return losses, scaler, gradient_norms


def save_checkpoint(path, model, optimizer, scaler, stage, step):
    unwrapped = unwrap(model)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": unwrapped.config.to_dict(),
        "stage": stage,
        "step": step,
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    torch.save(state, path)


def load_checkpoint(path, model, optimizer=None, scaler=None):
    # Local checkpoints include Python RNG state, so weights_only won't work.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    unwrap(model).load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    torch.set_rng_state(checkpoint["torch_rng"])
    random.setstate(checkpoint["python_rng"])
    if "cuda_rng" in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    return checkpoint


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
