# Adapted from jingyaogong/minimind under Apache-2.0; see NOTICE.
# Modified: validate adapter rank and preserve each target layer's dtype/device.

import torch
from torch import nn


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
            raise ValueError("LoRA rank must be a positive integer")
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)

        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # Zero B keeps the initial output unchanged.
        self.B.weight.data.zero_()

    def forward(self, hidden_states):
        return self.B(self.A(hidden_states))


def apply_lora(model, rank=16):
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            if hasattr(module, "lora") or ".lora." in name:
                raise ValueError("LoRA adapters have already been applied")
            adapter = LoRA(module.in_features, module.out_features, rank=rank).to(
                device=module.weight.device, dtype=module.weight.dtype
            )
            module.lora = adapter
            original_forward = module.forward

            # Bind each wrapper to its own layer.
            def forward_with_lora(
                hidden_states, projection=original_forward, residual=adapter
            ):
                return projection(hidden_states) + residual(hidden_states)

            module.forward = forward_with_lora


def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)
    state_dict = {
        (name[7:] if name.startswith("module.") else name): weights
        for name, weights in state_dict.items()
    }

    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            prefix = f"{name}.lora."
            lora_state = {
                key.replace(prefix, ""): weights
                for key, weights in state_dict.items()
                if prefix in key
            }
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, "lora"):
            clean_name = name[7:] if name.startswith("module.") else name
            lora_state = {
                f"{clean_name}.lora.{key}": weights.cpu().half()
                for key, weights in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    raw_model = getattr(model, "_orig_mod", model)
    state_dict = {
        name: weights.cpu().half()
        for name, weights in raw_model.state_dict().items()
        if ".lora." not in name
    }
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and ".lora." not in name:
            state_dict[f"{name}.weight"] = module.weight.data.clone().cpu().half()
            if hasattr(module, "lora"):
                adapter_update = module.lora.B.weight.data @ module.lora.A.weight.data
                state_dict[f"{name}.weight"] += adapter_update.cpu().half()
    torch.save(state_dict, save_path)
