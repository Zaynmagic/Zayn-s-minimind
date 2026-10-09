# Run the training stages and save checkpoints.

import argparse
import copy
import datetime
import json
import platform
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora
from .data import LazyBatches, read_records, stack_rows, file_hash
from .engine import (
    initialize,
    primary,
    wrap,
    unwrap,
    optimize,
    save_checkpoint,
    load_checkpoint,
    write_json,
)
from .objectives import (
    token_logp,
    masked_mean,
    dpo_loss,
    distill_loss,
    gae,
    ppo_loss,
    grpo_advantages,
    sampled_kl,
)

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("pretrain", "sft", "lora", "dpo", "ppo", "grpo", "moe", "distill")


class ActorCritic(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor
        self.config = actor.config
        self.value_head = nn.Linear(actor.config.hidden_size, 1)

    def forward(self, **kwargs):
        output = self.actor(**kwargs)
        output.value_estimates = self.value_head(output.hidden_states).squeeze(-1)
        return output


def actor_of(model):
    raw = unwrap(model)
    return raw.actor if isinstance(raw, ActorCritic) else raw


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def frozen_copy(model):
    # Rebuild it: deepcopy can fail on the MoE's saved autograd graph.
    frozen = MiniMindForCausalLM(copy.deepcopy(model.config)).to(model.device)
    frozen.load_state_dict(model.state_dict())
    return frozen.eval().requires_grad_(False)


@torch.no_grad()
def rollout(actor, rows, tokenizer, device, prompt_length, new_tokens, group_size=1, temperature=1.0):
    prompts, expected = [], []
    for row in rows:
        messages = row["conversations"]
        if messages[-1]["role"] != "assistant":
            raise ValueError("RL rows need a final assistant answer for the selected local reward")
        ids = tokenizer.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True
        )[-prompt_length:]
        if not ids:
            raise ValueError("empty rollout prompt")
        for _ in range(group_size):
            prompts.append(ids)
            expected.append(messages[-1]["content"])
    width = max(map(len, prompts))
    input_ids = torch.tensor(
        [[tokenizer.pad_token_id] * (width - len(ids)) + ids for ids in prompts],
        device=device,
    )
    attention_mask = torch.tensor(
        [[0] * (width - len(ids)) + [1] * len(ids) for ids in prompts], device=device
    )
    was_training = actor.training
    actor.eval()
    cache, current = None, input_ids
    done = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    completion_masks = []
    for _ in range(new_tokens):
        output = actor(
            input_ids=current,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
        )
        sampled = torch.multinomial((output.logits[:, -1].float() / temperature).softmax(-1), 1).squeeze(-1)
        sampled = torch.where(done, tokenizer.pad_token_id, sampled)
        valid = ~done
        completion_masks.append(valid)
        done |= sampled.eq(tokenizer.eos_token_id)
        input_ids = torch.cat((input_ids, sampled[:, None]), 1)
        attention_mask = torch.cat((attention_mask, valid.long()[:, None]), 1)
        current, cache = sampled[:, None], output.past_key_values
    actor.train(was_training)
    labels = input_ids.clone()
    labels[:, :width] = -100
    completion_mask = torch.stack(completion_masks, 1)
    labels[:, width:][~completion_mask] = -100
    return ({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}, expected, width)


def rewards_for(batch, expected, width, tokenizer, mode):
    rewards = []
    for ids, mask, target in zip(
        batch["input_ids"][:, width:], batch["attention_mask"][:, width:], expected
    ):
        tokens = ids[mask.bool()].tolist()
        prediction = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        if mode == "exact_match":
            reward = float(prediction == target.strip())
        else:
            # Toy token-distance reward, just for checking RL gradients.
            target_id = tokenizer.encode(target, add_special_tokens=False)[0]
            reward = (
                -sum(abs(token - target_id) for token in tokens)
                / max(1, len(tokens))
                / len(tokenizer)
            )
        rewards.append(reward)
    return torch.tensor(rewards, dtype=torch.float32, device=batch["input_ids"].device)


def merge_adapters(model):
    state = {key: value for key, value in model.state_dict().items() if ".lora." not in key}
    for name, module in model.named_modules():
        if hasattr(module, "lora"):
            state[name + ".weight"] = (module.weight.detach() + module.lora.B.weight @ module.lora.A.weight)
    merged = MiniMindForCausalLM(model.config).to(model.device)
    merged.load_state_dict(state)
    return merged


def execute(args):
    if args.steps < 1 or args.batch_size < 1 or args.new_tokens < 1:
        raise ValueError("steps, batch size, and new tokens must be positive")
    if args.temperature <= 0 or args.length <= args.new_tokens:
        raise ValueError("temperature must be positive and sequence length must exceed generated length")
    stages = args.stages.split(",")
    if any(stage not in STAGES for stage in stages):
        raise ValueError(f"stages must come from {STAGES}")
    if args.resume and len(stages) != 1:
        raise ValueError("resume one stage at a time")
    if args.resume and stages[0] not in {"pretrain", "sft", "moe"}:
        raise ValueError("resume currently supports pretrain, sft, and moe")
    device = initialize(args.device, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    if len(tokenizer) != 6400:
        raise ValueError("expected the upstream 6,400-token BPE tokenizer")
    config_values = json.loads(Path(args.config).read_text())
    model = MiniMindForCausalLM(MiniMindConfig(**config_values)).to(device)
    if args.init_checkpoint:
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(initial.get("model", initial))
    report = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "kind": "language_training",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "world_size": dist.get_world_size() if dist.is_initialized() else 1,
        "config": model.config.to_dict(),
        "seed": args.seed,
        "precision": args.precision,
        "accumulation": args.accumulation,
        "reward": args.reward,
        "stages": [],
        "data_hashes": {},
    }
    for stage in stages:
        if stage == "moe":
            moe_config = {**config_values, "use_moe": True, "num_experts": 4, "num_experts_per_tok": 1}
            model = MiniMindForCausalLM(MiniMindConfig(**moe_config)).to(device)
        teacher = None
        if stage == "distill":
            # Same vocab, one extra layer when using the random test teacher.
            teacher_config = {**config_values, "num_hidden_layers": config_values["num_hidden_layers"] + 1}
            teacher_state = None
            if args.teacher_checkpoint:
                teacher_state = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
                if "config" not in teacher_state:
                    raise ValueError("teacher checkpoint must include its model configuration")
                teacher_config = teacher_state["config"]
                if teacher_config.get("vocab_size") != model.config.vocab_size:
                    raise ValueError("teacher and student must share a vocabulary")
            teacher = (
                MiniMindForCausalLM(MiniMindConfig(**teacher_config))
                .to(device)
                .eval()
                .requires_grad_(False)
            )
            if teacher_state is not None:
                teacher.load_state_dict(teacher_state["model"])
            elif not args.allow_random_teacher:
                raise ValueError(
                    "distillation requires --teacher-checkpoint or explicit --allow-random-teacher"
                )
        if stage == "lora":
            model.requires_grad_(False)
            apply_lora(model, rank=args.lora_rank)
            for name, p in model.named_parameters():
                p.requires_grad_(".lora." in name)
        else:
            model.requires_grad_(True)
        model.zero_grad(set_to_none=True)
        reference = frozen_copy(model) if stage in {"dpo", "ppo", "grpo"} else None
        if stage == "ppo":
            model = ActorCritic(model).to(device)
        model = wrap(model, device)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
        raw = unwrap(model)
        start_step = 0
        resumed_scaler = None
        if args.resume:
            resumed_scaler = torch.amp.GradScaler(
                "cuda", enabled=device.type == "cuda" and args.precision == "fp16"
            )
            checkpoint = load_checkpoint(args.resume, model, optimizer, resumed_scaler)
            if checkpoint["stage"] != stage:
                raise ValueError("resume checkpoint stage does not match")
            start_step = checkpoint["step"]
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        def record_index(index, j):
            return (((start_step + index) * world_size + rank) * args.batch_size + j) % len(records)

        data_name = ("pretrain" if stage == "pretrain" else "dpo" if stage == "dpo" else "sft")
        data_path = Path(args.data) / f"{data_name}.jsonl"
        records = read_records(data_path)
        report["data_hashes"][data_name] = file_hash(data_path)
        if stage == "dpo":
            def build_batch(index):
                rows = [records[record_index(index, j)] for j in range(args.batch_size)]
                chosen = stack_rows(
                    [{"conversations": row["chosen"]} for row in rows],
                    tokenizer,
                    args.length,
                )
                rejected = stack_rows(
                    [{"conversations": row["rejected"]} for row in rows],
                    tokenizer,
                    args.length,
                )
                return to_device(
                    {key: torch.cat((chosen[key], rejected[key])) for key in chosen},
                    device,
                )

            encoded = LazyBatches(args.steps, build_batch)

            def objective(active, batch):
                output = active(**batch)
                logp, mask = token_logp(output.logits, batch["labels"])
                with torch.no_grad():
                    ref_logp, _ = token_logp(reference(**batch).logits, batch["labels"])
                scores, ref_scores = (logp * mask).sum(-1), (ref_logp * mask).sum(-1)
                half = scores.numel() // 2
                return (
                    dpo_loss(
                        scores[:half],
                        scores[half:],
                        ref_scores[:half],
                        ref_scores[half:],
                        args.beta,
                    )
                    + output.aux_loss
                )
        elif stage in {"ppo", "grpo"}:
            if args.accumulation != 1:
                raise ValueError("on-policy PPO/GRPO use --accumulation 1")
            encoded = LazyBatches(
                args.steps,
                lambda index: [records[record_index(index, j)] for j in range(args.batch_size)],
            )
            reward_summaries = []

            def objective(active, rows):
                actor = actor_of(active)
                batch, expected, width = rollout(
                    actor,
                    rows,
                    tokenizer,
                    device,
                    args.length - args.new_tokens,
                    args.new_tokens,
                    args.group_size if stage == "grpo" else 1,
                    args.temperature,
                )
                rewards = rewards_for(batch, expected, width, tokenizer, args.reward)
                reward_summaries.append(
                    {
                        "mean": rewards.mean().item(),
                        "std": rewards.std(unbiased=False).item(),
                    }
                )
                with torch.no_grad():
                    old_output = unwrap(active)(**batch)
                    old_logp, mask = token_logp(old_output.logits / args.temperature, batch["labels"])
                    ref_logp, _ = token_logp(reference(**batch).logits / args.temperature, batch["labels"])
                    if stage == "ppo":
                        values = old_output.value_estimates[:, :-1]
                        token_rewards = -args.beta * (old_logp - ref_logp) * mask
                        last = mask.long().sum(-1) + width - 2
                        token_rewards[torch.arange(mask.size(0), device=device), last] += rewards
                        advantages, returns = gae(token_rewards, values, mask)
                    else:
                        advantages = grpo_advantages(rewards, args.group_size)[:, None].expand_as(old_logp)
                output = active(**batch)
                logp, mask = token_logp(output.logits / args.temperature, batch["labels"])
                loss = ppo_loss(logp, old_logp, advantages, mask)
                if stage == "ppo":
                    loss = loss + 0.5 * masked_mean(
                        (output.value_estimates[:, :-1] - returns).square(), mask
                    )
                else:
                    loss = loss + args.beta * masked_mean(sampled_kl(logp, ref_logp), mask)
                return loss + output.aux_loss
        else:
            def build_batch(index):
                rows = [records[record_index(index, j)] for j in range(args.batch_size)]
                return to_device(
                    stack_rows(rows, tokenizer, args.length, stage == "pretrain"),
                    device,
                )

            encoded = LazyBatches(args.steps, build_batch)

            def objective(active, batch):
                output = active(**batch)
                loss = output.loss + output.aux_loss
                if teacher is not None:
                    with torch.no_grad():
                        teacher_output = teacher(**batch)
                    loss = 0.5 * loss + 0.5 * distill_loss(
                        output.logits, teacher_output.logits, batch["labels"]
                    )
                return loss

        started = time.perf_counter()
        losses, scaler, gradient_norms = optimize(
            model,
            optimizer,
            encoded,
            objective,
            device,
            args.precision,
            args.accumulation,
            scaler=resumed_scaler,
        )
        result = {
            "stage": stage,
            "microbatches": args.steps,
            "optimizer_steps": (args.steps + args.accumulation - 1)
            // args.accumulation,
            "losses": losses,
            "gradient_norms_before_clipping": gradient_norms,
            "seconds": time.perf_counter() - started,
            "parameters": sum(p.numel() for p in raw.parameters()),
            "trainable_parameters": sum(p.numel() for p in raw.parameters() if p.requires_grad),
            "teacher_random_initialization": stage == "distill"
            and not bool(args.teacher_checkpoint),
        }
        if stage in {"ppo", "grpo"}:
            result["reward_summaries"] = reward_summaries
            result["sampling_temperature"] = args.temperature
        report["stages"].append(result)
        if primary():
            save_checkpoint(
                Path(args.output) / f"{stage}.pt",
                model,
                optimizer,
                scaler,
                stage,
                start_step + args.steps,
            )
            write_json(Path(args.output) / "metrics.json", report)
            print(json.dumps(result), flush=True)
        model = unwrap(model)
        if stage == "lora":
            model = merge_adapters(model)
        if stage == "ppo":
            model = model.actor
    if primary():
        model.config.tie_word_embeddings = True
        model.config._name_or_path = ""
        model.save_pretrained(Path(args.output) / "hf", safe_serialization=True)
        tokenizer.save_pretrained(Path(args.output) / "hf")
        write_json(Path(args.output) / "metrics.json", report)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return report


def parser():
    cli = argparse.ArgumentParser(description="Train MiniMind")
    cli.add_argument("--config", default=str(ROOT / "configs/tiny.json"))
    cli.add_argument("--stages", default=",".join(STAGES))
    cli.add_argument("--data", default=str(ROOT / "data/synthetic"))
    cli.add_argument("--output", default=str(ROOT / "runs/smoke"))
    cli.add_argument("--steps", type=int, default=6)
    cli.add_argument("--batch-size", type=int, default=2)
    cli.add_argument("--length", type=int, default=96)
    cli.add_argument("--new-tokens", type=int, default=8)
    cli.add_argument("--group-size", type=int, default=4)
    cli.add_argument("--temperature", type=float, default=1.0)
    cli.add_argument("--lora-rank", type=int, default=16)
    cli.add_argument("--lr", type=float, default=5e-4)
    cli.add_argument("--beta", type=float, default=0.1)
    cli.add_argument("--accumulation", type=int, default=1)
    cli.add_argument("--device", default="auto")
    cli.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="fp32")
    cli.add_argument("--reward", choices=["exact_match", "synthetic_token_distance"], default="exact_match")
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--resume")
    cli.add_argument("--init-checkpoint")
    cli.add_argument("--teacher-checkpoint")
    cli.add_argument("--allow-random-teacher", action="store_true")
    return cli


if __name__ == "__main__":
    execute(parser().parse_args())
