# DPO, PPO, GRPO, and distillation losses.

import torch
from torch.nn import functional as F


def token_logp(logits, labels):
    targets = labels[:, 1:]
    mask = targets.ne(-100)
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    selected = logp.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return selected, mask


def masked_mean(values, mask):
    count = mask.sum()
    if not count:
        raise ValueError("objective has no supervised tokens")
    return (values * mask).sum() / count


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta=0.1):
    margin = beta * ((chosen - rejected) - (ref_chosen - ref_rejected).detach())
    return -F.logsigmoid(margin).mean()


def distill_loss(student_logits, teacher_logits, labels, temperature=2.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    mask = labels[:, 1:].ne(-100)
    student = F.log_softmax(student_logits[:, :-1].float() / temperature, -1)
    teacher = F.softmax(teacher_logits[:, :-1].detach().float() / temperature, -1)
    kl = F.kl_div(student, teacher, reduction="none").sum(-1) * temperature**2
    return masked_mean(kl, mask)


def gae(rewards, values, mask, gamma=1.0, lam=0.95):
    # Completions are right-padded; don't bootstrap past the last token.
    rewards, values = rewards.detach(), values.detach()
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros(rewards.size(0), device=rewards.device)
    for index in range(rewards.size(1) - 1, -1, -1):
        next_valid = (mask[:, index + 1] if index + 1 < mask.size(1) else torch.zeros_like(mask[:, index]))
        next_value = values[:, index + 1] if index + 1 < values.size(1) else 0.0
        delta = rewards[:, index] + gamma * next_value * next_valid - values[:, index]
        carry = (delta + gamma * lam * next_valid * carry) * mask[:, index]
        advantages[:, index] = carry
    return advantages, (advantages + values) * mask


def ppo_loss(logp, old_logp, advantages, mask, clip=0.2):
    ratio = (logp - old_logp.detach()).exp()
    advantages = advantages.detach()
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages)
    return -masked_mean(surrogate, mask)


def grpo_advantages(rewards, group_size):
    if group_size < 2 or rewards.numel() % group_size:
        raise ValueError("GRPO requires complete groups with at least two completions")
    grouped = rewards.detach().view(-1, group_size)
    return (
        (grouped - grouped.mean(-1, keepdim=True))
        / grouped.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    ).flatten()


def sampled_kl(logp, reference_logp):
    difference = reference_logp.detach() - logp
    return difference.exp() - difference - 1
