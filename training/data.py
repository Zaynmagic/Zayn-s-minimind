# Read JSONL and make masked token batches.

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import torch


class LazyBatches(Sequence):
    """Materialize only the batches requested by the current optimizer window."""

    def __init__(self, count, build_batch):
        if count < 0:
            raise ValueError("batch count cannot be negative")
        self.count = count
        self.build_batch = build_batch

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self.build_batch(position) for position in range(*index.indices(self.count))]
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError("batch index out of range")
        return self.build_batch(index)


def read_records(path):
    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        records.append(row)
    if not records:
        raise ValueError(f"{path}: no records")
    return records


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("conversations must be a nonempty list")
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        if message.get("role") not in {"system", "user", "assistant"}:
            raise ValueError("supported roles: system, user, assistant")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("each message needs nonempty string content")
        if "<|im_start|>" in content or "<|im_end|>" in content:
            raise ValueError("chat boundary tokens cannot occur inside message content")


def encode_record(row, tokenizer, length, pretrain=False):
    if length < 4:
        raise ValueError("sequence length must be at least 4")
    if pretrain:
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("pretraining records require nonempty 'text'")
        ids = [tokenizer.bos_token_id] + tokenizer.encode(
            row["text"], add_special_tokens=False
        )[: length - 2]
        ids.append(tokenizer.eos_token_id)
        labels = ids.copy()
    else:
        messages = row.get("conversations")
        validate_messages(messages)
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        ids = tokenizer.encode(text, add_special_tokens=False)[:length]
        labels = [-100] * len(ids)
        marker = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        i = 0
        while i < len(ids):
            if ids[i : i + len(marker)] == marker:
                i += len(marker)
                while i < len(ids):
                    labels[i] = ids[i]
                    i += 1
                    if ids[i - 1] == tokenizer.eos_token_id:
                        break
            else:
                i += 1
        if not any(value != -100 for value in labels[1:]):
            raise ValueError("no assistant targets remain after truncation; increase sequence length")
    padding = length - len(ids)
    attention_mask = [1] * len(ids) + [0] * padding
    ids += [tokenizer.pad_token_id] * padding
    labels += [-100] * padding
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }


def stack_rows(rows, tokenizer, length, pretrain=False):
    if not rows:
        raise ValueError("cannot encode an empty batch")
    encoded = [encode_record(row, tokenizer, length, pretrain) for row in rows]
    return {key: torch.stack([row[key] for row in encoded]) for key in encoded[0]}


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
