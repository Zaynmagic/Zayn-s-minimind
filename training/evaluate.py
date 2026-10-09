# Check loss, perplexity, and cached decoding.

import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindForCausalLM
from .data import read_records, stack_rows


@torch.no_grad()
def greedy(model, ids, tokens=8, cache=True):
    result, past, current = ids.clone(), None, ids
    for _ in range(tokens):
        output = model(input_ids=current, past_key_values=past, use_cache=cache)
        next_id = output.logits[:, -1].argmax(-1, keepdim=True)
        result = torch.cat((result, next_id), 1)
        past = output.past_key_values if cache else None
        current = next_id if cache else result
    return result


def main(args):
    model = MiniMindForCausalLM.from_pretrained(args.model).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    torch.set_num_threads(2)
    rows = read_records(args.data)
    batches = [
        stack_rows(rows[i : i + 2], tokenizer, args.length, args.pretrain)
        for i in range(0, len(rows), 2)
    ]
    numerator, count = 0.0, 0
    with torch.no_grad():
        for batch in batches:
            output = model(**batch)
            valid = batch["labels"][:, 1:].ne(-100).sum().item()
            numerator += output.loss.item() * valid
            count += valid
        ids = torch.tensor([tokenizer.encode("What is 2 + 3?", add_special_tokens=False)])
        cached, uncached = (greedy(model, ids, cache=True), greedy(model, ids, cache=False))
    result = {
        "data": args.data,
        "supervised_tokens": count,
        "nll": numerator / count,
        "perplexity": float(torch.exp(torch.tensor(numerator / count))),
        "cached_greedy_equal": torch.equal(cached, uncached),
        "sample": tokenizer.decode(cached[0], skip_special_tokens=True),
        "interpretation": "Perplexity is for this dataset only; toy data isn't a benchmark.",
    }
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--length", type=int, default=96)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--output")
    main(parser.parse_args())
