"""Generate one greedy reply from a local model export."""

import argparse
import json

import torch
from transformers import AutoTokenizer

from model.model_minimind import MiniMindForCausalLM


@torch.inference_mode()
def reply(model, tokenizer, prompt, max_new_tokens=16):
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    inputs = torch.tensor([ids], device=model.device)
    output = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0, len(ids) :], skip_special_tokens=True).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="What is 2 + 3?")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    torch.set_num_threads(2)
    model = MiniMindForCausalLM.from_pretrained(args.model).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    answer = reply(model, tokenizer, args.prompt, args.max_new_tokens)
    print(json.dumps({"prompt": args.prompt, "answer": answer}, indent=2))
