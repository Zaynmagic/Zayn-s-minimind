# Make small datasets for quick training checks.

import argparse
import json
from pathlib import Path


def generate(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    pretrain, sft, dpo = [], [], []
    for a in range(8):
        for b in range(4):
            question, answer = f"What is {a} + {b}?", str(a + b)
            pretrain.append({"text": f"{question} The answer is {answer}."})
            messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
            sft.append({"conversations": messages})
            dpo.append(
                {
                    "chosen": messages,
                    "rejected": [
                        messages[0],
                        {"role": "assistant", "content": str(a + b + 1)},
                    ],
                }
            )
    for name, rows in (("pretrain", pretrain), ("sft", sft), ("dpo", dpo)):
        (destination / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    heldout = [
        {
            "conversations": [
                {"role": "user", "content": f"What is {a} + {b}?"},
                {"role": "assistant", "content": str(a + b)},
            ]
        }
        for a in (8, 9)
        for b in (5, 6, 7)
    ]
    (destination / "eval.jsonl").write_text("".join(json.dumps(row) + "\n" for row in heldout))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/synthetic")
    generate(parser.parse_args().output)
