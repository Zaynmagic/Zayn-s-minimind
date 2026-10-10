# MiniMind

[![CI](https://github.com/Zaynmagic/Zayn-s-minimind/actions/workflows/tests.yml/badge.svg)](https://github.com/Zaynmagic/Zayn-s-minimind/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)

**64M-parameter language model · PyTorch**

A practice implementation of a decoder-only Transformer and its training pipeline, following the MiniMind architecture. The full configuration has **63,912,192 parameters** and a **6,400-token BPE tokenizer**.

## Example

Actual greedy output from the **0.51M-parameter tiny configuration**, after 500 SFT updates on the bundled arithmetic fixtures:

```text
Question: What is 2 + 3?
Answer: 5
```

This question is a training example. The run checks training, export, and chat generation; it does not measure the 64M model's language ability.

| Tiny-model check | Result |
| --- | ---: |
| Training answers matched | 32 / 32 |
| Held-out answers matched | 0 / 6 |
| Held-out loss (NLL) | 2.8479 |
| Held-out perplexity | 17.2515 |

The six held-out questions use operands outside the training range. This run memorizes the training examples and does not generalize to that split. Loss covers 48 supervised tokens, including assistant template markers. [Recorded results and run details](docs/demo-results.json).

<details>
<summary>Reproduce the example and evaluation on CPU</summary>

```bash
python -m training.run --config configs/tiny.json --stages sft \
  --data data/synthetic --steps 500 --batch-size 8 --length 96 \
  --lr 0.001 --device cpu --seed 42 --output runs/portfolio-demo
python -m examples.generate --model runs/portfolio-demo/hf --prompt "What is 2 + 3?"
python -m training.evaluate --model runs/portfolio-demo/hf \
  --data data/synthetic/eval.jsonl --output runs/portfolio-demo/evaluation.json
```

Install the runtime dependencies from the setup section first. Weights are generated locally; numerical results can vary slightly across environments.

</details>

## Project overview

| Area | What is implemented | Code |
| --- | --- | --- |
| Model architecture | Grouped-query attention, RoPE, RMSNorm, and SwiGLU | [Decoder](model/model_minimind.py), [64M config](configs/64m.json), [tokenizer](model/tokenizer.json) |
| Training pipeline | Pretraining, SFT, LoRA, DPO, PPO/GRPO, MoE, and knowledge distillation | [Training stages](training/run.py), [objectives](training/objectives.py), [LoRA](model/model_lora.py) |
| Training and inference | AMP, gradient accumulation, CPU/CUDA DDP, and KV-cached decoding | [Training engine](training/engine.py), [model and generation](model/model_minimind.py), [evaluation](training/evaluate.py) |

The project keeps stage orchestration in one pipeline and encodes batches only when needed. Dense checkpoints also initialize the companion [MiniMind-V](https://github.com/Zaynmagic/Zayn-s-minimindv) project. Model parameter names and tokenizer IDs stay compatible across the two projects.

Model and tokenizer foundations are adapted from [MiniMind](https://github.com/jingyaogong/minimind); see [NOTICE](NOTICE) for source details and modifications.

## Setup and checks

Python 3.10+. Install the PyTorch build appropriate for your CPU or CUDA device.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m training.run --steps 2 --reward synthetic_token_distance --allow-random-teacher --output runs/smoke
python -m training.evaluate --model runs/smoke/hf --data data/synthetic/eval.jsonl --output runs/smoke/evaluation.json
```

For runtime dependencies alone, use `requirements.txt`. The bundled arithmetic data, token-distance reward, and random teacher are smoke-test fixtures. Regenerate the data with `python -m training.fixtures`.

## Training

```bash
python -m training.run --config configs/64m.json --stages pretrain,sft \
  --data /path/to/corpus --steps 10000 --batch-size 8 --length 512 \
  --accumulation 4 --precision bf16 --output runs/dense
```

`configs/tiny.json` is the default for quick runs. `--steps` counts microbatches; records cycle at the end of the dataset. Encoding happens one accumulation window at a time to limit device memory. JSONL records remain in host memory.

The data directory contains the files needed by the selected stages:

- `pretrain.jsonl`: `{"text":"A paragraph of training text."}`
- `sft.jsonl`: `{"conversations":[{"role":"user","content":"What is 2 + 3?"},{"role":"assistant","content":"5"}]}`
- `dpo.jsonl`: `chosen` and `rejected` conversation arrays using the same message format.

PPO/GRPO use the last assistant answer as an exact-match target and require `--accumulation 1` and zero dropout. Distillation takes `--teacher-checkpoint` with configuration and weights sharing the student's token IDs. The MoE stage starts a fresh model because its expert weights differ from the dense model; stages after it continue that model.

## Checkpoints

Every stage saves a checkpoint; the final model is exported under `<output>/hf/`. Load an export with `MiniMindForCausalLM.from_pretrained(...)`, as shown in `training/evaluate.py`. Use `--init-checkpoint runs/dense/sft.pt` for another dense run. The same checkpoint initializes [MiniMind-V](https://github.com/Zaynmagic/Zayn-s-minimindv) through its `--llm-checkpoint` option.

```bash
python -m training.run --config configs/64m.json --stages sft --data /path/to/corpus \
  --resume runs/dense/sft.pt --steps 1000 --output runs/resumed
python -m torch.distributed.run --standalone --nproc_per_node=2 -m training.run \
  --stages pretrain,sft,moe --steps 3 --accumulation 2 --output runs/ddp
```

Resume supports pretraining, SFT, and MoE. Keep the configuration, dataset, batch size, and accumulation setting unchanged, and resume at an optimizer boundary. DDP uses Gloo on CPU and NCCL on CUDA. Training checkpoints contain Python state, so load only trusted files. Model parameter keys and tokenizer IDs are preserved from the source architecture.

Apache-2.0; see [LICENSE](LICENSE).
