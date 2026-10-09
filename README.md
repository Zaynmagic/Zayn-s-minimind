# Zayn's MiniMind

A compact PyTorch project for training a small decoder language model. It includes pretraining, supervised fine-tuning, LoRA, DPO, PPO, GRPO, mixture-of-experts training, and distillation through one command-line pipeline.

The architecture and tokenizer are adapted from [MiniMind](https://github.com/jingyaogong/minimind). This implementation adds lazy training batches, explicit attention masks for cached and padded sequences, configuration checks, resumable training, and focused regression tests. Model parameter names and tokenizer IDs stay compatible with the source model. See [NOTICE](NOTICE) for attribution and changes.

## Install

Use Python 3.10 or newer and install a PyTorch build suitable for your CPU or CUDA device:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick check

The repository includes small English arithmetic fixtures. They check the training code; they do not train a useful assistant.

```bash
python -m pytest -q
python -m training.run --steps 2 --reward synthetic_token_distance --allow-random-teacher --output runs/smoke
python -m training.evaluate --model runs/smoke/hf --data data/synthetic/eval.jsonl --output runs/smoke/evaluation.json
```

The smoke command runs all eight stages with a tiny configuration. Its token-distance reward and randomly initialized teacher are test fixtures. The MoE stage starts a new model, and the final distillation stage continues that MoE model. Earlier dense checkpoints remain in the run folder.

## Train

```bash
python -m training.run --config configs/64m.json --stages pretrain,sft --data /path/to/corpus --steps 10000 --batch-size 8 --length 512 --accumulation 4 --precision bf16 --output runs/dense
```

`configs/tiny.json` is for quick checks; `configs/64m.json` configures the larger dense model. `--steps` counts microbatches, and input rows cycle when the dataset ends. Only the current accumulation window is encoded and moved to the device; the JSONL records themselves are held in host memory.

The corpus directory needs one file per selected data type:

- `pretrain.jsonl`: `{"text":"A paragraph of training text."}`
- `sft.jsonl`: `{"conversations":[{"role":"user","content":"What is 2 + 3?"},{"role":"assistant","content":"5"}]}`
- `dpo.jsonl`: objects with `chosen` and `rejected` conversation arrays in the same message format.

PPO and GRPO use the final assistant answer in each SFT row for an exact-match reward and require `--accumulation 1`. Replace that reward for open-ended tasks. Distillation requires `--teacher-checkpoint` with model weights and configuration; the teacher and student must share token IDs. Use `--stages moe` for a separate mixture-of-experts run.

Start a dense stage from a checkpoint with `--init-checkpoint runs/dense/sft.pt`. The same dense checkpoint can initialize the companion [Zayn's MiniMind-V](https://github.com/Zaynmagic/Zayn-s-minimindv) project.

## Resume and distributed training

```bash
python -m training.run --config configs/64m.json --stages sft --data /path/to/corpus --resume runs/dense/sft.pt --steps 1000 --output runs/resumed
python -m torch.distributed.run --standalone --nproc_per_node=2 -m training.run --stages pretrain,sft,moe --steps 3 --accumulation 2 --output runs/ddp
```

Resume supports pretrain, SFT, and MoE and restores the optimizer, scaler, RNG state, and data offset. Keep the same configuration, dataset, batch size, and accumulation setting, and resume at an optimizer boundary. Distributed training uses Gloo on CPU and NCCL on CUDA. Load checkpoint files only from sources you trust; training checkpoints include Python state.

Each run saves stage checkpoints, `metrics.json`, and a Hugging Face export under `hf/`. The export loads with `MiniMindForCausalLM.from_pretrained(...)`; `training.evaluate` shows the complete loading path. Checkpoints and run outputs are excluded from Git.

## Layout

- `model/`: decoder, LoRA adapters, and the 6,400-token BPE tokenizer.
- `training/`: data preparation, objectives, optimizer engine, training CLI, and evaluation.
- `configs/`: tiny and larger model configurations.
- `data/synthetic/`: generated English examples; regenerate with `python -m training.fixtures`.
- `tests/`: masking, cached decoding, optimizer, objective, export, and resume checks.

Apache-2.0. See [LICENSE](LICENSE).
