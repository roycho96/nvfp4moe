"""Fine-tune one expert layer in Qwen3-30B-A3B on a real LM batch."""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from nvfp4moe import Qwen3Nvfp4Experts


def training_batch(tokenizer, tokens):
    data = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        "sample-10BT",
        split="train",
        streaming=True,
    )
    text = next(row["text"] for row in data if len(row["text"]) > 4 * tokens)
    batch = tokenizer(
        text,
        max_length=tokens,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    batch["labels"] = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100)
    return {name: value.cuda() for name, value in batch.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.requires_grad_(False)
    model.config.use_cache = False

    index, moe = next(
        (i, layer.mlp)
        for i, layer in reversed(list(enumerate(model.model.layers)))
        if hasattr(layer.mlp, "experts")
    )
    original = moe.experts
    experts = Qwen3Nvfp4Experts.from_hf(
        original,
        tokens=args.tokens,
        topk=model.config.num_experts_per_tok,
    )
    moe.experts = experts
    moe.gate.requires_grad_(True)
    del original
    torch.cuda.empty_cache()

    parameters = list(experts.parameters()) + list(moe.gate.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, fused=True)
    batch = training_batch(tokenizer, args.tokens)
    model.train()

    count = sum(parameter.numel() for parameter in parameters)
    print(f"training layer {index}: {count / 1e6:.1f}M parameters")
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch, use_cache=False).loss
        loss.backward()
        router_grad = moe.gate.weight.grad.float().norm().item()
        expert_grad = experts.layer.w1.grad.float().norm().item()
        optimizer.step()
        experts.refresh_weights()
        print(
            f"step {step + 1}: loss={loss.item():.4f} "
            f"router_grad={router_grad:.3e} expert_grad={expert_grad:.3e}"
        )


if __name__ == "__main__":
    main()
