"""
train_stockholm.py
==================

Tier-C training run — train a GPT-2-class causal Transformer from
scratch on the Stockholm spatial-token corpus.

Architecture (defaults):
  * 12 layers, 12 heads, 768-d embeddings (~86 M parameters)
  * Context length 2048, bf16 on A100
  * 40 epochs on the full corpus with a 90/10 train/val split
  * AdamW, lr 3e-4, cosine schedule, 200-step warmup
  * Evaluate val loss once per epoch so we can see train vs. val
    divergence (generalisation gap).

All overrides are exposed as CLI flags so the same script drives any
tier (A / B / C) or a fast smoke test.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

logger = logging.getLogger("train_stockholm")


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def load_vocab(path: str) -> Dict[str, int]:
    """Read the tokenizer's vocab JSON and return a deterministic
    `{str_token: int_id}` mapping. IDs are assigned by sorted token
    order so the mapping is reproducible across machines and runs.
    """
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    tokens = sorted(payload["tokens"])
    if len(tokens) != payload["size"]:
        raise ValueError(
            f"vocab JSON inconsistency: size={payload['size']} but "
            f"{len(tokens)} unique tokens after sort"
        )
    return {tok: idx for idx, tok in enumerate(tokens)}


# ---------------------------------------------------------------------------
# Dataset pipeline
# ---------------------------------------------------------------------------


def build_dataset(
    parquet_path: str,
    token_to_id: Dict[str, int],
    block_size: int,
    num_proc: int,
    val_fraction: float,
    seed: int,
):
    """Stream per-object token lists out of the Parquet file, map to int
    IDs, concatenate, and chunk into fixed-size causal-LM blocks, then
    split into train/val.
    """
    logger.info("loading parquet dataset: %s", parquet_path)
    raw = load_dataset("parquet", data_files=parquet_path, split="train")
    logger.info("raw rows (objects): %d", len(raw))

    cols_to_drop = [c for c in raw.column_names if c != "tokens"]

    def to_ids(batch):
        # Map each per-object token list to its integer IDs. Unknown
        # tokens would be a vocab-dump bug — blow up loudly.
        out = []
        for seq in batch["tokens"]:
            out.append([token_to_id[t] for t in seq])
        return {"ids": out}

    ids_ds = raw.map(
        to_ids,
        batched=True,
        batch_size=2_000,
        num_proc=num_proc,
        remove_columns=cols_to_drop + ["tokens"],
        desc="str -> int",
    )

    def group_into_blocks(batch):
        """Concatenate all ID sequences in the batch into one long stream
        and cut it into `block_size`-length chunks. The trailing tail
        shorter than one block is discarded (standard GPT-style prep).
        """
        flat: List[int] = [tok for seq in batch["ids"] for tok in seq]
        total = (len(flat) // block_size) * block_size
        blocks = [flat[i : i + block_size] for i in range(0, total, block_size)]
        return {"input_ids": blocks, "labels": [b.copy() for b in blocks]}

    chunked = ids_ds.map(
        group_into_blocks,
        batched=True,
        batch_size=4_000,
        num_proc=num_proc,
        remove_columns=["ids"],
        desc=f"chunk into {block_size}-blocks",
    )

    logger.info(
        "total blocks: %d (block_size=%d, tokens=%d)",
        len(chunked), block_size, len(chunked) * block_size,
    )

    if val_fraction > 0:
        split = chunked.train_test_split(test_size=val_fraction, seed=seed)
        train_ds, val_ds = split["train"], split["test"]
        logger.info(
            "split train=%d / val=%d blocks (val_fraction=%.2f)",
            len(train_ds), len(val_ds), val_fraction,
        )
    else:
        train_ds, val_ds = chunked, None

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(
    vocab_size: int,
    block_size: int,
    n_layer: int,
    n_head: int,
    n_embd: int,
) -> GPT2LMHeadModel:
    """GPT-2 from scratch at the requested capacity."""
    config = GPT2Config(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_positions=block_size,
        n_ctx=block_size,
        bos_token_id=0,
        eos_token_id=0,
        use_cache=False,  # redundant at train-time, saves some memory
    )
    model = GPT2LMHeadModel(config)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "model built: %dL/%dH/%dD -> %.2fM parameters",
        n_layer, n_head, n_embd, n_params / 1e6,
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_trainer(
    model: GPT2LMHeadModel,
    train_ds,
    val_ds,
    output_dir: str,
    per_device_batch_size: int,
    per_device_eval_batch_size: int,
    num_train_epochs: float,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    logging_steps: int,
    save_epochs: bool,
    bf16: bool,
    dataloader_num_workers: int,
    gradient_accumulation_steps: int,
    seed: int,
) -> Trainer:
    eval_strategy = "epoch" if val_ds is not None else "no"

    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        logging_first_step=True,
        eval_strategy=eval_strategy,
        save_strategy="epoch" if save_epochs else "no",
        save_total_limit=2,
        bf16=bf16,
        fp16=False,
        optim="adamw_torch",
        dataloader_num_workers=dataloader_num_workers,
        report_to=["none"],
        seed=seed,
        lr_scheduler_type="cosine",
        # Overfit test: keep dropout default (0) and no early stopping —
        # we want the train curve to collapse.
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=default_data_collator,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # Data
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output-dir",
                        default="./checkpoints/stockholm_overfit")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Fraction of blocks to hold out as val (0 disables eval)")
    # Tokenisation / chunking
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--num-proc", type=int, default=4)
    # Model
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    # Training
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=40.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--no-save-epochs", action="store_true")
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    set_seed(args.seed)

    if not os.path.isfile(args.parquet):
        logger.error("parquet not found: %s", args.parquet)
        return 2
    if not os.path.isfile(args.vocab):
        logger.error("vocab not found: %s", args.vocab)
        return 2

    token_to_id = load_vocab(args.vocab)
    vocab_size = len(token_to_id)
    logger.info("vocab size: %d", vocab_size)

    train_ds, val_ds = build_dataset(
        parquet_path=args.parquet,
        token_to_id=token_to_id,
        block_size=args.block_size,
        num_proc=args.num_proc,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    if len(train_ds) == 0:
        logger.error("no training blocks produced; aborting")
        return 3

    model = build_model(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )

    trainer = build_trainer(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        output_dir=args.output_dir,
        per_device_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_epochs=not args.no_save_epochs,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_workers,
        gradient_accumulation_steps=args.grad_accum_steps,
        seed=args.seed,
    )

    logger.info("starting training")
    train_result = trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))

    # Persist the token <-> id mapping alongside the model so any loader
    # can decode generated integer IDs back to human-readable tokens.
    mapping_path = os.path.join(args.output_dir, "final", "token_to_id.json")
    with open(mapping_path, "w", encoding="utf-8") as fh:
        json.dump(token_to_id, fh, indent=2)
    logger.info("token mapping saved to %s", mapping_path)

    metrics = train_result.metrics
    logger.info("final train metrics: %s", metrics)
    if val_ds is not None:
        eval_metrics = trainer.evaluate()
        logger.info("final eval metrics: %s", eval_metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
