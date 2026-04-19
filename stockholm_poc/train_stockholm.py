"""
train_stockholm.py
==================

The Overfit Test — train a tiny causal Transformer from scratch on the
Stockholm spatial-token corpus. We deliberately pick a small architecture
and run enough epochs for the training loss to collapse toward zero,
proving the model can mathematically learn the grammar (kind START/END
brackets, anchor-then-moves ordering) and the spatial geometry (which
MOVE tokens plausibly follow which X/Y anchor) of our tokenizer.

Stages
------
1. Load the vocabulary JSON, build `{str_token: int_id}` mapping.
2. Stream the Parquet file via `datasets`, map each per-object token list
   to integer IDs, then concatenate and chunk into fixed-size 1024-token
   blocks for causal LM training.
3. Build a tiny `GPT2LMHeadModel` from scratch (no pretrained weights)
   with vocab_size tied to the real vocabulary.
4. Train with `Trainer` on AdamW, lr=3e-4, standard cross-entropy CLM
   loss. Log every 50 steps; save best/final weights under
   `./checkpoints/stockholm_overfit`.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np

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
    `{str_token: int_id}` mapping. IDs are assigned by the sorted order
    of tokens so the mapping is reproducible across machines / runs.
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
):
    """Stream per-object token lists out of the Parquet file, map to int
    IDs, concatenate, and chunk into fixed-size causal-LM blocks.

    Returns a HuggingFace `Dataset` with `input_ids` and `labels` columns
    (labels == input_ids for next-token prediction).
    """
    logger.info("loading parquet dataset: %s", parquet_path)
    raw = load_dataset("parquet", data_files=parquet_path, split="train")
    logger.info("raw rows (objects): %d", len(raw))

    cols_to_drop = [c for c in raw.column_names if c != "tokens"]

    def to_ids(batch):
        # Map each per-object token list to its integer IDs. Unknown
        # tokens would be a bug in the vocab dump — blow up loudly.
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
        "training blocks: %d (block_size=%d, total_tokens=%d)",
        len(chunked), block_size, len(chunked) * block_size,
    )
    return chunked


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def build_model(vocab_size: int, block_size: int) -> GPT2LMHeadModel:
    """Tiny GPT-2 from scratch. The 6L/6H/384 dim config is small enough
    to overfit ~15M tokens quickly on a single A100, which is exactly
    what the Overfit Test wants.
    """
    config = GPT2Config(
        vocab_size=vocab_size,
        n_layer=6,
        n_head=6,
        n_embd=384,
        n_positions=block_size,
        n_ctx=block_size,
        bos_token_id=0,
        eos_token_id=0,
        use_cache=False,  # redundant during training, saves a bit of memory
    )
    model = GPT2LMHeadModel(config)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("model built from scratch: %.2fM parameters", n_params / 1e6)
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_trainer(
    model: GPT2LMHeadModel,
    train_ds,
    output_dir: str,
    per_device_batch_size: int,
    num_train_epochs: float,
    learning_rate: float,
    weight_decay: float,
    warmup_steps: int,
    logging_steps: int,
    save_epochs: bool,
    bf16: bool,
    dataloader_num_workers: int,
    seed: int,
) -> Trainer:
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        logging_first_step=True,
        save_strategy="epoch" if save_epochs else "no",
        save_total_limit=2,
        bf16=bf16,
        fp16=False,
        optim="adamw_torch",
        dataloader_num_workers=dataloader_num_workers,
        report_to=["none"],
        seed=seed,
        lr_scheduler_type="cosine",
        # Overfit test: do NOT add dropout, early stopping, eval split.
        # We *want* train loss -> 0.
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=default_data_collator,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True,
                        help="Path to stockholm_tokens.parquet")
    parser.add_argument("--vocab", required=True,
                        help="Path to stockholm_vocab.json")
    parser.add_argument("--output-dir",
                        default="./checkpoints/stockholm_overfit",
                        help="Checkpoint directory")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--no-save-epochs", action="store_true",
                        help="Disable per-epoch checkpointing")
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--num-proc", type=int, default=4,
                        help="Workers for dataset.map() preprocessing")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bf16 (A100 native); disable with --no-bf16")
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

    # Vocab.
    token_to_id = load_vocab(args.vocab)
    vocab_size = len(token_to_id)
    logger.info("vocab size: %d", vocab_size)

    # Dataset.
    train_ds = build_dataset(
        parquet_path=args.parquet,
        token_to_id=token_to_id,
        block_size=args.block_size,
        num_proc=args.num_proc,
    )
    if len(train_ds) == 0:
        logger.error("no training blocks produced; aborting")
        return 3

    # Model.
    model = build_model(vocab_size=vocab_size, block_size=args.block_size)

    # Trainer.
    trainer = build_trainer(
        model=model,
        train_ds=train_ds,
        output_dir=args.output_dir,
        per_device_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_epochs=not args.no_save_epochs,
        bf16=args.bf16,
        dataloader_num_workers=args.dataloader_workers,
        seed=args.seed,
    )

    logger.info("starting training")
    train_result = trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final"))

    # Persist the token<->id mapping alongside the model so any loader
    # can decode generated integer IDs back to human-readable tokens.
    mapping_path = os.path.join(args.output_dir, "final", "token_to_id.json")
    with open(mapping_path, "w", encoding="utf-8") as fh:
        json.dump(token_to_id, fh, indent=2)
    logger.info("token mapping saved to %s", mapping_path)

    metrics = train_result.metrics
    logger.info("final metrics: %s", metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
