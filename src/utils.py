# utils.py -------------------------------------------------------------------------
# Helper utilities for the YAML‑driven NER comparison framework.
# v2:  • cast YAML strings → float for LR/WD (bug‑fix)
#      • pass label_names into HuggingFace Trainer to silence warning
#      • set tokenizer max_length=512 to stop truncation warning
# ---------------------------------------------------------------------------
from __future__ import annotations

import os, random, time
from typing import Any, Dict, Tuple, List

import numpy as np, torch
from datasets import DatasetDict
import datasets
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

# ------------------------------------------------------------------
# 1. Deterministic everything
# ------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------------
# 2. Light entity‑swap augmentation (quick & dirty)
# ------------------------------------------------------------------

def run_augmentation(train_ds, label_names: List[str]):
    new_examples = []
    for ex in train_ds:
        tok, tags = ex["tokens"], ex["tags"]
        buckets = {}
        for t, lab in zip(tok, tags):
            if lab != 0:
                buckets.setdefault(lab, []).append(t)
        for k in buckets:
            random.shuffle(buckets[k])
        ctr = {k:0 for k in buckets}
        new_tok = []
        for t, lab in zip(tok, tags):
            if lab == 0:
                new_tok.append(t)
            else:
                pool = buckets[lab]
                new_tok.append(pool[ctr[lab] % len(pool)])
                ctr[lab] += 1
        new_examples.append({"tokens": new_tok, "tags": tags})
    
    # build a Dataset, then *cast* to the same schema as the original split
    aug_ds = datasets.Dataset.from_list(new_examples)
    aug_ds = aug_ds.cast(train_ds.features)      # align dtypes (int32)

    return datasets.concatenate_datasets([train_ds, aug_ds])



# ------------------------------------------------------------------
# 3. Trainer builder (vanilla vs CRF)
# ------------------------------------------------------------------

class CRFTrainer(Trainer):
    """Wraps masking logic before feeding the CRF model."""
    def compute_loss(self, model, inputs, return_outputs=False, **_):
        labels = inputs.pop("labels")

        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            # label_mask=label_mask,
            # labels_clamped=labels_clamped,
            labels=labels,
        )
        loss = out["loss"]
        return (loss, out) if return_outputs else loss
    
    def prediction_step(           # NEW / REPLACE the previous stub
        self, model, inputs, prediction_loss_only=False, ignore_keys=None
    ):
        labels = inputs.pop("labels")
        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        # --- convert list[list[int]] ➜ padded tensor ------------------
        decoded = out["logits"]                 # list of sequences
        max_len  = labels.size(1)
        preds = torch.full(
            (len(decoded), max_len),
            fill_value=-100,                    # match HF masking
            dtype=torch.long,
            device=labels.device,
        )
        for i, seq in enumerate(decoded):
            seq = torch.tensor(seq, dtype=torch.long, device=labels.device)
            preds[i, : min(len(seq), max_len)] = seq

        # Trainer expects (loss, logits, labels); loss is None here
        return (None, preds, labels)


def build_trainer(
    exp: Dict[str, Any],
    model: torch.nn.Module,
    train_ds,
    raw_ds: DatasetDict,
    label_names: List[str],
    device: torch.device,
    epochs: int,
):
    tokenizer = AutoTokenizer.from_pretrained(
        exp.get("base_model", "dmis-lab/biobert-base-cased-v1.1"),
        use_fast=True,
    )

    def tok_map(batch):
        enc = tokenizer(
            batch["tokens"],
            truncation=True,
            max_length=512,
            is_split_into_words=True,
        )
        lab_out = []
        for i, tags in enumerate(batch["tags"]):
            ids = enc.word_ids(batch_index=i)
            prev = None; cur = []
            for wid in ids:
                if wid is None:
                    cur.append(-100)
                elif wid != prev:
                    cur.append(tags[wid])
                else:
                    cur.append(-100)
                prev = wid
            lab_out.append(cur)
        enc["labels"] = lab_out
        return enc

    drop_cols = train_ds.column_names
    train_tok = train_ds.map(tok_map, batched=True, remove_columns=drop_cols)
    val_tok   = raw_ds["validation"].map(tok_map, batched=True, remove_columns=drop_cols)
    test_tok  = raw_ds["test"].map(tok_map, batched=True, remove_columns=drop_cols)

    collator = DataCollatorForTokenClassification(tokenizer)
    metric   = evaluate.load("seqeval")

    def metrics_fn(pred):
        logits, labels = pred

        # ── 1. Convert logits → tag-id matrix ──────────────────────────
        if isinstance(logits, np.ndarray) and logits.ndim == 3:
            preds = logits.argmax(-1)                     # [B,L,C] ➜ [B,L]
        else:
            # list-of-lists (CRF.decode).  Pad to labels.shape with –100
            preds = np.full(labels.shape, fill_value=-100)
            for i, seq in enumerate(logits):
                preds[i, :len(seq)] = seq

        # ── 2. Build seqeval inputs ────────────────────────────────────
        true_preds = [
            [label_names[p] for p, l in zip(pr, la) if l != -100]
            for pr, la in zip(preds, labels)
        ]
        true_labels = [
            [label_names[l] for p, l in zip(pr, la) if l != -100]
            for pr, la in zip(preds, labels)
        ]
        res = metric.compute(predictions=true_preds, references=true_labels)
        return {k.replace("overall_", ""): v for k, v in res.items()}

    args = TrainingArguments(
        output_dir=f"./runs/{exp['name']}",
        learning_rate=float(exp.get("lr", 1e-5)),
        weight_decay=float(exp.get("weight_decay", 0.01)),
        per_device_train_batch_size=exp.get("per_device_train_batch", 16),
        per_device_eval_batch_size=exp.get("per_device_eval_batch", 16),
        num_train_epochs=epochs,
        fp16=bool(exp.get("fp16", True)),
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        lr_scheduler_type="cosine",
        warmup_steps=int(exp.get("warmup_steps", 0)),
        # max_steps=5,
    )

    TrainerCls = CRFTrainer if exp.get("use_crf", False) else Trainer
    trainer = TrainerCls(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=metrics_fn,
        # label_names=label_names,  # new: silences warning
    )

    return trainer, val_tok, test_tok
