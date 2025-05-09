# train_compare.py  – YAML‑driven experiment launcher (adds α, peak‑MB, s/epoch)
# ---------------------------------------------------------------------------
#   Run a suite of NER fine‑tuning experiments defined in a YAML file and
#   print / optionally save a comparison table.  This version now tracks:
#     • LoRA α (already added)
#     • Peak GPU memory (MB)
#     • Seconds per epoch
# ---------------------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml
from prettytable import PrettyTable
from datasets import load_dataset

import torch

from models import build_model  # local factory
from utils import (
    seed_everything,
    build_trainer,
    run_augmentation,
)

# ---------------------------------------------------------------------------
# YAML helpers ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_results(rows: List[Dict[str, Any]], out_csv: Path | None, out_json: Path | None):
    if out_csv:
        import csv
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)


# ---------------------------------------------------------------------------
# Main driver ----------------------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PEFT vs full fine‑tune models for NER")
    parser.add_argument("--config", type=Path, default=Path("experiments.yml"), help="YAML config file")
    parser.add_argument("--save_csv", type=Path, default=None)
    parser.add_argument("--save_json", type=Path, default=None)
    args = parser.parse_args()

    cfg = read_yaml(args.config)
    seed_everything(cfg.get("seed", 42))

    device = (
        torch.device("cuda", cfg.get("gpu", 0))
        if cfg.get("gpu", 0) >= 0 and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # ---------------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------------
    ds_name = cfg.get("dataset", "bc5cdr")
    raw_ds = load_dataset("tner/" + ds_name) if ds_name == "bc5cdr" else load_dataset(ds_name)

    label_names = (
        ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]
        if ds_name == "bc5cdr"
        else raw_ds["train"].features["tags"].feature.names
    )

    results: List[Dict[str, Any]] = []

    for exp in cfg["experiments"]:
        exp_name = exp["name"]
        print(f"\n=== Running experiment: {exp_name} ===")

        # Augmentation (optional)
        train_ds = run_augmentation(raw_ds["train"], label_names) if exp.get("augment") else raw_ds["train"]

        # Build model & trainer ----------------------------------------
        model, meta = build_model(exp, label_names)
        model.to(device)

        trainer, val_ds, test_ds = build_trainer(
            exp,
            model,
            train_ds,
            raw_ds,
            label_names,
            device,
            epochs=exp.get("epochs", cfg.get("epochs", 3)),
        )

        # Training ------------------------------------------------------
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.time()
        trainer.train()
        train_time = time.time() - start
        epochs_run = trainer.args.num_train_epochs

        peak_mb = (
            torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
        )
        sec_per_epoch = train_time / epochs_run if epochs_run else train_time

        # Evaluation ----------------------------------------------------
        val_metrics = trainer.evaluate(val_ds)
        test_metrics = trainer.evaluate(test_ds)
        
        if "eval_f1" not in val_metrics:
            preds_out = trainer.predict(val_ds)
            try:
                extra = trainer.compute_metrics(
                    (preds_out.predictions, preds_out.label_ids)
                )
                val_metrics.update({f"eval_{k}": v for k, v in extra.items()})
            except Exception:
                pass
        
        val_f1   = val_metrics.get("eval_f1",   val_metrics.get("f1",   0.0))
        val_prec = val_metrics.get("eval_precision", val_metrics.get("precision", 0.0))
        val_rec  = val_metrics.get("eval_recall",    val_metrics.get("recall",    0.0))

        test_f1   = test_metrics.get("eval_f1",   test_metrics.get("f1",   0.0))
        test_prec = test_metrics.get("eval_precision", test_metrics.get("precision", 0.0))
        test_rec  = test_metrics.get("eval_recall",    test_metrics.get("recall",    0.0))

        # Collect row ---------------------------------------------------
        row = {
            "exp": exp_name,
            "method": exp["method"],
            "rank": exp.get("rank", "-"),
            "alpha": exp.get("alpha", "-"),
            "use_crf": bool(exp.get("use_crf", False)),
            "f1": round(val_f1, 4),
            "precision": round(val_prec, 4),
            "recall": round(val_rec, 4),
            "test_f1": round(test_f1, 4),
            "params": meta["all"],
            "trainable": meta["trainable"],
            "train_sec": round(train_time, 1),
            "sec_epoch": round(sec_per_epoch, 2),
            "peak_MB": int(peak_mb),
        }
        results.append(row)

    # ------------------------------------------------------------------
    # Pretty table -----------------------------------------------------
    # ------------------------------------------------------------------
    tbl = PrettyTable()
    tbl.field_names = list(results[0].keys())
    for r in results:
        tbl.add_row(list(r.values()))

    print("\n=== Comparison ===")
    print(tbl)

    # optional save
    save_results(results, args.save_csv, args.save_json)


if __name__ == "__main__":
    main()
