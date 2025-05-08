# train_compare.py  ─────────────────────────────────────────────────────
import time, copy, torch, evaluate
from datasets import load_dataset
from transformers import (TrainingArguments,
                          DataCollatorForTokenClassification,
                          AutoTokenizer)
from transformers import Trainer
from torchcrf import CRF
import numpy as np

from models import BioBertCRF, BioBertLoRaCRF   # <<<—  models.py from earlier
# ----------------------------------------------------------------------

class CRFTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **_):
        labels = inputs.pop("labels")
        out = model(input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=labels)
        return (out["loss"], out) if return_outputs else out["loss"]


# ── Data ───────────────────────────────────────────────────────────────
MODEL  = "dmis-lab/biobert-base-cased-v1.1"
LABELS = ["O", "B-Chemical", "I-Chemical", "B-Disease", "I-Disease"]
NLAB   = len(LABELS)

tok      = AutoTokenizer.from_pretrained(MODEL)
dataset  = load_dataset("tner/bc5cdr")

def tok_align(batch):
    tok_out = tok(batch["tokens"], is_split_into_words=True,
                  truncation=True, max_length=512)
    all_lab = []
    for i, wl in enumerate(batch["tags"]):
        wids, prev, ids = tok_out.word_ids(batch_index=i), None, []
        for w in wids:
            if w is None: ids.append(-100)
            elif w != prev: ids.append(wl[w] if 0 <= wl[w] < NLAB else -100)
            else: ids.append(-100)
            prev = w
        all_lab.append(ids)
    tok_out["labels"] = all_lab
    return tok_out

data = dataset.map(tok_align, batched=True)
coll = DataCollatorForTokenClassification(tokenizer=tok)
metric = evaluate.load("seqeval")

def metrics(eval_pred):
    p,l = eval_pred
    p = p.argmax(-1) if p.ndim==3 else np.array([[ *seq, *([-100]*(l.shape[1]-len(seq))) ] for seq in p])
    tp = [[LABELS[pj] for pj, lj in zip(pi, li) if lj!=-100] for pi,li in zip(p,l)]
    tl = [[LABELS[lj] for pj, lj in zip(pi, li) if lj!=-100] for pi,li in zip(p,l)]
    r = metric.compute(predictions=tp, references=tl)
    return dict(precision=r["overall_precision"], recall=r["overall_recall"],
                f1=r["overall_f1"], accuracy=r["overall_accuracy"])

# ── shared TrainingArguments ───────────────────────────────────────────
base_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,                     # overwritten for LoRA
    per_device_train_batch_size=16,
    per_device_eval_batch_size =16,
    num_train_epochs=10,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,
    eval_strategy="epoch",                  # <- old API spelling
    save_strategy="no",
    logging_steps=100,
    report_to="none",
)

# ── Experiments dict ───────────────────────────────────────────────────
experiments = {
    "lora_crf": BioBertLoRaCRF(MODEL, NLAB),
    "full_crf": BioBertCRF(MODEL, NLAB),
}

results = {}

for tag, model in experiments.items():
    print(f"\n🟢  Experiment: {tag}")
    # ---- clone TrainingArguments safely for old HF versions ----
    args = copy.deepcopy(base_args)
    args.learning_rate = 1e-5 if tag=="lora_crf" else 2e-5

    # ---- reset CUDA stats & start timing ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    trainer = CRFTrainer(
        model=model, args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        data_collator=coll,
        compute_metrics=metrics,
    )
    trainer.train()

    total_time = time.time()-t0
    peak_mem   = torch.cuda.max_memory_allocated(device)/1e6  # MB
    epoch_time = total_time / args.num_train_epochs

    val = trainer.evaluate()
    test= trainer.evaluate(data["test"])

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    results[tag] = dict(val=val, test=test,
                        mem=peak_mem, tt=total_time, et=epoch_time,
                        params=trainable)

# ── Comparison printout ────────────────────────────────────────────────
print("\n================== COMPARISON ==================")
print("Model      | Val F1 | Test F1 | Peak MB | s/epoch | Total s | Trainable")
for tag, d in results.items():
    print(f"{tag:<10} | {d['val']['eval_f1']:.3f}  | {d['test']['eval_f1']:.3f} "
          f"| {d['mem']:.0f}    | {d['et']:.1f}  | {d['tt']:.1f}  | {d['params']:,}")
