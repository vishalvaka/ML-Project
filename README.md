````markdown
# 🔬 BioBERT + LoRA + CRF — Biomedical NER Comparison Suite

A compact framework for benchmarking **parameter-efficient** fine-tuning
strategies on biomedical named-entity recognition (NER) datasets.

| Variant | Description | Trainable % | Typical F1* |
|---------|-------------|-------------|-------------|
| **Full fine-tune** | Update all BioBERT weights | 100 % | 0.88 |
| **LoRA (r = 8–64)** | Low-rank adapters in every attention layer | 0.3 – 2 % | 0.59 → 0.85 |
| **LoRA + CRF** | LoRA encoder + CRF decoding head | 0.6 % | **0.86** |
| **LoRA + Aug** | Entity-swap data augmentation | 0.8 % | 0.65 |
| **Adapter fusion** | Fuse two domain LoRA adapters | 0.6 % | 0.74 |

\*Numbers from BC5CDR after 3-5 epochs on a single GPU.

---

## 1 · Quick start

```bash
# 1️⃣  Create & activate Python 3.10+ env
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2️⃣  Install deps
pip install -r requirements.txt

# 3️⃣  Run the default sweep (≈10 min on a single T4/A10)
python src/train_compare.py --config experiments.yml
````

The script trains each experiment three times (seeds = 42/43/44) and prints an
ASCII summary table:

```
+---------------+-----+-----+-------+-------+
| exp           | f1  | ... | sec   | MB    |
+---------------+-----+-----+-------+-------+
| full          |0.886| ... | 25.0  | 3206 |
| lora-r32      |0.848| ... | 14.6  | 1963 |
| lora-r16-crf  |0.863| ... | 30.7  | 1961 |
+---------------+-----+-----+-------+-------+
```

---

## 2 · Project layout

```
.
├── experiments.yml        # YAML sweep definitions & global defaults
├── src/
│   ├── train_compare.py   # orchestrates multi-run comparisons
│   ├── models.py          # BioBERT / LoRA / CRF factory
│   └── utils.py           # custom Trainer, metrics, augmentation
└── requirements.txt
```

---

## 3 · Editing / adding experiments

Open `experiments.yml` and append e.g.:

```yaml
- name: lora-r32-cosine
  method: lora
  rank: 32
  lr: 5e-5
  epochs: 5
  lr_scheduler_type: cosine
  warmup_steps: 300
  lora_dropout: 0.05
```

Keys at the top level act as **global defaults** (dataset, epochs, gpu, etc.);
stanza-level keys override them.

---

## 4 · Datasets

Datasets are fetched lazily via 🤗 **Datasets** and cached to
`~/.cache/huggingface/`.
Default: **[BC5CDR](https://huggingface.co/datasets/bigbio/bc5cdr)**
To switch:

```yaml
dataset: ncbi_disease   # or linnaeus, bc4chemd, etc.
```

---

## 5 · GPU & mixed precision

* `gpu: -1` ⇒ CPU. Set to `0`/`1`… for specific CUDA device.
* `fp16: true` is on by default; set `false` if you see NaNs on old hardware.

Peak VRAM:

| model      | MB          |
| ---------- | ----------- |
| full       | 3200        |
| LoRA / CRF | 1900 – 2000 |

---

## 6 · Reproducibility

* `seed:` in YAML is the **base seed**; the runner adds +1, +2 for repeats.
* Outputs (logs, metrics JSON, LoRA adapters) are in `outputs/{exp_name}/`.

---

## 7 · Extending the framework

* **Layer-wise LoRA** – adapt last *k* layers only.
* **Mixed fine-tune** – LoRA for 3 epochs → unfreeze encoder for 1 very low-LR epoch.
* **Entity-mask augmentation** – replace entities with `[MASK]` 30 % of the time.

All require only small additions to `models.py` or `utils.py`.

---

## 8 · Citation

If you use this code, please cite the relevant works:

```bibtex
@article{hu2021lora,
  title     = {LoRA: Low-Rank Adaptation of Large Language Models},
  author    = {Edward J. Hu and Yelong Shen and et al.},
  year      = {2021},
  eprint    = {2106.09685},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

@article{li2016bc5cdr,
  title={BioCreative V CDR task corpus: a resource for chemical disease relation extraction},
  author={Jiao Li and Alan Tam and et al.},
  journal={Database},
  year={2016}
}
```

---

## 9 · License

[MIT](LICENSE). Feel free to fork and build upon the project.

````

---

### `requirements.txt` (for convenience)

```text
torch>=2.1.0            # CUDA 11.8 binaries preferred
transformers>=4.40.0
datasets>=2.19.0
accelerate>=0.29.0
peft>=0.11.0
torchcrf>=1.2.0

scikit-learn>=1.4.2
evaluate>=0.4.1
tqdm>=4.66.0
pyyaml>=6.0.1
numpy>=1.26.4
pandas>=2.2.2

matplotlib>=3.8.4       # optional (plots)
````

> **Tip:** pin exact versions once your environment is stable to ensure
> reproducible runs (`pip freeze > requirements_locked.txt`).

```
```