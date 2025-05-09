# ML-Project
# BioBERT + LoRA + CRF – NER Comparison Suite

This repo benchmarks parameter-efficient fine-tuning techniques for biomedical
named-entity recognition (NER).  
It compares

| Approach | Notes |
|----------|-------|
| **Full fine-tune** | All BioBERT parameters updated. |
| **LoRA** | Low-rank adapters (r = 8 – 64) inserted in every attention layer. |
| **LoRA + CRF** | Adds a CRF decoding head on top of LoRA-adapted encoder. |
| **LoRA + Aug** | Simple entity-swap data augmentation during training. |
| **Adapter fusion** | Fusion of two domain-specific LoRA adapters. |

Results are logged in an ASCII table after every run (see `train_compare.py`).

---

## 1.  Quick start

```bash
# 1️⃣  create & activate env (Python 3.10+)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2️⃣  install deps
pip install -r requirements.txt

# 3️⃣  run the baseline comparison (≈10 min on 1 × T4 / A10)
python src/train_compare.py --config experiments.yml

After training you’ll see something like

+-------------+-------+-----+-------+--------+
| exp         |  f1   | ... | peak  |
+-------------+-------+-----+-------+--------+
| full        | 0.88  | ... | 3.2 G |
| lora-r32    | 0.85  | ... | 1.9 G |
| lora-r16-crf| 0.86  | ... | 1.9 G |
+-------------+-------+-----+-------+--------+

2. Project layout

.
├── experiments.yml         # YAML sweep definitions
├── src/
│   ├── train_compare.py    # orchestrates a multi-run comparison
│   ├── models.py           # BioBERT / LoRA / CRF model factory
│   └── utils.py            # Trainer subclass, metrics, augmentation
└── requirements.txt

3. Editing / adding experiments

Open experiments.yml and append a new block, e.g.

- name: lora-r32-cosine
  method: lora
  rank: 32
  lr: 5e-5
  epochs: 5
  lr_scheduler_type: cosine
  warmup_steps: 300

Any key placed at the top level acts as a global default
(e.g. dataset, epochs, gpu).
4. Datasets

All datasets are retrieved via 🤗 Datasets on first use and cached to
~/.cache/huggingface/.
The default configuration trains on bc5cdr
(chemicals + diseases, 1 500 Medline abstracts).

To switch datasets globally:

dataset: bc5cdr        # ← change to e.g. 'ncbi_disease'

5. GPU / CPU

    Set gpu: -1 in experiments.yml to force CPU.

    Mixed-precision (fp16: true) is enabled by default; turn it off if you
    hit NaNs on older cards.

6. Reproducibility

    seed in the YAML is the base seed; train_compare.py automatically
    adds +1, +2 for the three repeats of each experiment.

    Logs, metrics and adapter checkpoints are written to outputs/{exp_name}/.

7. Citing

If you use this code, please cite:

@article{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward and et al.},
  year={2021},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}

and the respective dataset papers (BC5CDR, NCBI Disease, etc.).
8. License

MIT License – see LICENSE