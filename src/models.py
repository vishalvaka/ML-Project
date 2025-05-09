# models.py -------------------------------------------------------------------
"""Model factory and custom model definitions for the NER comparison suite.
This revision adds an **automatic fallback** inside `BioBertLoRaCRF.forward` so
that it can be called *either* with the pre‑computed `label_mask` &
`labels_clamped` (training loop) **or** with a raw `labels` tensor (evaluation
loop).  This eliminates the ‘mask of the first timestep must all be on’ error
thrown during the evaluation phase.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoModelForTokenClassification
from torchcrf import CRF

# ------------------------------------------------------------------
# 1. BioBERT + optional LoRA + optional CRF
# ------------------------------------------------------------------

# class BioBertLoRaCRF(nn.Module):
#     def __init__(self, model_name: str, num_labels: int, *, rank: int = 16, alpha: int | None = None):
#         super().__init__()
#         base = AutoModel.from_pretrained(model_name)
#         lora_cfg = LoraConfig(
#             task_type="TOKEN_CLS",
#             r=rank,
#             lora_alpha=alpha or rank * 2,
#             lora_dropout=0.1,
#             bias="none",
#         )
#         self.encoder = get_peft_model(base, lora_cfg)
#         self.dropout = nn.Dropout(0.1)
#         self.classifier = nn.Linear(base.config.hidden_size, num_labels)
#         self.crf = nn.Module()  # placeholder; real CRF lazily imported
#         try:
#             from torchcrf import CRF
#             self.crf = CRF(num_labels, batch_first=True)
#         except ImportError as e:
#             raise RuntimeError("torchcrf must be installed for CRF models") from e

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         attention_mask: torch.Tensor,
#         labels: torch.Tensor | None = None,
#         *,
#         label_mask: torch.Tensor | None = None,
#         labels_clamped: torch.Tensor | None = None,
#         **extra
#     ):
        
#                 # strip any leftover keys that BertModel cannot swallow
#         extra.pop("labels", None)
#         extra.pop("label", None)
        
#         out = self.encoder(input_ids=input_ids,
#                             attention_mask=attention_mask,
#                             return_dict=True,
#                             **extra)(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
#         hidden = self.dropout(out.last_hidden_state)
#         emissions = self.classifier(hidden)

#         # ── CRF branch ──────────────────────────────────────────────
#         if labels is not None:
#             # fall back to on‑the‑fly mask construction when not supplied
#             if label_mask is None or labels_clamped is None:
#                 label_mask = (labels != -100) & attention_mask.bool()
#                 labels_clamped = labels.clone()
#                 labels_clamped[labels_clamped == -100] = 0
#                 label_mask[:, 0] = True
#                 labels_clamped[:, 0] = 0
#             loss = -self.crf(emissions, labels_clamped, mask=label_mask)
#             return {"loss": loss, "logits": emissions}
#         else:
#             pred = self.crf.decode(emissions, mask=attention_mask.bool())
#             return {"logits": pred}

class BioBertCRF(nn.Module):
    """
    Full-fine-tune BioBERT + CRF  (all encoder weights trainable)
    """
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        ).last_hidden_state
        emissions = self.classifier(self.dropout(hidden))

        if labels is not None:                    # training
            mask = (labels != -100) & attention_mask.bool()
            labels_ = labels.clone()
            labels_[labels_ == -100] = 0
            mask[:, 0] = True
            labels_[:, 0] = 0
            loss = -self.crf(emissions, labels_, mask=mask)
            return {"loss": loss, "logits": emissions}
        else:                                     # predict
            pred = self.crf.decode(emissions, mask=attention_mask.bool())
            return {"logits": pred}


class BioBertLoRaCRF(BioBertCRF):
    """
    Same head as above, but encoder is *wrapped* with LoRA adapters
    so only ~0.5 % of parameters are trainable.
    """
    def __init__(self, model_name: str, num_labels: int,
                 r: int = 32, alpha: int = 32, dropout: float = 0.05):
        super().__init__(model_name, num_labels)
        peft_cfg = LoraConfig(
            task_type="FEATURE_EXTRACTION",
            r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none"
        )
        # replace the plain encoder with the LoRA-wrapped one
        self.encoder = get_peft_model(self.encoder, peft_cfg)


# ------------------------------------------------------------------
# 2. Factory
# ------------------------------------------------------------------

def _count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def build_model(cfg: Dict[str, Any], label_names: list[str]):
    method = cfg.get("method", "full")
    rank = int(cfg.get("rank", 0))
    alpha = cfg.get("alpha", rank * 2)
    base_name = cfg.get("base_model", "dmis-lab/biobert-base-cased-v1.1")
    num_labels = len(label_names)

    if method == "full":
        model = AutoModelForTokenClassification.from_pretrained(
            base_name,
            num_labels=num_labels,
        )
    elif method == "lora":
        if rank <= 0:
            raise ValueError("LoRA experiments must set a positive `rank`.")
        if cfg.get("use_crf", False):
            model = BioBertLoRaCRF(base_name, num_labels, r=rank, alpha=alpha)
        else:
            base_cls = AutoModelForTokenClassification.from_pretrained(
                base_name,
                num_labels=num_labels,
            )
            lora_cfg = LoraConfig(
                task_type="TOKEN_CLS",
                r=rank,
                lora_alpha=alpha or rank * 2,
                lora_dropout=float(cfg.get("lora_dropout", 0.1)),
                bias="none",
            )
            model = get_peft_model(base_cls, lora_cfg)
    else:
        raise ValueError(f"Unknown method {method}")

    total, trainable = _count_params(model)
    meta = {"all": total, "trainable": trainable}
    return model, meta

__all__ = ["build_model"]
