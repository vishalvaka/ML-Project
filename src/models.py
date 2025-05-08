# models.py
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
import torch
import torch.nn as nn
from torchcrf import CRF

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
