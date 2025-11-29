import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from include.losses import FocalLoss

# --- Dual encoder model ---
class DualEncoderForSequenceClassification(PreTrainedModel):
    config_class = AutoConfig

    def __init__(
        self,
        config,
        use_focal_loss: bool = False,
        gamma: float = 2.0,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        # instantiate two encoders from the pretrained config
        self.encoder_text = AutoModel.from_config(config)
        self.encoder_bio = AutoModel.from_config(config)
        hidden_size = config.hidden_size

        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size, dtype=torch.bfloat16),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size,dtype=torch.bfloat16),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        self.classifier = nn.Linear(hidden_size, config.num_labels, dtype=torch.bfloat16)
        self.use_focal_loss = use_focal_loss
        self.gamma = gamma
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, return_dict=True):
        # both encoders receive the same input (text and bio were concatenated/tokenized as pair)
        out_text = self.encoder_text(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        out_bio = self.encoder_bio(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)

        h_text = out_text.last_hidden_state[:, 0]
        h_bio = out_bio.last_hidden_state[:, 0]

        combined = torch.cat([h_text, h_bio], dim=-1)
        gate = self.gate_layer(combined)  # shape (batch, 1)
        h_final = gate * h_text + (1 - gate) * h_bio
        pooled = self.dropout(h_final)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            cw = None
            if hasattr(self.config, "class_weights") and self.config.class_weights is not None:
                cw = torch.tensor(self.config.class_weights, device=logits.device, dtype=torch.bfloat16)
            if self.use_focal_loss:
                loss_fct = FocalLoss(gamma=self.gamma, weight=cw)
            else:
                loss_fct = nn.CrossEntropyLoss(weight=cw)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)