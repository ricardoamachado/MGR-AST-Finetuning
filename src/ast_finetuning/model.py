"""Frozen AST backbone with a trainable softmax classification head."""

import torch
from torch import nn
from transformers import ASTModel, ASTPreTrainedModel


class ASTFineTuner(ASTPreTrainedModel):
    """Frozen AST backbone with a trainable Softmax classification head."""

    def __init__(self, config, num_labels: int, dropout: float = 0.1):
        super().__init__(config)
        self.num_labels = num_labels
        self.ast = ASTModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.softmax = nn.Softmax(dim=-1)
        self._classification_loss = nn.CrossEntropyLoss()
        self.post_init()
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """Freeze AST weights and keep the backbone deterministic during training."""
        self.ast.requires_grad_(False)
        self.ast.eval()

    def train(self, mode: bool = True):
        """Train the head while keeping the frozen backbone in evaluation mode."""
        super().train(mode)
        self.ast.eval()
        return self

    @classmethod
    def from_backbone(cls, model_name: str, num_labels: int, dropout: float = 0.1) -> "ASTFineTuner":
        """Create the classifier while loading weights from a Hugging Face AST checkpoint."""
        backbone = ASTModel.from_pretrained(model_name)
        model = cls(backbone.config, num_labels=num_labels, dropout=dropout)
        model.ast.load_state_dict(backbone.state_dict())
        return model

    def forward(self, input_values: torch.Tensor, labels: torch.Tensor | None = None):
        outputs = self.ast(input_values=input_values)
        pooled = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))
        loss = self._classification_loss(logits, labels) if labels is not None else None
        probabilities = self.softmax(logits)
        return {"loss": loss, "logits": logits, "probabilities": probabilities}
