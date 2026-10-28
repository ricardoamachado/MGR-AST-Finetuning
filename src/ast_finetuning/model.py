"""AST classifier fine-tuned either through its softmax head or through LoRA adapters."""

import torch
from peft import LoraConfig, inject_adapter_in_model
from torch import nn
from transformers import ASTModel, ASTPreTrainedModel

FINETUNING_METHODS = ("head", "lora")
# Linear layer names of transformers' AST: attention.{q,k,v,o}_proj and mlp.{fc1,fc2}.
DEFAULT_LORA_TARGET_MODULES = ("q_proj", "v_proj")


class ASTFineTuner(ASTPreTrainedModel):
    """AST backbone with a trainable Softmax classification head.

    The fine-tuning method is stored in the config so that ``from_pretrained`` rebuilds the same
    module structure that was saved:

    - ``head``: the AST backbone is frozen and only the classifier is trained.
    - ``lora``: the original AST weights are frozen, low-rank adapters are injected into the
      attention projections and trained together with the classifier.
    """

    def __init__(self, config, num_labels: int | None = None, dropout: float = 0.1):
        super().__init__(config)
        # from_pretrained moves a num_labels kwarg into the config, so fall back to it.
        self.num_labels = num_labels if num_labels is not None else config.num_labels
        self.ast = ASTModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.softmax = nn.Softmax(dim=-1)
        self._classification_loss = nn.CrossEntropyLoss()
        self.post_init()
        self.freeze_backbone()
        if self.finetuning_method == "lora":
            self._inject_lora()

    @property
    def finetuning_method(self) -> str:
        return getattr(self.config, "finetuning_method", "head")

    def freeze_backbone(self) -> None:
        """Freeze AST weights and keep the backbone deterministic during training."""
        self.ast.requires_grad_(False)
        self.ast.eval()

    def enable_lora(
        self,
        rank: int = 1,
        alpha: int = 2,
        dropout: float = 0.2,
        target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES,
    ) -> None:
        """Store the LoRA hyperparameters in the config and inject the adapters into AST."""
        self.config.finetuning_method = "lora"
        self.config.lora_rank = rank
        self.config.lora_alpha = alpha
        self.config.lora_dropout = dropout
        self.config.lora_target_modules = list(target_modules)
        self._inject_lora()

    def _inject_lora(self) -> None:
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            bias="none",
        )
        # Only the injected lora_A/lora_B matrices are left trainable inside the backbone.
        inject_adapter_in_model(lora_config, self.ast)

    def train(self, mode: bool = True):
        """Train the head while keeping a fully frozen backbone in evaluation mode."""
        super().train(mode)
        if self.finetuning_method == "head":
            self.ast.eval()
        return self

    @classmethod
    def from_backbone(
        cls,
        model_name: str,
        num_labels: int,
        dropout: float = 0.2,
        finetuning_method: str = "head",
        lora_rank: int = 1,
        lora_alpha: int = 2,
        lora_dropout: float = 0.2,
        lora_target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES,
    ) -> "ASTFineTuner":
        """Create the classifier while loading weights from a Hugging Face AST checkpoint."""
        if finetuning_method not in FINETUNING_METHODS:
            raise ValueError(f"finetuning_method must be one of {FINETUNING_METHODS}, got {finetuning_method!r}")
        backbone = ASTModel.from_pretrained(model_name)
        backbone.config.finetuning_method = "head"
        # Replace the AudioSet label count so the saved config matches the new classifier.
        backbone.config.num_labels = num_labels
        model = cls(backbone.config, num_labels=num_labels, dropout=dropout)
        # Pretrained weights must be loaded before LoRA wraps the target linear layers.
        model.ast.load_state_dict(backbone.state_dict())
        if finetuning_method == "lora":
            model.enable_lora(lora_rank, lora_alpha, lora_dropout, lora_target_modules)
        return model

    def forward(self, input_values: torch.Tensor, labels: torch.Tensor | None = None):
        outputs = self.ast(input_values=input_values)
        pooled = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))
        loss = self._classification_loss(logits, labels) if labels is not None else None
        probabilities = self.softmax(logits)
        return {"loss": loss, "logits": logits, "probabilities": probabilities}
