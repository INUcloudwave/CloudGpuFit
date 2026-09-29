from __future__ import annotations

import torch


class LastTokenOutputProjection(torch.nn.Module):
    def __init__(self, projection: torch.nn.Module):
        super().__init__()
        self.projection = projection

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_states[:, -1:, :])


def restrict_causal_lm_to_last_token(model: torch.nn.Module) -> torch.nn.Module:
    projection = model.get_output_embeddings()
    if projection is None:
        raise ValueError("CausalLM model has no output embedding / LM head")
    if isinstance(projection, LastTokenOutputProjection):
        return model
    model.set_output_embeddings(LastTokenOutputProjection(projection))
    return model


class LastTokenLogits(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, **inputs):
        output = self.model(**inputs, use_cache=False, return_dict=True)
        return output.logits[:, -1, :]

