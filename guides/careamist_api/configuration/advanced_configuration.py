#!/usr/bin/env python
# %%
# --8<-- [start:data]
from careamics.config import DataConfig

data_config = DataConfig(
    data_type="custom",  # (1)!
    axes="YX",
    patch_size=[128, 128],
    batch_size=8,
    num_epochs=20,
)
# --8<-- [end:data]

# %%
# --8<-- [start:model]
from careamics.config import FCNAlgorithmConfig, register_model
from torch import nn, ones


@register_model(name="linear_model")  # (1)!
class LinearModel(nn.Module):
    def __init__(self, in_features, out_features, *args, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(ones(in_features, out_features))
        self.bias = nn.Parameter(ones(out_features))

    def forward(self, input):
        return (input @ self.weight) + self.bias


config = FCNAlgorithmConfig(
    algorithm_type="fcn",
    algorithm="custom",  # (2)!
    loss="mse",
    model={
        "architecture": "custom",  # (3)!
        "name": "linear_model",  # (4)!
        "in_features": 10,
        "out_features": 5,
    },
)
# --8<-- [end:model]
