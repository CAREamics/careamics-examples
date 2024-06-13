#!/usr/bin/env python
# %%
# --8<-- [start:as_dict]
from careamics import Configuration

config_as_dict = {
    "experiment_name": "my_experiment",  # (1)!
    "algorithm_config": {  # (2)!
        "algorithm": "n2v",
        "loss": "n2v",
        "model": {  # (3)!
            "architecture": "UNet",
        },
    },
    "data_config": {  # (4)!
        "data_type": "array",
        "patch_size": [128, 128],
        "axes": "YX",
    },
    "training_config": {
        "num_epochs": 1,
    },
}
config = Configuration(**config_as_dict)  # (5)!
# --8<-- [end:as_dict]

# %%
# --8<-- [start:pydantic]
from careamics import Configuration
from careamics.config import (  # (1)!
    AlgorithmConfig,
    DataConfig,
    TrainingConfig,
)
from careamics.config.architectures import UNetModel
from careamics.config.support import (
    SupportedAlgorithm,
    SupportedArchitecture,
    SupportedData,
    SupportedLogger,
    SupportedLoss,
    SupportedTransform,
)
from careamics.config.transformations import N2VManipulateModel

experiment_name = "Pydantic N2V2 example"

# build AlgorithmConfig
algorithm_model = AlgorithmConfig(  # (2)!
    algorithm=SupportedAlgorithm.N2V.value,  # (3)!
    loss=SupportedLoss.N2V.value,
    model=UNetModel(  # (4)!
        architecture=SupportedArchitecture.UNET.value,
        in_channels=1,
        num_classes=1,
    ),
)

# then the DataConfig
data_model = DataConfig(
    data_type=SupportedData.ARRAY.value,
    patch_size=(256, 256),
    batch_size=8,
    axes="YX",
    transforms=[
        {  # (5)!
            "name": SupportedTransform.NORMALIZE.value,
        },
        {
            "name": SupportedTransform.NDFLIP.value,
            "is_3D": False,
        },
        N2VManipulateModel(  # (6)!
            masked_pixel_percentage=0.15,
        ),
    ],
    dataloader_params={  # (7)!
        "num_workers": 4,
    },
)

# then the TrainingConfig
training_model = TrainingConfig(
    num_epochs=30,
    logger=SupportedLogger.WANDB.value,
)

# finally, build the Configuration
config = Configuration(  # (8)!
    experiment_name=experiment_name,
    algorithm_config=algorithm_model,
    data_config=data_model,
    training_config=training_model,
)
# --8<-- [end:pydantic]
