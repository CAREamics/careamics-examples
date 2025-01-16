#!/usr/bin/env python

# %%
# --8<-- [start:pydantic]
from careamics.config import (  # (1)!
    N2VAlgorithm,
    N2VConfiguration,
    N2VDataConfig,
    TrainingConfig,
    configuration_factory,
)
from careamics.config.architectures import UNetModel
from careamics.config.callback_model import CheckpointModel, EarlyStoppingModel
from careamics.config.support import (
    SupportedData,
    SupportedLogger,
)
from careamics.config.transformations import N2VManipulateModel, XYFlipModel

experiment_name = "N2V_example"

# build the model and algorithm configurations
model = UNetModel(
    architecture="UNet",  # (2)!
    num_channels_init=64,  # (3)!
    depth=3,
    # (4)!
)

algorithm_model = N2VAlgorithm(  # (5)!
    model=model,
    # (6)!
)

# then the N2VDataConfig
data_model = N2VDataConfig(  # (7)!
    data_type=SupportedData.ARRAY.value,
    patch_size=(256, 256),
    batch_size=8,
    axes="YX",
    transforms=[XYFlipModel(flip_y=False), N2VManipulateModel()],  # (8)!  # (9)!
    dataloader_params={  # (10)!
        "num_workers": 4,
    },
)

# then the TrainingConfig
earlystopping = EarlyStoppingModel(
    # (11)!
)

checkpoints = CheckpointModel(every_n_epochs=10)  # (12)!

training_model = TrainingConfig(
    num_epochs=30,
    logger=SupportedLogger.WANDB.value,
    early_stopping_callback=earlystopping,
    checkpoint_callback=checkpoints,
    # (13)!
)

# finally, build the Configuration
config = N2VConfiguration(  # (14)!
    experiment_name=experiment_name,
    algorithm_config=algorithm_model,
    data_config=data_model,
    training_config=training_model,
)

# alternatively, use the factory method
config2 = configuration_factory(  # (15)!
    {
        "experiment_name": experiment_name,
        "algorithm_config": algorithm_model,
        "data_config": data_model,
        "training_config": training_model,
    }
)
# --8<-- [end:pydantic]

if config != config2:
    raise ValueError("Configurations are not equal (Pydantic).")

# %%
# --8<-- [start:as_dict]
from careamics.config import N2VConfiguration, configuration_factory

config_dict = {
    "experiment_name": "N2V_example",
    "algorithm_config": {
        "algorithm": "n2v",  # (1)!
        "loss": "n2v",
        "model": {
            "architecture": "UNet",  # (2)!
            "num_channels_init": 64,
            "depth": 3,
        },
    },
    "data_config": {
        "data_type": "array",
        "patch_size": [256, 256],
        "batch_size": 8,
        "axes": "YX",
        "transforms": [
            {
                "name": "XYFlip",
                "flip_y": False,
            },
            {
                "name": "N2VManipulate",
            },
        ],
        "dataloader_params": {
            "num_workers": 4,
        },
    },
    "training_config": {
        "num_epochs": 30,
        "logger": "wandb",
        "early_stopping_callback": {},  # (3)!
        "checkpoint_callback": {
            "every_n_epochs": 10,
        },
    },
}

# instantiate specific configuration
config_as_dict = N2VConfiguration(**config_dict)  # (4)!

# alternatively, use the factory method
config_as_dict2 = configuration_factory(config_dict)  # (5)!
# --8<-- [end:as_dict]

if config_as_dict != config_as_dict2:
    raise ValueError("Configurations are not equal (Dict).")

if config != config_as_dict:
    raise ValueError("Configurations are not equal (Pydantic vs Dict).")
