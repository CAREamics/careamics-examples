#!/usr/bin/env python
# %%
# --8<-- [start:specs]
from careamics import Configuration
from careamics.config import (
    DataConfig,
    FCNAlgorithmConfig,
    TrainingConfig,
)
from careamics.config.architectures import UNetModel
from careamics.config.callback_model import EarlyStoppingModel
from careamics.config.optimizer_models import LrSchedulerModel, OptimizerModel
from careamics.config.support import (
    SupportedActivation,
    SupportedAlgorithm,
    SupportedArchitecture,
    SupportedData,
    SupportedLogger,
    SupportedLoss,
    SupportedOptimizer,
    SupportedPixelManipulation,
    SupportedScheduler,
    SupportedStructAxis,
)
from careamics.config.transformations import (
    N2VManipulateModel,
    XYFlipModel,
    XYRandomRotate90Model,
)

experiment_name = "Full example"

# Algorithm
model = UNetModel(  # (1)!
    architecture=SupportedArchitecture.UNET.value,
    in_channels=1,
    num_classes=1,
    depth=2,
    num_channels_init=32,
    final_activation=SupportedActivation.NONE.value,
    n2v2=False,
)

optimizer = OptimizerModel(
    name=SupportedOptimizer.ADAM.value, parameters={"lr": 0.0001}
)

scheduler = LrSchedulerModel(
    name=SupportedScheduler.REDUCE_LR_ON_PLATEAU.value,
)

algorithm_model = FCNAlgorithmConfig(
    algorithm_type="fcn",
    algorithm=SupportedAlgorithm.N2V.value,
    loss=SupportedLoss.N2V.value,
    model=model,
    optimizer=optimizer,
    lr_scheduler=scheduler,
)

# Data
xyflip = XYFlipModel()
rotate = XYRandomRotate90Model()
n2vmanipulate = N2VManipulateModel(
    roi_size=11,
    masked_pixel_percentage=0.2,
    strategy=SupportedPixelManipulation.MEDIAN.value,
    struct_mask_axis=SupportedStructAxis.NONE.value,
    struct_mask_span=7,
)

data_model = DataConfig(
    data_type=SupportedData.ARRAY.value,
    patch_size=(256, 256),
    batch_size=8,
    axes="YX",
    transforms=[xyflip, rotate, n2vmanipulate],  # (2)!
    dataloader_params={
        "num_workers": 4,
        # (3)!
    },
)

# Traning
earlystopping = EarlyStoppingModel(
    # (4)!
)

training_model = TrainingConfig(
    num_epochs=30,
    logger=SupportedLogger.WANDB.value,
    early_stopping_callback=earlystopping,
)

config = Configuration(
    experiment_name=experiment_name,
    algorithm_config=algorithm_model,
    data_config=data_model,
    training_config=training_model,
)
# --8<-- [end:specs]
