#!/usr/bin/env python
# %%
from careamics.utils import get_careamics_home

mypath = get_careamics_home()

# --8<-- [start:basic_usage]

import numpy as np
from careamics.lightning import (
    CAREamicsModuleWrapper,
    PredictDataWrapper,
    TrainingDataWrapper,
)
from careamics.prediction_utils import convert_outputs
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
)

# training data
rng = np.random.default_rng(42)
train_array = rng.integers(0, 255, (32, 32)).astype(np.float32)
val_array = rng.integers(0, 255, (32, 32)).astype(np.float32)

# create lightning module
model = CAREamicsModuleWrapper(
    algorithm="n2v",
    loss="n2v",
    architecture="UNet",
)

# create data module
data = TrainingDataWrapper(
    train_data=train_array,
    val_data=val_array,
    data_type="array",
    patch_size=(16, 16),
    axes="YX",
    batch_size=2,
)

# create trainer
trainer = Trainer(
    max_epochs=1,
    default_root_dir=mypath,
    callbacks=[
        ModelCheckpoint(
            dirpath=mypath / "checkpoints",
            filename="basic_usage_lightning_api",
        )
    ],
)

# train
trainer.fit(model, datamodule=data)

# predict
means, stds = data.get_data_statistics()
predict_data = PredictDataWrapper(
    pred_data=val_array,
    data_type="array",
    image_means=means,
    image_stds=stds,
    tile_size=(8, 8),
    tile_overlap=(2, 2),
)

# predict
predicted = trainer.predict(model, datamodule=predict_data)
predicted_stitched = convert_outputs(predicted, tiled=True)
# --8<-- [end:basic_usage]