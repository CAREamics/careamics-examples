#!/usr/bin/env python
# %%
from careamics import CAREamist
from careamics.config import create_n2v_configuration

config = create_n2v_configuration(
    experiment_name="n2v_2D",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=1,
    num_epochs=1,
)

careamist = CAREamist(config)

# %%
import numpy as np

train_array = np.random.rand(128, 128)
val_array = np.random.rand(128, 128)

careamist.train(
    train_source=train_array,  # (1)!
    val_source=val_array,  # (2)!
)

# %%
careamist.train(
    train_source="path/to/my/data.tiff",  # (1)!
    val_source="path/to/my/val_data.tiff",
)

# %%
careamist.train(
    train_source=train_array,
    val_percentage=0.1,  # (1)!
    val_minimum_split=5,  # (2)!
)

# %%
from careamics import CAREamicsTrainData

data_module = CAREamicsTrainData(  # (1)!
    data_config=config.data_config, train_source=train_array
)

careamist.train(datamodule=data_module)
