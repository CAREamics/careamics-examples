#!/usr/bin/env python
# %%
import numpy as np
from careamics import CAREamicsTrainData
from careamics.config import create_n2v_configuration

train_array = np.random.rand(128, 128)

config = create_n2v_configuration(
    experiment_name="n2v_2D",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=1,
    num_epochs=1,
)

data_module = CAREamicsTrainData(  # (1)!
    data_config=config.data_config, train_data=train_array
)

# %%
from pathlib import Path
from typing import Any

import numpy as np
from careamics import CAREamicsTrainData
from careamics.config import create_n2v_configuration


def read_npy(  # (1)!
    path: Path,  # (2)!
    *args: Any,
    **kwargs: Any,  # (3)!
) -> np.ndarray:
    return np.load(path)  # (4)!


# example data
train_array = np.random.rand(128, 128)
np.save("train_array.npy", train_array)

# configuration
config = create_n2v_configuration(
    experiment_name="n2v_2D",
    data_type="custom",  # (5)!
    axes="YX",
    patch_size=[32, 32],
    batch_size=1,
    num_epochs=1,
)

data_module = CAREamicsTrainData(
    data_config=config.data_config,
    train_data="train_array.npy",  # (6)!
    read_source_func=read_npy,  # (7)!
    extension_filter="*.npy",  # (8)!
)
data_module.prepare_data()
data_module.setup()  # (9)!

# check dataset output
dataloader = data_module.train_dataloader()
print(dataloader.dataset[0][0].shape)  # (10)!
