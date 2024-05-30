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

train_array = np.random.rand(256, 256)
val_array = np.random.rand(256, 256)

careamist.train(
    train_source=train_array,  # (1)!
    val_source=val_array,  # (2)!
)

# %%
import tifffile
from careamics.utils import get_careamics_home

path_to_train_data = get_careamics_home() / "test" / "train_data.tiff"
path_to_train_data.parent.mkdir(exist_ok=True, parents=True)
tifffile.imwrite(path_to_train_data, train_array)
assert path_to_train_data.exists()

path_to_val_data = get_careamics_home() / "test" / "val_data.tiff"
path_to_val_data.parent.mkdir(exist_ok=True, parents=True)
tifffile.imwrite(path_to_val_data, train_array)
assert path_to_val_data.exists()

config.data_config.data_type = "tiff"

careamist.train(
    train_source=path_to_train_data,  # (1)!
    val_source=path_to_val_data,
)

# %%
config.data_config.data_type = "array"

careamist.train(
    train_source=train_array,
    val_percentage=0.1,  # (1)!
    val_minimum_split=5,  # (2)!
)

# %%
from careamics import CAREamicsTrainData

data_module = CAREamicsTrainData(  # (1)!
    data_config=config.data_config, train_data=train_array
)

careamist.train(datamodule=data_module)
