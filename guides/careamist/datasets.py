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
    data_config=config.data_config, train_source=train_array
)
