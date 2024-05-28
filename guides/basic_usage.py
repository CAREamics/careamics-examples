#!/usr/bin/env python
"""Example showcasing the basic usage of CAREamics."""
import numpy as np
from careamics import CAREamist
from careamics.config import create_n2v_configuration

# create a configuration
config = create_n2v_configuration(
    experiment_name="n2v_2D",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=1,
    num_epochs=1,
)

# instantiate a careamist
careamist = CAREamist(config)

# train the model
train_data = np.random.randint(0, 255, (256, 256))
careamist.train(train_source=train_data)

# once trained, predict
pred_data = np.random.randint(0, 255, (128, 128))
predction = careamist.predict(source=pred_data)

# export to BMZ format
careamist.export_to_bmz(
    path="my_model.bmz", name="N2V 2D", authors=[{"name": "CAREamics authors"}]
)
