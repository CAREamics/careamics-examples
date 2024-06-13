#!/usr/bin/env python
# %%
# --8<-- [start:config]
from careamics import CAREamist
from careamics.config import create_n2v_configuration

config = create_n2v_configuration(
    experiment_name="n2v_2D",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
)  # (1)!

careamist = CAREamist(config)
# --8<-- [end:config]

# %%
# --8<-- [start:config_path]
from careamics import CAREamist
from careamics.config import create_n2v_configuration, save_configuration

config = create_n2v_configuration(
    experiment_name="n2v_2D",
    data_type="array",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
)

save_configuration(config, "configuration_example.yml")

careamist = CAREamist("configuration_example.yml")
# --8<-- [end:config_path]

# %%
# necessary to export to the BMZ (pretending it trained)
careamist.cfg.data_config.set_mean_and_std(0.0, 1.0)

# %%
careamist.export_to_bmz(
    path="model.zip", name="MyExampleModel", authors=[{"name": "CAREamics"}]
)

# %%
# --8<-- [start:load_model]
from careamics import CAREamist

path_to_model = "model.zip"  # (1)!

careamist = CAREamist(path_to_model)
# --8<-- [end:load_model]

# %%
# --8<-- [start:exp_name]
careamist = CAREamist(path_to_model, experiment_name="a_new_experiment")
# --8<-- [end:exp_name]

# %%
# --8<-- [start:work_dir]
careamist = CAREamist(config, work_dir="work_dir")
# --8<-- [end:work_dir]
