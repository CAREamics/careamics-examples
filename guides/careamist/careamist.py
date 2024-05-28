# %%
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


# %%
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

# %%
# necessary to export to the BMZ (pretending it trained)
careamist.cfg.data_config.set_mean_and_std(0.0, 1.0)

# %%
careamist.export_to_bmz(
    path="model.zip", name="MyExampleModel", authors=[{"name": "CAREamics"}]
)

# %%
from careamics import CAREamist

path_to_model = "model.zip"  # (1)!

careamist = CAREamist(path_to_model)

# %%
careamist = CAREamist(path_to_model, experiment_name="a_new_experiment")


# %%
careamist = CAREamist(config, work_dir="work_dir")
