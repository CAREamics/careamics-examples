#!/usr/bin/env python
# %%
# Intro
from careamics.config import (
    create_care_configuration,  # CARE
    create_n2n_configuration,  # Noise2Noise
    create_n2v_configuration,  # Noise2Void, N2V2, structN2V
)

# %%
# Example Noise2Void with channels
config = create_n2v_configuration(
    experiment_name="n2v_2D_channels",
    data_type="tiff",
    axes="YXC",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    n_channels=3,  # (2)!
)
# %%
# Non independent channels
config = create_n2v_configuration(
    experiment_name="n2v_2D_channels",
    data_type="tiff",
    axes="YXC",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    n_channels=3,
    independent_channels=False,  # (2)!
)
# %%
# Example Noise2Noise with channels
config = create_n2n_configuration(
    experiment_name="n2n_2D_channels",
    data_type="tiff",
    axes="YXC",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    n_channels=3,  # (2)!
)
# %%
# Example CARE with channels
config = create_care_configuration(
    experiment_name="care_2D_channels",
    data_type="tiff",
    axes="YXC",  # (1)!
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    n_channels_in=3,  # (2)!
    n_channels_out=2,  # (3)!
)
# %%
# Example with augmentations
config = create_care_configuration(
    experiment_name="care_2D",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    use_augmentations=False,  # (1)!
)
# %%
# Example with logger
config = create_n2n_configuration(
    experiment_name="n2n_2D",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    logger="wandb",  # (1)!
)

# %%
# Example with model kwargs
config = create_care_configuration(
    experiment_name="care_3D",
    data_type="tiff",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    num_epochs=20,
    model_kwargs={
        "depth": 3,  # (1)!
        "num_channels_init": 64,  # (2)!
        # (3)!
    },
)

# %%
# Example with loss
config = create_care_configuration(
    experiment_name="care_3D",
    data_type="tiff",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    num_epochs=20,
    loss="mae",  # (1)!
)

# %%
# Example with N2V parameters
config = create_n2v_configuration(
    experiment_name="n2v_2D",
    data_type="tiff",
    axes="YX",
    patch_size=[64, 64],
    batch_size=8,
    num_epochs=20,
    roi_size=7,
    masked_pixel_percentage=0.5,
)

# %%
# Example with N2V2
config = create_n2v_configuration(
    experiment_name="n2v2_3D",
    data_type="tiff",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    num_epochs=20,
    use_n2v2=True,  # (1)!
)

# %%
# Example with structN2V
config = create_n2v_configuration(
    experiment_name="structn2v_3D",
    data_type="tiff",
    axes="ZYX",
    patch_size=[16, 64, 64],
    batch_size=8,
    num_epochs=20,
    struct_n2v_axis="horizontal",
    struct_n2v_span=5,
)
