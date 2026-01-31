from pathlib import Path
from typing import Union

import numpy as np
import tifffile
import torch
from careamics.dataset.dataset_utils.dataset_utils import reshape_array
from careamics.lvae_training.dataset import DataSplitType
from careamics.lvae_training.dataset.utils.data_utils import get_datasplit_tuples
from careamics.lvae_training.eval_utils import get_device

# TODO move all this to careamics


def load_one_file(fpath):
    """Load a single 2D image file."""
    data = tifffile.imread(fpath)
    if len(data.shape) == 2:
        axes = "YX"
    elif len(data.shape) == 3:
        axes = "SYX"
    elif len(data.shape) == 4:
        axes = "STYX"
    else:
        raise ValueError(f"Invalid data shape: {data.shape}")
    data = reshape_array(data, axes)
    data = data.reshape(-1, data.shape[-2], data.shape[-1])
    return data


def load_data(datadir): # TODO probably doesn't work and obsolete
    data_path = Path(datadir)

    channel_dirs = sorted(p for p in data_path.iterdir() if p.is_dir())
    channels_data = []

    for channel_dir in channel_dirs:
        image_files = sorted(f for f in channel_dir.iterdir() if f.is_file())
        channel_images = [load_one_file(image_path) for image_path in image_files]

        channel_stack = np.concatenate(
            channel_images, axis=0
        )  # FIXME: this line works iff images have
        # a singleton channel dimension. Specify in the notebook or change with `torch.stack`??
        channels_data.append(channel_stack)

    final_data = np.stack(channels_data, axis=-1)
    return final_data


def get_train_val_data(
    data_config=None,
    datadir=None,
    datasplit_type: DataSplitType = None,
    val_fraction=None,
    test_fraction=None,
    **kwargs,
):
    data = load_data(datadir)
    train_idx, val_idx, test_idx = get_datasplit_tuples(
        val_fraction, test_fraction, len(data)
    )
    # FIXME: this is a hack to make the data split work with 2D custom datasets
    # val_idx = train_idx
    # test_idx = train_idx
    if datasplit_type == DataSplitType.All:
        data = data.astype(np.float64)
    elif datasplit_type == DataSplitType.Train:
        data = data[train_idx].astype(np.float64)
    elif datasplit_type == DataSplitType.Val:
        data = data[val_idx].astype(np.float64)
    elif datasplit_type == DataSplitType.Test:
        data = data[test_idx].astype(np.float64)
    else:
        raise Exception("invalid datasplit")

    return data


def get_test_data(datadir: Union[str, Path]) -> np.ndarray:
    """
    Load the complete test dataset without applying any data split selection.

    Parameters
    ----------
    datadir : Union[str, Path]
        Directory containing the preprocessed test dataset.

    Returns
    -------
    np.ndarray
        Full test dataset as a float64 NumPy array.
    """
    data = load_data(datadir)
    return data.astype(np.float64)


def load_pretrained_model(model, ckpt_path):
    device = get_device()
    ckpt_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt_dict['state_dict'], strict=False)
    print(f"Loaded model from {ckpt_path}")
