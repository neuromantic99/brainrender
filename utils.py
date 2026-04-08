from pathlib import Path
from typing_extensions import Literal

import h5py
import numpy as np
from bs4 import BeautifulSoup


def substring_in_array_element(
    array: np.ndarray, substring: str
) -> np.ndarray:
    """Check if a substring is present in any element of a numpy array.

    Parameters
    ----------
    array : np.ndarray
        The input array to check for the presence of the substring.
    substring : str
        The substring to search for within the elements of the array.

    Returns
    -------
    np.ndarray
        A boolean array of the same shape as the input array, where each
        element is True if the corresponding element in the input array contains
        the substring, and False otherwise.
    """
    return np.array([substring in str(element) for element in array])


def get_wavelengths_of_channels(xml_path: Path) -> dict[str, int]:

    with open(xml_path, "r") as f:
        xml_content = f.read()
    soup = BeautifulSoup(xml_content, "xml")
    channels = soup.find_all("Channel")
    return {
        channel.find("name").text: int(channel.find("id").text)
        for channel in channels
    }


def load_bigstitched_data(
    path: Path,
    pyramid_level: int,
    channel: Literal["488 nm", "638 nm"],
    stack_start: int,
    stack_end: int,
) -> np.ndarray:

    first_level = "t00000"
    # I think bigstitcher gives the xml the same name as the h5 always
    wavelength_lookup = get_wavelengths_of_channels(
        path.parent / path.name.replace("h5", "xml")
    )
    channel_id = wavelength_lookup[channel]
    with h5py.File(path, "r") as f:
        channels = sorted(list(f[first_level].keys()))
        array = f[first_level][channels[channel_id]][str(pyramid_level)][
            "cells"
        ][stack_start:stack_end, :, :]

    return array
