import json
from pathlib import Path
import pickle

import h5py
from matplotlib import pyplot as plt
from natsort import natsorted
import numpy as np
import pandas as pd
import tifffile
from vedo import Volume, show
import napari

from brainrender import Scene
from brainrender import actors
from brainrender.actor import Actor
from brainrender.actors import Points
from brainglobe_utils.brainreg.transform import (
    get_anatomical_space_from_image_path,
    transform_points_to_atlas_space,
)
from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_utils.cells.cells import to_numpy_pos

from bs4 import BeautifulSoup

from utils import load_bigstitched_data, substring_in_array_element


def display(
    coordinates: np.ndarray, roi: str, actual_brain: np.ndarray
) -> None:

    # # Display the Allen Brain mouse atlas.
    scene = Scene(
        atlas_name="perens_lsfm_mouse_20um",
        title="perens",
    )
    plane = scene.atlas.get_plane(pos=[0, 0, 0], plane="frontal")

    # # Display a brain region
    roi_area = scene.add_brain_region(roi, alpha=0.2)

    # # Create a Points actor

    # coordinates = coordinates[:, [0, 2, 1]]  # convert to (x, y, z) for display
    cells = Points(coordinates)

    # # Add to scene
    scene.add(cells, transform=True)

    # # Add label to the brain region
    # scene.add_label(roi_area, roi)

    actual_brain = np.transpose(actual_brain, axes=[2, 1, 0])

    actual_brain[actual_brain != 0] = 255
    vol = Volume(actual_brain, spacing=(20, 20, 20))

    mesh = vol.isosurface(value=255)
    scene.add(
        Actor(
            mesh,
            alpha=0.1,
        ),
        transform=True,
        voxel_size=20,
    )

    # Display the figure.
    scene.render()


def get_bigstitched_data_shape(
    path: Path,
    pyramid_level: int,
    channel: int,
) -> np.ndarray:

    first_level = "t00000"
    with h5py.File(path, "r") as f:
        channels = sorted(list(f[first_level].keys()))
        return f[first_level][channels[channel]][str(pyramid_level)][
            "cells"
        ].shape


def score_points_in_atlas(
    points_in_atlas_voxels: np.ndarray,
    atlas: BrainGlobeAtlas,
    max_points: int = 2000,
) -> float:
    """
    Score transformed points by fraction that map to a valid atlas structure.
    """
    n_points = len(points_in_atlas_voxels)
    if n_points == 0:
        return 0.0

    if n_points > max_points:
        idx = np.linspace(0, n_points - 1, max_points, dtype=int)
        sample = points_in_atlas_voxels[idx]
    else:
        sample = points_in_atlas_voxels

    valid = 0
    for point in sample:
        try:
            _ = atlas.structure_from_coords(point)
            valid += 1
        except Exception:
            continue

    return valid / len(sample)


def convert_points_to_axis_space(umbrella: Path) -> np.ndarray:
    """
    Data that we have:

    The points in the original orientation and resolution of the data.
    The registered atlas in the brainreg folder. We have this both in data orientation and the atlas orientation. Both of these are downsampled.

    We can use either orientation of atlas for visualisation. But we need to check the csv file that tells you which region is which.
    I believe the csv file is in the atlas orientation, so we should use the atlas orientation for visualisation.
    We can use the data orientation for checking the points are in the right place.
    We have the deformation field that goes from data space to atlas space.

    So we need to
    1) downsample the points to the atlas resolution
    2) transform the points to the atlas space using the deformation field

    There are three spaces used for this:
    original space: the space of the raw data, in the original orientation and resolution
    downsampled space: the space of the downsampled data, in the original orientation
    atlas space: the space of the atlas, in the atlas orientation

    Whatever data you give to brainreg, it will downsample it to match the atlas resolution
    So I think the downsampled space is the same for both the data and the atlas, but they are in different orientations. The deformation field will then transform from the downsampled space to the atlas space.

    I actually think the downsampled space needs to be in the atlas orientation.
    May or may not be working currently, proof in the pudding.

    The atlas is displayed in microns, so do we need to multiple by the voxel sizes?

    """

    full_data = umbrella / "stitched.h5"
    xml_path = umbrella / "stitched.xml"

    # (z, y, x) I think
    full_data_shape = get_bigstitched_data_shape(
        path=full_data,
        pyramid_level=0,
        channel=0,
    )

    with open(xml_path, "r") as f:
        xml = f.read()
    soup = BeautifulSoup(xml, "xml")
    x_vox_original, y_vox_original, z_vox_original = map(
        float, soup.find("voxelSize").size.text.split(" ")
    )

    with open(umbrella / "brainreg" / "brainreg.json", "r") as f:
        brainreg_info = json.loads(f.read())

    original_orientation = brainreg_info["orientation"]

    atlas = BrainGlobeAtlas(atlas_name="perens_lsfm_mouse_20um")

    # brainreg's downsampled space is in atlas orientation and atlas resolution.
    downsampled_space = get_anatomical_space_from_image_path(
        image_path=umbrella / "brainreg" / "registered_atlas.tiff",
        orientation=atlas.orientation,
        voxel_sizes=[float(v) for v in atlas.resolution],
    )

    with open(umbrella / "cellfinder" / "detected_cells.pkl", "rb") as f:
        coordinates = pickle.load(f)

    defomation_field_paths = natsorted(
        list((umbrella / "brainreg").glob("deformation_field_*.tiff"))
    )

    assert (
        len(defomation_field_paths) == 3
    ), "Expected 3 deformation fields, one for each axis"

    print("Transforming points to atlas space...")

    points = to_numpy_pos(coordinates)

    # Try the two most likely raw coordinate conventions and keep the one
    # that best maps into valid atlas structures.
    candidate_point_orders = {
        "xyz": points,
        "zyx": points[:, [2, 1, 0]],
    }

    candidate_results = {}
    for point_order_name, candidate_points in candidate_point_orders.items():
        transformed = transform_points_to_atlas_space(
            points=candidate_points,
            source_image_plane=np.zeros(
                full_data_shape
            ),  # dummy image plane just to get the shape
            orientation=original_orientation,
            voxel_sizes=[z_vox_original, y_vox_original, x_vox_original],
            downsampled_space=downsampled_space,
            atlas=atlas,
            deformation_field_paths=defomation_field_paths,
        )

        score = score_points_in_atlas(transformed[0], atlas)

        candidate_results[point_order_name] = {
            "result": transformed,
            "score": score,
        }

        print(
            f"Point order '{point_order_name}': atlas-valid fraction = {score:.3f}"
        )

    # Chatgpt wrote this hack to check which way round it should be without prior knowledge. I have
    # disabled it but I believe the result was correct. The one it chose was zyx. When plotting both of them
    # xyz only gives cells on one side of the brain, whereas zyx gives cells on both sides. So I think it chose correctly.
    # Obviously can take this computation out going forward

    # best_point_order = max(
    #     candidate_results, key=lambda key: candidate_results[key]["score"]
    # )
    # print(f"Using point order: '{best_point_order}'")
    # res = candidate_results["xyz"]["result"]

    res = candidate_results["zyx"]["result"]

    with open("result.pkl", "wb") as f:
        pickle.dump(res, f)

    return res[0]


def main() -> None:
    umbrella = Path("/Volumes/MarcBusche/James/Mesospim/2026-03-03/N011/001")
    # res = convert_points_to_axis_space(umbrella)

    with open("result.pkl", "rb") as f:
        res = pickle.load(f)

    atlas = BrainGlobeAtlas(atlas_name="perens_lsfm_mouse_20um")
    resolution_um = np.array(atlas.resolution, dtype=float)
    structures = []
    for i, point in enumerate(res[0]):
        try:
            structure = atlas.structure_from_coords(point, as_acronym=True)
        except Exception:
            structure = "Unknown"
        structures.append(structure)

    structures = np.array(structures)

    roi = "RSP"
    mask = substring_in_array_element(structures, roi)

    actual_brain = tifffile.imread(
        umbrella / "brainreg" / "downsampled_standard.tiff"
    )

    # mask_cent = substring_in_array_element(structures, "CENT")
    # mask = mask & ~mask_cent

    result = res[0][mask]

    threshold = np.percentile(actual_brain, 50)
    thresholded = actual_brain.copy()
    thresholded[thresholded < threshold] = 0

    display(result * resolution_um, roi, thresholded)


if __name__ == "__main__":
    main()
