# Copyright (c) Meta Platforms, Inc. and affiliates.

import numpy as np
from scipy.spatial.transform import Rotation


def crop_map(raster, xy, size, seed=None):
    h, w = raster.shape[-2:]
    state = np.random.RandomState(seed)
    top = state.randint(0, h - size + 1)
    left = state.randint(0, w - size + 1)
    raster = raster[..., top : top + size, left : left + size]
    xy -= np.array([left, top])
    return raster, xy


def random_rot90(raster, xy, heading, seed=None):
    rot = np.random.RandomState(seed).randint(0, 4)
    heading = (heading + rot * np.pi / 2) % (2 * np.pi)
    h, w = raster.shape[-2:]
    if rot == 0:
        xy2 = xy
    elif rot == 2:
        xy2 = np.array([w, h]) - 1 - xy
    elif rot == 1:
        xy2 = np.array([xy[1], w - 1 - xy[0]])
    elif rot == 3:
        xy2 = np.array([h - 1 - xy[1], xy[0]])
    else:
        raise ValueError(rot)
    raster = np.rot90(raster, rot, axes=(-2, -1))
    return raster, xy2, heading


def flip_decision(seed=None):
    """The random draws of random_flip: (mirrored, raster mirrored along x rather than along y)."""
    state = np.random.RandomState(seed)
    mirrored = bool(state.rand() <= 0.5)
    return mirrored, bool(state.rand() > 0.5) if mirrored else False


def mirror_raster(raster, flip_x):
    """Mirror a raster [..., H, W] along x (columns) or along y (rows)."""
    return raster[..., :, ::-1] if flip_x else raster[..., ::-1, :]


def random_flip(image, raster, xy, heading, seed=None):
    """Mirror the image with probability 1/2; the map raster is then mirrored along x or y (1/2 each),
    with the pixel position xy and the heading (rad) following it."""
    mirrored, flip_x = flip_decision(seed)
    if not mirrored:
        return image, raster, xy, heading
    image = image[:, ::-1]
    raster = mirror_raster(raster, flip_x)
    h, w = raster.shape[-2:]
    if flip_x:
        xy = np.array([w - 1 - xy[0], xy[1]])
        heading = np.pi - heading
    else:
        xy = np.array([xy[0], h - 1 - xy[1]])
        heading = -heading
    return image, raster, xy, heading % (2 * np.pi)


def decompose_rotmat(R_c2w):
    R_cv2xyz = Rotation.from_euler("X", -90, degrees=True)
    rot_w2c = R_cv2xyz * Rotation.from_matrix(R_c2w).inv()
    roll, pitch, yaw = rot_w2c.as_euler("YXZ", degrees=True)
    # R_plane2c = Rotation.from_euler("ZX", [roll, pitch], degrees=True).as_matrix()
    return roll, pitch, yaw
