"""
=============================================================================
Data Augmentation
=============================================================================
Online augmentation transforms applied during training to improve model
robustness and reduce overfitting to scan-specific characteristics.

All transforms operate on (H, W) numpy arrays and are applied identically
to the image patch and its corresponding mask patch to maintain alignment.

Since the 2.5D dataset (data/dataset.py) pairs a stacked (N, H, W) image
with a single (H, W) mask, every geometric transform below acts on the
trailing two axes specifically (H, W) rather than assuming a bare 2D
array — so the same flip/rotation/deformation is applied identically to
every slice-channel of the image stack and to the mask, keeping them
spatially aligned regardless of N.
=============================================================================
"""

import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    AUG_FLIP_PROB, AUG_ROTATE_PROB, AUG_ELASTIC_PROB,
    AUG_ELASTIC_ALPHA, AUG_ELASTIC_SIGMA,
    AUG_INTENSITY_PROB, AUG_INTENSITY_RANGE,
    AUG_NOISE_PROB, AUG_NOISE_STD_RANGE,
    AUG_GAMMA_PROB, AUG_GAMMA_RANGE
)


def random_flip(image: np.ndarray,
                mask:  np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly flip image and mask horizontally and/or vertically.

    Parameters
    ----------
    image : np.ndarray  — float32 patch (H, W) or (N, H, W)
    mask  : np.ndarray  — uint8 mask patch (H, W)

    Returns
    -------
    Augmented (image, mask) tuple.
    """
    # axis=-1/-2 (not np.fliplr/np.flipud, which flip fixed axes 1/0) so
    # this flips the trailing (H, W) axes the same way whether image is
    # 2D (H, W) or a 2.5D stack (N, H, W).
    if np.random.random() < AUG_FLIP_PROB:
        image = np.flip(image, axis=-1)
        mask  = np.flip(mask,  axis=-1)
    if np.random.random() < AUG_FLIP_PROB:
        image = np.flip(image, axis=-2)
        mask  = np.flip(mask,  axis=-2)
    return image, mask


def random_rotate90(image: np.ndarray,
                    mask:  np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly rotate image and mask by a multiple of 90 degrees.

    Parameters
    ----------
    image : np.ndarray  — float32 patch (H, W) or (N, H, W)
    mask  : np.ndarray  — uint8 mask patch (H, W)

    Returns
    -------
    Augmented (image, mask) tuple.
    """
    if np.random.random() < AUG_ROTATE_PROB:
        k     = np.random.choice([1, 2, 3])    # 90, 180, or 270 degrees
        # axes=(-2, -1): rotate the trailing (H, W) plane only, so a 2.5D
        # (N, H, W) stack rotates every slice-channel identically instead
        # of rotating across the N/H axes.
        image = np.rot90(image, k, axes=(-2, -1))
        mask  = np.rot90(mask,  k, axes=(-2, -1))
    return image, mask


def elastic_deformation(image: np.ndarray,
                        mask:  np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply random elastic deformation to simulate scan variability.

    Generates a smooth random displacement field by convolving a random
    noise field with a Gaussian kernel. The same displacement field is
    applied to both image and mask to preserve alignment.

    Parameters
    ----------
    image : np.ndarray  — float32 patch (H, W) or (N, H, W)
    mask  : np.ndarray  — uint8 mask patch (H, W)

    Returns
    -------
    Augmented (image, mask) tuple.
    """
    if np.random.random() >= AUG_ELASTIC_PROB:
        return image, mask

    h, w = mask.shape  # mask is always (H, W), even when image is a 2.5D stack

    # Random displacement fields
    dx = gaussian_filter(
        (np.random.rand(h, w) * 2 - 1), AUG_ELASTIC_SIGMA
    ) * AUG_ELASTIC_ALPHA

    dy = gaussian_filter(
        (np.random.rand(h, w) * 2 - 1), AUG_ELASTIC_SIGMA
    ) * AUG_ELASTIC_ALPHA

    # Compute displaced coordinates
    x, y      = np.meshgrid(np.arange(w), np.arange(h))
    coords_x  = np.clip(x + dx, 0, w - 1)
    coords_y  = np.clip(y + dy, 0, h - 1)
    coords    = [coords_y.ravel(), coords_x.ravel()]

    if image.ndim == 2:
        image_def = map_coordinates(image, coords, order=1).reshape(h, w)
    else:
        # (N, H, W) 2.5D stack — apply the *same* displacement field to
        # every slice-channel so all channels stay spatially aligned with
        # each other and with the (single) mask.
        image_def = np.stack([
            map_coordinates(image[c], coords, order=1).reshape(h, w)
            for c in range(image.shape[0])
        ], axis=0)

    mask_def = map_coordinates(
        mask.astype(np.float32), coords, order=0
    ).reshape(h, w).astype(np.uint8)

    return image_def.astype(np.float32), mask_def


def random_intensity_scale(image: np.ndarray,
                            mask:  np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly scale voxel intensities to simulate acquisition variability.

    Parameters
    ----------
    image : np.ndarray  — float32 patch (H, W), values in [0, 1]
    mask  : np.ndarray  — uint8 mask patch (H, W)  [unchanged]

    Returns
    -------
    Augmented (image, mask) tuple.
    """
    if np.random.random() < AUG_INTENSITY_PROB:
        scale = np.random.uniform(*AUG_INTENSITY_RANGE)
        image = np.clip(image * scale, 0.0, 1.0)
    return image, mask


def random_gaussian_noise(image: np.ndarray,
                           mask:  np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Add random Gaussian noise to simulate detector noise variability.

    Parameters
    ----------
    image : np.ndarray  — float32 patch (H, W), values in [0, 1]
    mask  : np.ndarray  — uint8 mask patch (H, W)  [unchanged]

    Returns
    -------
    Augmented (image, mask) tuple.
    """
    if np.random.random() < AUG_NOISE_PROB:
        std   = np.random.uniform(*AUG_NOISE_STD_RANGE)
        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)
    return image, mask


def random_gamma_correction(image: np.ndarray,
                             mask:  np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply random gamma correction to simulate contrast variability.

    Parameters
    ----------
    image : np.ndarray  — float32 patch (H, W), values in [0, 1]
    mask  : np.ndarray  — uint8 mask patch (H, W)  [unchanged]

    Returns
    -------
    Augmented (image, mask) tuple.
    """
    if np.random.random() < AUG_GAMMA_PROB:
        gamma = np.random.uniform(*AUG_GAMMA_RANGE)
        image = np.clip(image ** gamma, 0.0, 1.0)
    return image, mask


def apply_augmentation(image: np.ndarray,
                        mask:  np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the full augmentation pipeline to an (image, mask) patch pair.

    All transforms are applied in sequence with their configured
    probabilities. The same random state affects both image and mask
    to maintain spatial alignment.

    Parameters
    ----------
    image : np.ndarray  — float32 patch (H, W) or (N, H, W)
    mask  : np.ndarray  — uint8 mask patch (H, W)

    Returns
    -------
    Augmented (image, mask) tuple — same dtypes and shapes as input.
    """
    image, mask = random_flip(image,             mask)
    image, mask = random_rotate90(image,         mask)
    image, mask = elastic_deformation(image,     mask)
    image, mask = random_intensity_scale(image,  mask)
    image, mask = random_gaussian_noise(image,   mask)
    image, mask = random_gamma_correction(image, mask)
    return image, mask
