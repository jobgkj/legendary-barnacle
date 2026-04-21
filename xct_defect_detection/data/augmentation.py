"""
=============================================================================
Data Augmentation Pipeline
=============================================================================
Implements 7 different augmentation transforms for XCT defect detection:
    1. Horizontal/vertical flips
    2. 90-degree rotations
    3. Elastic deformations
    4. Intensity scaling
    5. Gaussian noise
    6. Gamma correction
    7. Combined random application

All transforms preserve the mask-image correspondence exactly.
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


class XCTAugmentor:
    """Composable augmentation transforms for XCT patches."""

    @staticmethod
    def flip_horizontal(image: np.ndarray, mask: np.ndarray) -> tuple:
        """Flip horizontally with probability AUG_FLIP_PROB."""
        if np.random.rand() < AUG_FLIP_PROB:
            image = np.fliplr(image)
            mask  = np.fliplr(mask)
        return image, mask

    @staticmethod
    def flip_vertical(image: np.ndarray, mask: np.ndarray) -> tuple:
        """Flip vertically with probability AUG_FLIP_PROB."""
        if np.random.rand() < AUG_FLIP_PROB:
            image = np.flipud(image)
            mask  = np.flipud(mask)
        return image, mask

    @staticmethod
    def rotate_90(image: np.ndarray, mask: np.ndarray) -> tuple:
        """Randomly rotate 90, 180, or 270 degrees."""
        if np.random.rand() < AUG_ROTATE_PROB:
            k = np.random.randint(1, 4)  # 1 = 90°, 2 = 180°, 3 = 270°
            image = np.rot90(image, k=k)
            mask  = np.rot90(mask,  k=k)
        return image, mask

    @staticmethod
    def elastic_deformation(image: np.ndarray, mask: np.ndarray) -> tuple:
        """Apply elastic deformation via random displacement fields."""
        if np.random.rand() < AUG_ELASTIC_PROB:
            h, w = image.shape
            
            # Random displacement field
            dx = gaussian_filter(
                (np.random.rand(h, w) * 2 - 1) * AUG_ELASTIC_ALPHA,
                sigma=AUG_ELASTIC_SIGMA
            )
            dy = gaussian_filter(
                (np.random.rand(h, w) * 2 - 1) * AUG_ELASTIC_ALPHA,
                sigma=AUG_ELASTIC_SIGMA
            )

            # Apply displacement
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            indices_x = np.reshape(x + dx, (-1, 1))
            indices_y = np.reshape(y + dy, (-1, 1))
            indices   = np.concatenate([indices_y, indices_x], axis=1).T

            # Interpolate image and mask
            image = map_coordinates(image, indices, order=1, mode='reflect')
            image = image.reshape((h, w))
            mask  = map_coordinates(mask,  indices, order=0, mode='reflect')
            mask  = mask.reshape((h, w))

        return image, mask

    @staticmethod
    def intensity_scaling(image: np.ndarray, mask: np.ndarray) -> tuple:
        """Scale intensity by random factor in AUG_INTENSITY_RANGE."""
        if np.random.rand() < AUG_INTENSITY_PROB:
            factor = np.random.uniform(*AUG_INTENSITY_RANGE)
            image  = np.clip(image * factor, 0.0, 1.0)
        return image, mask

    @staticmethod
    def gaussian_noise(image: np.ndarray, mask: np.ndarray) -> tuple:
        """Add Gaussian noise with std in AUG_NOISE_STD_RANGE."""
        if np.random.rand() < AUG_NOISE_PROB:
            std   = np.random.uniform(*AUG_NOISE_STD_RANGE)
            noise = np.random.normal(0, std, image.shape)
            image = np.clip(image + noise, 0.0, 1.0)
        return image, mask

    @staticmethod
    def gamma_correction(image: np.ndarray, mask: np.ndarray) -> tuple:
        """Apply gamma correction with gamma in AUG_GAMMA_RANGE."""
        if np.random.rand() < AUG_GAMMA_PROB:
            gamma  = np.random.uniform(*AUG_GAMMA_RANGE)
            image  = np.power(image, gamma)
        return image, mask

    @staticmethod
    def augment(image: np.ndarray, mask: np.ndarray) -> tuple:
        """
        Apply full augmentation pipeline in order:
            1. Flips (horizontal + vertical)
            2. Rotations
            3. Elastic deformation
            4. Intensity scaling
            5. Gaussian noise
            6. Gamma correction

        Parameters
        ----------
        image : np.ndarray  — single 2D XCT patch, float32 in [0, 1]
        mask  : np.ndarray  — corresponding binary mask, uint8

        Returns
        -------
        tuple (augmented_image, augmented_mask)  — both float32
        """
        image, mask = XCTAugmentor.flip_horizontal(image, mask)
        image, mask = XCTAugmentor.flip_vertical(image, mask)
        image, mask = XCTAugmentor.rotate_90(image, mask)
        image, mask = XCTAugmentor.elastic_deformation(image, mask)
        image, mask = XCTAugmentor.intensity_scaling(image, mask)
        image, mask = XCTAugmentor.gaussian_noise(image, mask)
        image, mask = XCTAugmentor.gamma_correction(image, mask)

        return image.astype(np.float32), mask.astype(np.float32)