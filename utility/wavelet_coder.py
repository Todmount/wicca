from abc import ABC, abstractmethod
import cv2
import numpy as np

from utility.data_loader import get_padded_copy


class WaveletCoder(ABC):
    """
    Abstract base class for image compression based on multi-resolution analysis.
    """

    @abstractmethod
    def get_small_copy(self, image: np.ndarray, transform_depth: int,
                       border_type: int = cv2.BORDER_REPLICATE,
                       border_constant: int = 0
                       ) -> np.ndarray:
        """
        Resize the image using wavelet transform.
        """
        pass


class HaarCoder(WaveletCoder):
    """
    The simplified image compressor based on the Haar wavelet.
    """

    def __init__(self):
        super().__init__()
        self._ONE_STEP_RATIO = 2

    def get_small_copy(self, image: np.ndarray,
                       transform_depth: int,
                       border_type: int = cv2.BORDER_REPLICATE,
                       border_constant: int = 0
                       ) -> np.ndarray:

        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy array")
        if transform_depth < 0:
            raise ValueError("Transform depth must be non-negative")
        if not isinstance(transform_depth, int):
            raise TypeError("Transform depth must be an integer")

        ratio = self._ONE_STEP_RATIO ** transform_depth
        low_left = get_padded_copy(image, ratio, border_type, border_constant).astype(np.float32)

        for _ in range(transform_depth):
            evens, odds = low_left[::2, :, :], low_left[1::2, :, :]
            sums = evens + odds
            evens, odds = sums[:, ::2, :], sums[:, 1::2, :]
            low_left = (evens + odds) * 0.25

        return np.clip(low_left, 0, 255).astype(np.uint8)
