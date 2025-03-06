import cv2
import numpy as np

from utility.loader import get_padded_copy

class WaveletCoder:
    """
    A basic class for image compression based on multi-resolution analysis.
    """

    def __init__(self):
        pass

    def get_small_copy(self, image, transform_depth, border_type=cv2.BORDER_REPLICATE, border_constant=0):
        """
        Resize the image to fit within the specified dimensions while maintaining the aspect ratio.

        Parameters:
            image (numpy.ndarray): The input image.
            transform_depth (int): The depth of the discrete wavelet transform (DWT).
            border_type (int): The padding type.
            border_constant (int): The padded value

        Returns:
            numpy.ndarray: Image small copy obtained using DWT of the specified depth
        """
        return None


class HaarCoder(WaveletCoder):
    """
    The simplified image compressor based on the Haar wavelet.
    """

    def __init__(self):
        super().__init__()
        self._ONE_STEP_RATIO = 2

    def get_small_copy(self, image, transform_depth, border_type=cv2.BORDER_REPLICATE, border_constant=0):
        """
        Resize the image to fit within the specified dimensions while maintaining the aspect ratio.

        Parameters:
            image (numpy.ndarray): The input image.
            transform_depth (int): The depth of the discrete wavelet transform (DWT).
            border_type (int): The padding type.
            border_constant (int): The padded value

        Returns:
            numpy.ndarray: Image small copy obtained using DWT of the specified depth
        """
        ratio = self._ONE_STEP_RATIO ** transform_depth

        low_left = get_padded_copy(image, ratio, border_type, border_constant).astype(np.float32)

        for i in range(transform_depth):
            evens, odds = low_left[::2, :, :], low_left[1::2, :, :]
            sums = evens + odds
            evens, odds = sums[:, ::2, :], sums[:, 1::2, :]
            low_left = (evens + odds) * 0.25

        return low_left.astype(np.uint8)