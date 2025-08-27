# WICCA: Wavelet-based Image Compression and Classification Analysis
# Copyright (C) 2025  Andrii Lesniak
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
from pathlib import Path

from wicca.config.aliases import Depth


def normalize_depth(depth: Depth) -> Depth:
    """
    Normalizes the given depth input.

    The function ensures the provided `depth` is converted into a uniform tuple of
    integers, ensuring compatibility and usability in further operations. It raises
    errors if the input does not conform to the expected types and conditions.

    Args:
        depth: The depth input to be normalized. It can be an integer greater than
            0, a tuple, a list, or a range. A `None` value and invalid types will
            raise respective errors.

    Returns:
        A tuple of integers representing the normalized depth.

    Raises:
        ValueError: If depth is not provided (None).
        ValueError: If depth is not a positive integer, tuple, list, or range.
        ValueError: If any element in the depth is not an integer.
    """
    if depth is None:
        raise ValueError("Depth must be provided")
    if isinstance(depth, int) and depth > 0:
        depth = (depth,)
    if isinstance(depth, (tuple, list, range)):
        depth = tuple(depth)
    else:
        raise ValueError("Depth must be a positive integer, tuple, list, or range")
    if all(isinstance(x, int) and x > 0 for x in depth):
        return depth
    else:
        raise ValueError("All depths must be integers greater than 0")


def normalize_folder(folder: str | Path) -> Path:
    """Normalizes a folder path"""
    if not isinstance(folder, (Path, str)):
        msg = f"Invalid input type: {type(folder)}. Expected str or Path."
        logging.error(msg)
        raise TypeError(msg)
    return Path(folder)
