from typing import Any
from collections.abc import Callable

# Type aliases
ModelClass = Callable
ModelWithConfig = tuple[ModelClass, dict[str | Any]]  # For models with config like NASNetLarge
ModelsDict = dict[str, ModelClass | ModelWithConfig]
Depth = int | tuple[int, ...] | list[int] | range