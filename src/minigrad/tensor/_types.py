"""Project-defined types for tensor."""

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

NumericArray: TypeAlias = NDArray[np.float64 | np.int64]
