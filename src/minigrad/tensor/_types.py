from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

NumericArray: TypeAlias = NDArray[np.float64 | np.int64]
NestedSequence: TypeAlias = int | float | Sequence["NestedSequence"]
