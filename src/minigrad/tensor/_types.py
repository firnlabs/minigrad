from collections.abc import Sequence
from typing import Any
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

NumArray: TypeAlias = NDArray[np.integer[Any] | np.floating[Any]]
NestedSequence: TypeAlias = int | float | Sequence["NestedSequence"]
