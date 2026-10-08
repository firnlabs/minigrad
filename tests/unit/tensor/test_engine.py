import numpy as np

from minigrad.tensor import Tensor


class TestTensor:
    def test_tensor_1d_default_optional_params(self) -> None:
        data = [1.0, 2.0, 3.0]
        t = Tensor(data)

        assert np.allclose(t.data, np.array(data))
        assert t.op is None
        assert t.name == "unnamed"
        assert t.requires_grad is False
