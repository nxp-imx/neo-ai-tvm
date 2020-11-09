from tvm import relay
from tvm.relay import testing
import numpy as np
from infrastructure import verify_vsi_result

batch_size = 1
num_class = 10
image_shape = (1, 28, 28)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)
dtype="float32"

mod, params = relay.testing.mlp.get_workload(
    batch_size=batch_size, num_classes=num_class, image_shape=image_shape
)

verify_vsi_result(mod, params, data_shape, out_shape, dtype)