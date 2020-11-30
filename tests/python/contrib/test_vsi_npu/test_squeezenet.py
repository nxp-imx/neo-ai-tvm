from tvm import relay
from tvm.relay import testing
import numpy as np
from infrastructure import verify_vsi_result

batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)

data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)
dtype="float32"

mod, params = relay.testing.squeezenet.get_workload()

verify_vsi_result(mod, params, data_shape, out_shape, dtype)
