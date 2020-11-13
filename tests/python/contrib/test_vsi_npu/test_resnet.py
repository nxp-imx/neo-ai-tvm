import tvm
import numpy as np
from tvm import relay
from tvm.relay.testing import resnet
from infrastructure import get_ref_result
from infrastructure import verify_vsi_result

batch_size = 1
num_class = 1000
image_shape = (3, 224, 224)
data_shape = (batch_size,) + image_shape
out_shape = (batch_size, num_class)
dtype="float32"

mod, params = relay.testing.resnet.get_workload(
    num_layers=50, batch_size=batch_size, image_shape=image_shape
)

verify_vsi_result(mod, params, data_shape, out_shape, dtype)
