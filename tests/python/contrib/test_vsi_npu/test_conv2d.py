import tvm
from tvm import relay
from tvm.relay import testing
import numpy as np
from infrastructure import verify_vsi_result

data_shape = (1, 256, 64, 64)
weight_shape = (256, 256, 3, 3)
out_shape = (1, 256, 64, 64)
dtype="float32"
Pad=(1,1,1,1)
Strides=(1,1)
Dilation=(1,1)
Ksize=(3,3)
Groups=1

def get_workload(data_shape, weight_shape, dtype="float32"):
    """Function to construct a MobileNet"""
    data = relay.var("data", shape=data_shape, dtype=dtype)
    weight = relay.var("conv_weight")
    conv = relay.nn.conv2d(
        data,
        weight,
        channels=weight_shape[0],
        kernel_size=Ksize,
        strides=Strides,
        padding=Pad,
        groups=Groups,
        data_layout="NCHW",
        kernel_layout="OIHW"
    )

    args = relay.analysis.free_vars(conv)
    net = relay.Function(args, conv)

    return relay.testing.init.create_workload(net)


mod, params = get_workload(data_shape, weight_shape, dtype)

verify_vsi_result(mod, params, data_shape, out_shape, dtype)
