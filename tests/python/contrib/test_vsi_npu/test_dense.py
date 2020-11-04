from tvm import relay
from tvm.relay import testing
import numpy as np
from infrastructure import verify_vsi_result

data_shape = (1, 784)
weight_shape = (128, 784)
out_shape = (1, 128)
dtype="float32"

def get_workload(data_shape, weight_shape, dtype="float32"):
    data = relay.var("data", shape=data_shape, dtype=dtype)
    fc1 = relay.nn.dense(data, relay.var("fc1_weight"), units=weight_shape[0])
    fc = relay.nn.bias_add(fc1, relay.var("fc1_bias"), axis=1)
    args = relay.analysis.free_vars(fc)
    net = relay.Function(args, fc)

    return relay.testing.init.create_workload(net)


mod, params = get_workload(data_shape, weight_shape, dtype)

verify_vsi_result(mod, params, data_shape, out_shape, dtype)
