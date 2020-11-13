from tvm import relay
from tvm.relay import testing
import numpy as np
from infrastructure import verify_vsi_result


def _single_operation_test(relay_nn_func, dtype, data_shape, out_shape, *args):
    op_name = relay_nn_func.__name__.upper()
    print("\n================ testing %s =================" %(op_name))
    data = relay.var("data", shape=data_shape, dtype=dtype)

    if args:
        out = relay_nn_func(data, args)
    else:
        out = relay_nn_func(data)

    args = relay.analysis.free_vars(out)
    net = relay.Function(args, out)

    mod, params = relay.testing.init.create_workload(net)

    verify_vsi_result(mod, params, data_shape, out_shape, dtype)

def test_global_avg_pool2d():
    func = relay.nn.global_avg_pool2d

    dtype = "float32"
    data_shape = (1, 20, 12, 9)
    out_shape = (1, 20, 1, 1)
    _single_operation_test(func, dtype, data_shape, out_shape)


def test_global_max_pool2d():
    func = relay.nn.global_max_pool2d

    dtype = "float32"
    data_shape = (1, 20, 12, 9)
    out_shape = (1, 20, 1, 1)
    _single_operation_test(func, dtype, data_shape, out_shape)


def test_softmax():
    func = relay.nn.softmax

    dtype = "float32"
    data_shape = (1, 20, 12, 9)
    out_shape = data_shape
    _single_operation_test(func, dtype, data_shape, out_shape)

def test_relu():
    func = relay.nn.relu

    dtype = "float32"
    data_shape = (1, 20, 12, 9)
    out_shape = data_shape
    _single_operation_test(func, dtype, data_shape, out_shape)

def test_batch_flatten():
    func = relay.nn.batch_flatten

    dtype = "float32"
    data_shape = (1, 5, 10, 10)
    out_shape = (1, 500)
    _single_operation_test(func, dtype, data_shape, out_shape)

if __name__ == "__main__":
    test_softmax()
    test_relu()
    test_batch_flatten()
    test_global_avg_pool2d()
    test_global_max_pool2d()

