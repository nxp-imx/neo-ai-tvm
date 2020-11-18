import tvm
from tvm import relay
from tvm.relay import testing
import numpy as np
from infrastructure import verify_vsi_result


def _single_operation_test(relay_nn_func, dtype, data_shape, out_shape,
        *args, **kargs):
    op_name = relay_nn_func.__name__.upper()
    print("Testing {0: <50}".format(op_name), end="")
    data = relay.var("data", shape=data_shape, dtype=dtype)

    out = relay_nn_func(data, *args, **kargs)

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

def test_avg_pool2d():
    func = relay.nn.avg_pool2d

    dtype = "float32"
    data_shape = (1, 1, 20, 20)
    out_shape = (1, 1, 10, 10)
    _single_operation_test(func, dtype, data_shape, out_shape, pool_size=(3, 3),
            strides=(2, 2), padding=(1,1,1,1))

def test_max_pool2d():
    func = relay.nn.max_pool2d

    dtype = "float32"
    data_shape = (1, 1, 20, 20)
    out_shape = (1, 1, 10, 10)
    _single_operation_test(func, dtype, data_shape, out_shape, pool_size=(2, 2),
            strides=(2, 2), padding=(0,0,0,0))

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

def test_add():
    dtype = "float32"
    data_shape = (1, 20, 12, 9)
    out_shape = data_shape

    def get_workload(data_shape, dtype="float32"):
        '''customized keywords(like data0,data1...) are not supported in \
        relay.testing.init.create_workload
        '''
        data0 = relay.var("data", shape=data_shape, dtype=dtype)
        data1 = relay.var("weight", shape=data_shape, dtype=dtype)

        out = relay.add(data0, data1)

        args = relay.analysis.free_vars(out)
        net = relay.Function(args, out)

        return relay.testing.init.create_workload(net)

    print("Testing {0: <50}".format("ADD"), end="")
    mod, params = get_workload(data_shape, dtype)
    verify_vsi_result(mod, params, data_shape, out_shape, dtype)


def test_batch_flatten():
    func = relay.nn.batch_flatten

    dtype = "float32"
    data_shape = (1, 5, 10, 10)
    out_shape = (1, 500)
    _single_operation_test(func, dtype, data_shape, out_shape)

def test_batch_norm():
    data_shape = (1, 4)
    c_shape = (4,)
    out_shape = (1, 4)
    dtype="float32"

    def get_workload(data_shape, weight_shape, dtype="float32"):
        data = relay.var("data", shape=data_shape, dtype=dtype)

        w = tvm.nd.array(np.ones(weight_shape, dtype))
        gamma = relay.const(w, dtype)
        beta = relay.const(w, dtype)
        moving_mean = relay.const(w, dtype)
        moving_var = relay.const(w, dtype)

        bn = relay.nn.batch_norm(data, gamma, beta, moving_mean, moving_var)
        out = bn[0]

        args = relay.analysis.free_vars(out)
        net = relay.Function(args, out)

        return relay.testing.init.create_workload(net)

    print("Testing {0: <50}".format("BATCH_NORM"), end="")
    mod, params = get_workload(data_shape, c_shape, dtype)
    verify_vsi_result(mod, params, data_shape, out_shape, dtype)


if __name__ == "__main__":
    test_batch_norm()
    test_softmax()
    test_add()
    test_relu()
    test_batch_flatten()
    test_avg_pool2d()
    test_max_pool2d()
    test_global_avg_pool2d()
    test_global_max_pool2d()

