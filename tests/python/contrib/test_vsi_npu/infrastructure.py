import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_runtime
from tvm.relay.op.contrib import vsi_npu
from tvm import rpc
from tvm.contrib import util

HOST = "10.193.20.6"
PORT = 9090

def get_vsi_result(data, mod, params, out_shape, dtype):
    remote = rpc.connect(HOST, PORT)
    tmp_path = util.tempdir()
    lib_name = "mlp.so"
    lib_path = tmp_path.relpath(lib_name)

    kwargs = {}
    kwargs["cc"] = "aarch64-linux-gnu-g++"
    target = "llvm  -mtriple=aarch64-linux-gnu"
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        mod = vsi_npu.partition_for_vsi_npu(mod, params)
        lib  = relay.build(mod, target, params=params)
        lib.export_library(lib_path, fcompile=False, **kwargs)
        #print(mod.astext(show_meta_data=False))

    remote.upload(lib_path)
    lib = remote.load_module(lib_name)
    ctx = remote.cpu()

    rt_mod = graph_runtime.GraphModule(lib["default"](ctx))
    rt_mod.set_input("data", data)
    rt_mod.run()
    rt_out = tvm.nd.array(np.zeros(out_shape, dtype=dtype), ctx)
    rt_mod.get_output(0, rt_out)
    return rt_out

def get_ref_result(data, mod, params, out_shape, dtype):
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        lib  = relay.build(mod, target, params=params)
    cpu_mod = graph_runtime.GraphModule(lib["default"](tvm.cpu()))
    cpu_mod.set_input("data", data)
    cpu_mod.run()
    cpu_out = cpu_mod.get_output(0, tvm.nd.empty(out_shape, dtype))
    return cpu_out

def verify_vsi_result(mod, params, data_shape, out_shape, dtype="float32"):
    data = np.random.uniform(size=data_shape).astype(dtype)

    vsi_out = get_vsi_result(data, mod, params, out_shape, dtype)
    ref_out = get_ref_result(data, mod, params, out_shape, dtype)
    print(vsi_out.asnumpy())
    print(ref_out.asnumpy())

    tol = 1e-5
    tvm.testing.assert_allclose(ref_out.asnumpy(), vsi_out.asnumpy(), rtol=tol, atol=tol)
    print("PASS")
