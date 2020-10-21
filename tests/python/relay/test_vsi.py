# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Unit tests for graph partitioning."""
import os
import sys
import numpy as np

import tvm
from tvm import te
import tvm.relay.testing
import tvm.relay.transform
from tvm import relay
from tvm import runtime
from tvm.contrib import util


def check_result(mod, map_inputs, out_shape, result, tol=1e-5, target="llvm", ctx=tvm.cpu()):
    if sys.platform == "win32":
        print("Skip test on Windows for now")
        return

    def update_lib(lib):
        test_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
        source_dir = os.path.join(test_dir, "..", "..", "..")
        contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")

        kwargs = {}
        kwargs["options"] = ["-O2", "-std=c++14", "-I" + contrib_path]
        tmp_path = util.tempdir()
        lib_name = "lib.so"
        lib_path = tmp_path.relpath(lib_name)
        lib.export_library(lib_path, fcompile=False, **kwargs)
        lib = tvm.runtime.load_module(lib_path)

        return lib

    def check_vm_result():
        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            exe = relay.vm.compile(mod, target=target)
        code, lib = exe.save()

        lib = update_lib(lib)
        exe = runtime.vm.Executable.load_exec(code, lib)
        vm = runtime.vm.VirtualMachine(exe, ctx)
        out = vm.run(**map_inputs)
        tvm.testing.assert_allclose(out.asnumpy(), result, rtol=tol, atol=tol)

    def check_graph_runtime_result():
        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            json, lib, _ = relay.build(mod, target=target)

        lib = update_lib(lib)
        rt_mod = tvm.contrib.graph_runtime.create(json, lib, ctx)

        for name, data in map_inputs.items():
            rt_mod.set_input(name, data)
        rt_mod.run()
        out = tvm.nd.empty(out_shape, ctx=ctx)
        out = rt_mod.get_output(0, out)

        tvm.testing.assert_allclose(out.asnumpy(), result, rtol=tol, atol=tol)

    check_vm_result()
    check_graph_runtime_result()

def set_external_func_attr(func, compiler, ext_symbol):
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", compiler)
    func = func.with_attr("global_symbol", ext_symbol)
    return func

def test_vsi_codegen():
    if not tvm.get_global_func("relay.ext.vsi_npu", True):
        print("skip because VSI NPU codegen is not available")
        return

    dtype = "float32"
    ishape = (1, 32, 14, 14)
    w1shape = (1, 32, 14, 14)

    data0 = relay.var("data0", shape=(ishape), dtype=dtype)
    data1 = relay.var("data1", shape=(ishape), dtype=dtype)

    x = relay.var("x", shape=(ishape), dtype=dtype)
    y = relay.var("y", shape=(ishape), dtype=dtype)

    out = relay.add(data0, data1)

    f = relay.Function([data0, data1], out)
    ref_mod = tvm.IRModule()
    ref_mod["main"] = f

    f = set_external_func_attr(f, "vsi_npu", "vsi_npu_0")
    call = relay.Call(f, [x, y])
    mod = tvm.IRModule.from_expr(call)

    l_data = np.random.uniform(0, 1, ishape).astype(dtype)
    r_data = np.random.uniform(0, 1, w1shape).astype(dtype)

    ref_ex = relay.create_executor("graph", mod=ref_mod, ctx=tvm.cpu())
    ref_res = ref_ex.evaluate()(l_data, r_data)
    check_result(
        mod, {"x": l_data, "y": r_data}, (1, 32, 14, 14), ref_res.asnumpy(), tol=1e-5
    )


if __name__ == "__main__":
    test_vsi_codegen()
