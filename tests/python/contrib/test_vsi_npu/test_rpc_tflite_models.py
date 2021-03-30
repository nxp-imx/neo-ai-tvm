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
import os
import sys
import numpy as np
import argparse

import tvm
from tvm import rpc
from tvm.contrib import graph_runtime

from tflite_deeplab import *
from tflite_models import *

RPC_HOST = ""
RPC_PORT = 9090
MEASURE_PERF = False
VERBOSE = False

def inference_remotely(tfmodel, lib_path, image_data):
    remote = rpc.connect(RPC_HOST, RPC_PORT)
    remote.upload(lib_path)
    lib = remote.load_module(os.path.basename(lib_path))
    ctx = remote.cpu()

    # Create a runtime executor module
    module = graph_runtime.GraphModule(lib["default"](ctx))
    # Feed input data
    module.set_input(tfmodel.inputs, tvm.nd.array(image_data))

    if MEASURE_PERF:
        print("Evaluate graph runtime inference cost on VSI NPU")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
        # Measure in millisecond.
        prof_res = np.array(ftimer().results) * 1000
        print("VSI NPU runtime inference time (std dev): %.2f ms (%.2f ms)"
              % (np.mean(prof_res), np.std(prof_res)))
    module.run()
    tvm_output = module.get_output(0).asnumpy()

    return tvm_output


def get_ref_result(shape, model_name, image_data):

    m = SUPPORTED_MODELS[model_name]
    inputs = m.inputs
    DTYPE = "uint8" if m.is_quant else "float32"

    model = get_tflite_model(model_name)
    mod, params = tvm.relay.frontend.from_tflite(
        model, shape_dict={inputs: shape}, dtype_dict={inputs: DTYPE}
    )
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3,
                                   disabled_pass=["AlterOpLayout"]):
        lib = tvm.relay.build(mod, target, params=params)

    ctx = tvm.cpu()
    cpu_mod = graph_runtime.GraphModule(lib["default"](ctx))
    cpu_mod.set_input(inputs, tvm.nd.array(image_data))

    if MEASURE_PERF:
        print("Evaluate graph runtime inference cost on CPU")
        ftimer = cpu_mod.module.time_evaluator("run", ctx, number=1, repeat=1)
        # Measure in millisecond.
        prof_res = np.array(ftimer().results) * 1000
        print("CPU runtime inference time (std dev): %.2f ms (%.2f ms)"
              % (np.mean(prof_res), np.std(prof_res)))

    cpu_mod.run()
    ref_out = cpu_mod.get_output(0).asnumpy()
    return ref_out


def verify_tvm_result(ref_output, shape, model_name, image_data):

    m = SUPPORTED_MODELS[model_name]
    lib_path = compile_tflite_model(shape, model_name, VERBOSE)
    tvm_output = inference_remotely(m, lib_path, image_data)

    if m.name.startswith('deeplabv3') and m.is_quant:
        ref_output = ref_output.reshape(shape[1:3])
        tvm_output = tvm_output.reshape(shape[1:3])

        pix_acc = pixel_accuracy(ref_output, tvm_output)
        print("pixel accuracy:", pix_acc)
        m_acc = mean_accuracy(ref_output, tvm_output)
        print("mean accuracy:", m_acc)
        IoU = mean_IU(ref_output, tvm_output)
        print("mean IU:", IoU)
        freq_weighted_IU = frequency_weighted_IU(ref_output, tvm_output)
        print("frequency weighted IU:", freq_weighted_IU)

    elif 'deeplabv3' in m.name or "ssd" in m.name:
        # compare deeplabv3 float32 output
        np.testing.assert_allclose(ref_output, tvm_output,
                                   rtol=1e-4, atol=1e-4, verbose=True)
    else:  # label index comparison
        ref_idx = np.argmax(np.squeeze(ref_output))
        out_idx = np.argmax(np.squeeze(tvm_output))

        print(f'Expect predict id: {ref_idx}, got {out_idx}')
        assert ref_idx == out_idx


parser = argparse.ArgumentParser(description='VSI-NPU test script for tflite models.')
parser.add_argument('-i', '--ip', type=str, required=True,
                    help='ip address for remote target board')
parser.add_argument('-p', '--port', type=int, default=9090,
                    help='port number for remote target board')
parser.add_argument('-m', '--models', nargs='*', default=SUPPORTED_MODELS,
                    help='models list to test')
parser.add_argument('--perf', action='store_true',
                    help='benchmark performance')
parser.add_argument('--verbose', action='store_true',
                    help='print more logs')

args = parser.parse_args()

RPC_HOST = args.ip
RPC_PORT = args.port
MEASURE_PERF = args.perf
VERBOSE = args.verbose

init_supported_models()
models_to_run = {}

for m in args.models:
    if m not in SUPPORTED_MODELS.keys():
        print("{} is not supported!".format(m))
        print("Supported models: {}".format(list(SUPPORTED_MODELS.keys())))
        sys.exit(1)
    else:
        models_to_run[m] = SUPPORTED_MODELS[m]


print(f"\nTesting {len(models_to_run)} model(s): {list(models_to_run.keys())}")

pass_cases = 0
failed_list = []
for model_name, m in models_to_run.items():
    print("\nTesting {0: <50}".format(model_name.upper()))

    is_quant = m.is_quant
    input_size = m.input_size

    shape = (1, input_size, input_size, 3)

    image_data = get_img_data(shape[1:3], is_quant)
    ref_output = get_ref_result(shape, model_name, image_data)

    try:
        verify_tvm_result(ref_output, shape, model_name, image_data)
    except Exception as err:
        print("Exception", err)
        print(model_name, ": FAIL")
        failed_list.append(model_name)
    else:
        print(model_name, ": PASS")
        pass_cases += 1

print("\n\nTest", len(models_to_run), "cases: ", pass_cases, "Passed")
if len(failed_list) > 0:
    print("Failed list is:", failed_list)
