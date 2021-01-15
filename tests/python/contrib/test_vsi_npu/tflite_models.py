import os
import sys
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

import tvm
from tvm import relay, transform
from tvm import rpc
from tvm.contrib import graph_runtime
from tvm import te
from tvm.contrib import graph_runtime as runtime
from tvm.contrib import util
from tvm.relay.op.contrib import vsi_npu
from tvm.contrib.download import download_testdata

RPC_HOST = "10.193.20.8"
RPC_PORT = 9090

class TFModel:
    def __init__(self, name, where, is_quant, formats='tgz'):

        # expect the name looks like "mobilenet_v1_0.25_128"
        fields = name.split("_")

        try:
            size = int(fields[-1])
        except Exception:
            size = 224

        if is_quant:
            name = "{}_quant".format(name)

        self.name = name
        self.url = "{}/{}.{}".format(where, name, formats)
        self.is_quant = is_quant
        self.input_size = size
        self.formats = formats
        self.inputs = 'input'

def extract(path):
    import tarfile

    if path.endswith("tgz") or path.endswith("gz"):
        dir_path = os.path.dirname(path)
        tar = tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close()
    elif path.endswith("tflite"):
        pass
    else:
        raise RuntimeError("Could not decompress the file: " + path)

def get_tflite_model(model):
    # Download model tar file and extract it to get tflite
    model_path = download_testdata(model.url,
                                   model.name + '.' + model.formats,
                                   module=["tf", "official"])

    model_dir = os.path.dirname(model_path)
    extract(model_path)

    # Now we can open tflite model
    tflite_model_file = os.path.join(model_dir, model.name + ".tflite")
    tflite_model_buf = open(tflite_model_file, "rb").read()

    # Get TFLite model from buffer
    try:
        import tflite
        model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model
        model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    return model

def compile_tflite_model(shape, tfmodel):
    DTYPE = "uint8" if tfmodel.is_quant else "float32"

    model = get_tflite_model(tfmodel)

    # Parse TFLite model and convert it to a Relay module
    mod, params = relay.frontend.from_tflite(
        model, shape_dict={tfmodel.inputs: shape}, dtype_dict={tfmodel.inputs: DTYPE}
    )
    lib_path = "./model.so"

    kwargs = {}
    kwargs["cc"] = "aarch64-linux-gnu-gcc"
    target = "llvm  -mtriple=aarch64-linux-gnu"
    with transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        mod = vsi_npu.partition_for_vsi_npu(mod, params)
        print(mod.astext(show_meta_data=False))
        lib = relay.build(mod, target, params=params)
        lib.export_library(lib_path, fcompile=False, **kwargs)

    return lib_path

def inference_remotely(tfmodel, shape, image_data, measure_perf=False):
    lib_path = compile_tflite_model(shape, tfmodel)

    remote = rpc.connect(RPC_HOST, RPC_PORT)
    remote.upload(lib_path)
    lib = remote.load_module(os.path.basename(lib_path))
    ctx = remote.cpu()

    # Create a runtime executor module
    module = graph_runtime.GraphModule(lib["default"](ctx))
    # Feed input data
    module.set_input(tfmodel.inputs, tvm.nd.array(image_data))

    module.run()
    tvm_output = module.get_output(0).asnumpy()

    if measure_perf:
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=50)
        # Measure in millisecond.
        prof_res = np.array(ftimer().results) * 1000

        return tvm_output, prof_res
    else:
        return tvm_output, None


def get_ref_result(shape, tfmodel, image_data, measure_perf=False):
    DTYPE = "uint8" if tfmodel.is_quant else "float32"

    model = get_tflite_model(tfmodel)
    mod, params = relay.frontend.from_tflite(
        model, shape_dict={tfmodel.inputs: shape}, dtype_dict={tfmodel.inputs: DTYPE}
    )
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3,
                                   disabled_pass=["AlterOpLayout"]):
        lib = relay.build(mod, target, params=params)

    ctx = tvm.cpu()
    cpu_mod = graph_runtime.GraphModule(lib["default"](ctx))
    cpu_mod.set_input(tfmodel.inputs, tvm.nd.array(image_data))

    cpu_mod.run()
    ref_out = cpu_mod.get_output(0).asnumpy()

    if measure_perf:
        ftimer = cpu_mod.module.time_evaluator("run", ctx, number=1, repeat=10)
        # Measure in millisecond.
        prof_res = np.array(ftimer().results) * 1000

        return ref_out, prof_res
    else:
        return ref_out, None


def get_img_data(image_url, shape, is_quant):
    image_path = download_testdata(image_url, "cat.png", module="data")
    resized_image = Image.open(image_path).resize(shape)

    DTYPE = "uint8" if is_quant else "float32"

    image_data = np.asarray(resized_image).astype(DTYPE)

    # Add a dimension to the image so that we have NHWC format layout
    image_data = np.expand_dims(image_data, axis=0)

    if not is_quant:
        # Preprocess image as described here:
        # https://github.com/tensorflow/models/blob/edb6ed22a801665946c63d650ab9a0b23d98e1b1/research/slim/preprocessing/inception_preprocessing.py#L243

        image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
        image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
        image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1

    return image_data
