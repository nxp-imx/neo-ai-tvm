import os
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

RPC_HOST = "10.193.20.6"
RPC_PORT = 9090
TMP_PATH = util.tempdir()
IS_QUANT = False
MEASURE_PERF = False

MODEL_VERSION = "v1_1.0"
MODEL_SIZE = 224

if (IS_QUANT):
    DTYPE = "uint8"
else:
    DTYPE = "float32"

def extract(path):
    import tarfile

    if path.endswith("tgz") or path.endswith("gz"):
        dir_path = os.path.dirname(path)
        tar = tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close()
    else:
        raise RuntimeError("Could not decompress the file: " + path)


def get_tflite_model():
    if (IS_QUANT):
        name_prefix = "mobilenet_%s_%d_quant" % (MODEL_VERSION, MODEL_SIZE)
    else:
        name_prefix = "mobilenet_%s_%d" % (MODEL_VERSION, MODEL_SIZE)

    model_url = "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/" + name_prefix + ".tgz"
    if MODEL_VERSION[0:2] == "v2":
        model_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/"\
                     + name_prefix + ".tgz"

    # Download model tar file and extract it to get mobilenet_v1_1.0_224.tflite
    model_path = download_testdata(model_url, name_prefix + ".tgz", module=["tf", "official"])
    model_dir = os.path.dirname(model_path)
    extract(model_path)

    # Now we can open tflite model
    tflite_model_file = os.path.join(model_dir, name_prefix + ".tflite")
    tflite_model_buf = open(tflite_model_file, "rb").read()

    # Get TFLite model from buffer
    try:
        import tflite
        model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model
        model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)


    # Load label file
    label_file = "labels_mobilenet_quant_v1_224.txt"
    label_file_url = "".join(
        [
            "https://raw.githubusercontent.com/",
            "tensorflow/tensorflow/master/tensorflow/lite/java/demo/",
            "app/src/main/assets/",
            label_file,
        ]
    )
    label_path = download_testdata(label_file_url, label_file, module="data")

    # List of 1001 classes
    with open(label_path) as f:
        labels = f.readlines()

    # TFLite input tensor name, shape and type
    inputs = "input"
    shape = (1, MODEL_SIZE, MODEL_SIZE, 3)
    return inputs, shape, labels, model


def get_img_data(shape):
    image_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    image_path = download_testdata(image_url, "cat.png", module="data")
    resized_image = Image.open(image_path).resize(shape)
    image_data = np.asarray(resized_image).astype(DTYPE)

    # Add a dimension to the image so that we have NHWC format layout
    image_data = np.expand_dims(image_data, axis=0)

    if (IS_QUANT == False):
        # Preprocess image as described here:
        # https://github.com/tensorflow/models/blob/edb6ed22a801665946c63d650ab9a0b23d98e1b1/research/slim/preprocessing/inception_preprocessing.py#L243
        image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
        image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
        image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1
    return image_data

def compile_tflite_model(inputs, shape, model):
    # Parse TFLite model and convert it to a Relay module
    mod, params = relay.frontend.from_tflite(
        model, shape_dict={inputs: shape}, dtype_dict={inputs: DTYPE}
    )
    lib_name = "mobilenet.so"
    lib_path = TMP_PATH.relpath(lib_name)

    kwargs = {}
    kwargs["cc"] = "aarch64-linux-gnu-g++"
    target = "llvm  -mtriple=aarch64-linux-gnu"
    with transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        mod = vsi_npu.partition_for_vsi_npu(mod, params)
        print(mod.astext(show_meta_data=False))
        lib  = relay.build(mod, target, params=params)
        lib.export_library(lib_path, fcompile=False, **kwargs)

    return lib_path

def inference_remotely(lib_path, image_data):
    remote = rpc.connect(RPC_HOST, RPC_PORT)
    remote.upload(lib_path)
    lib = remote.load_module(os.path.basename(lib_path))
    ctx = remote.cpu()

    # Create a runtime executor module
    module = graph_runtime.GraphModule(lib["default"](ctx))
    # Feed input data
    module.set_input(inputs, tvm.nd.array(image_data))

    if MEASURE_PERF:
        print("Evaluate graph runtime inference cost on VSI NPU")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=50)
        # Measure in millisecond.
        prof_res = np.array(ftimer().results) * 1000
        print("VSI NPU runtime inference time (std dev): %.2f ms (%.2f ms)"
                % (np.mean(prof_res), np.std(prof_res)))
    # Run
    module.run()
    # Get output
    tvm_output = module.get_output(0).asnumpy()

    return tvm_output

def get_ref_result(inputs, shape, model, image_data):
    mod, params = relay.frontend.from_tflite(
        model, shape_dict={inputs: shape}, dtype_dict={inputs: DTYPE}
    )
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        lib  = relay.build(mod, target, params=params)

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


inputs, shape, labels, model = get_tflite_model()
image_data = get_img_data(shape[1:3])
lib_path = compile_tflite_model(inputs, shape, model)
tvm_output = inference_remotely(lib_path, image_data)

# Convert result to 1D data
predictions = np.squeeze(tvm_output)

# Get top 1 prediction
prediction = np.argmax(predictions)
EXPECT_RET = 283
if (prediction == EXPECT_RET):
    print("Passed")
else:
    print("Failed")
    print("Expect prediction result: id " + str(EXPECT_RET) + " name: " + labels[EXPECT_RET])

# Convert id to class name and show the result
print("Get prediction result: id " + str(prediction) + " name: " + labels[prediction])


