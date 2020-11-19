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

def extract(path):
    import tarfile

    if path.endswith("tgz") or path.endswith("gz"):
        dir_path = os.path.dirname(path)
        tar = tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close()
    else:
        raise RuntimeError("Could not decompress the file: " + path)


def get_tflite_model(is_quant):
    if (is_quant):
        name_prefix = "mobilenet_v1_1.0_224_quant"
    else:
        name_prefix = "mobilenet_v1_1.0_224"

    model_url = "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/" + name_prefix + ".tgz"

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
    label_file_url = "".join(
        [
            "https://raw.githubusercontent.com/",
            "tensorflow/tensorflow/master/tensorflow/lite/java/demo/",
            "app/src/main/assets/",
            "labels_mobilenet_quant_v1_224.txt",
        ]
    )
    label_file = "labels_mobilenet_quant_v1_224.txt"
    label_path = download_testdata(label_file_url, label_file, module="data")

    # List of 1001 classes
    with open(label_path) as f:
        labels = f.readlines()

    # TFLite input tensor name, shape and type
    inputs = "input"
    shape = (1, 224, 224, 3)
    dtype = "float32"
    return inputs, shape, dtype, labels, model


def get_img_data():
    image_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    image_path = download_testdata(image_url, "cat.png", module="data")
    resized_image = Image.open(image_path).resize((224, 224))
    image_data = np.asarray(resized_image).astype("float32")

    # Add a dimension to the image so that we have NHWC format layout
    image_data = np.expand_dims(image_data, axis=0)

    # Preprocess image as described here:
    # https://github.com/tensorflow/models/blob/edb6ed22a801665946c63d650ab9a0b23d98e1b1/research/slim/preprocessing/inception_preprocessing.py#L243
    image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
    image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
    image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1
    return image_data

def compile_tflite_model(inputs, shape, dtype, model):
    # Parse TFLite model and convert it to a Relay module
    mod, params = relay.frontend.from_tflite(
        model, shape_dict={inputs: shape}, dtype_dict={inputs: dtype}
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
    # Run
    module.run()
    # Get output
    tvm_output = module.get_output(0).asnumpy()

    return tvm_output


inputs, shape, dtype, labels, model = get_tflite_model(is_quant = False)
image_data = get_img_data()
lib_path = compile_tflite_model(inputs, shape, dtype, model)
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


