import os
import sys
from PIL import Image
import numpy as np

from tflite_models import *

MEASURE_PERF = True
VERBOSE = False
SUPPORTED_MODELS = {}  # name to TFModel mapping


def add_supported_model(name, where, is_quant=False, suffix=''):
    m = TFModel(name, where, is_quant, formats='tgz')
    SUPPORTED_MODELS[m.name] = m

    if suffix is not None:
        m.url = "{}/{}{}.tgz".format(where, m.name, suffix)

    return m


def init_supported_models():
    QUANT = True
    where = "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02"
    add_supported_model("mobilenet_v1_0.25_128", where, QUANT)
    add_supported_model("mobilenet_v1_0.25_224", where, QUANT)
    add_supported_model("mobilenet_v1_0.5_128", where, QUANT)
    add_supported_model("mobilenet_v1_0.5_224", where, QUANT)
    add_supported_model("mobilenet_v1_0.75_128", where, QUANT)
    add_supported_model("mobilenet_v1_0.75_224", where, QUANT)
    add_supported_model("mobilenet_v1_1.0_128", where, QUANT)
    add_supported_model("mobilenet_v1_1.0_224", where, QUANT)

    add_supported_model("mobilenet_v1_0.25_128", where)
    add_supported_model("mobilenet_v1_0.25_224", where)
    add_supported_model("mobilenet_v1_0.5_128", where)
    add_supported_model("mobilenet_v1_0.5_224", where)
    add_supported_model("mobilenet_v1_0.75_128", where)
    add_supported_model("mobilenet_v1_0.75_224", where)
    add_supported_model("mobilenet_v1_1.0_128", where)
    add_supported_model("mobilenet_v1_1.0_224", where)

    where = "https://storage.googleapis.com/"
    where += "download.tensorflow.org/models/tflite_11_05_08"
    add_supported_model("mobilenet_v2_1.0_224", where, QUANT)
    add_supported_model("mobilenet_v2_1.0_224", where)
    # the input size for inception_v3 model from tflite model zoo is 299
    # https://discuss.tvm.apache.org/t/possible-bug-relay-internal-invariant-was-violdated/2105/8
    m = add_supported_model("inception_v3", where, QUANT)
    m.input_size = 299

    where = "https://storage.googleapis.com/download.tensorflow.org/models"
    add_supported_model("inception_v1_224", where, QUANT, suffix="_20181026")
    add_supported_model("inception_v2_224", where, QUANT, suffix="_20181026")
    add_supported_model("inception_v4_299", where, QUANT, suffix="_20181026")

    where = "https://storage.googleapis.com/download.tensorflow.org/"
    where += "models/tflite/model_zoo/upload_20180427"
    m = add_supported_model("inception_v3", where, suffix="_2018_04_27")
    m.input_size = 299
    m = add_supported_model("inception_v4", where, suffix="_2018_04_27")
    m.input_size = 299

    return SUPPORTED_MODELS


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


def print_help():
    print("Usage: \n",
          '  -r[--rpc]    IP:Port\n',
          '  -m[--models] "model_list ...", all models by default\n',
          '  [--perf] benchmark performance\n',
          '  [--verbose] print more logs\n')


def parse_command_args():
    import getopt
    import re

    global RPC_HOST, RPC_PORT, MEASURE_PERF, VERBOSE
    models_input = []
    try:
        opts, args = getopt.getopt(
                         sys.argv[1:], "hr:m:v",
                         ["help", "rpc=", "models=", "perf", "verbose"])
    except getopt.GetoptError:
        print_help()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help()
            sys.exit()
        elif opt in ("-r", "--rpc"):
            rpc_ = arg.split(":")
            if (len(rpc_) != 2):
                raise Exception("invalid rcp address: %s" % arg)
            RPC_HOST = rpc_[0].strip()
            RPC_PORT = int(rpc_[1].strip())
        elif opt in ("-m", "--models"):
            models_input = ' '.join(re.split(' |,|;', arg.strip())).split()
        elif opt in ("--perf"):
            MEASURE_PERF = True
        elif opt in ("-v", "--verbose"):
            VERBOSE = True
    return models_input


args = parse_command_args()
models_to_run = {}
init_supported_models()
image_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"

for m in args:
    if m not in SUPPORTED_MODELS.keys():
        print("{} is not supported!".format(m))
        print("Supported models: {}".format(list(SUPPORTED_MODELS.keys())))
        sys.exit(1)
    else:
        models_to_run[m] = SUPPORTED_MODELS[m]

if len(models_to_run) == 0:
    models_to_run = SUPPORTED_MODELS

pass_cases = 0
failed_list = []
for model_name, m in models_to_run.items():
    print("\nTesting {0: <50}".format(model_name.upper()))

    is_quant = m.is_quant
    input_size = m.input_size

    shape = (1, input_size, input_size, 3)

    image_data = get_img_data(image_url, shape[1:3], is_quant)
    ref_output, prof_res = get_ref_result(shape, m, image_data, MEASURE_PERF)
    if MEASURE_PERF:
        print("CPU runtime inference time (std dev): %.2f ms (%.2f ms)"
              % (np.mean(prof_res), np.std(prof_res)))

    # Get top 1 prediction
    idx = np.argmax(np.squeeze(ref_output))

    try:
        tvm_output, prof_res = inference_remotely(m, shape, image_data, MEASURE_PERF)
        if MEASURE_PERF:
            print("VSI NPU runtime inference time (std dev): %.2f ms (%.2f ms)"
                  % (np.mean(prof_res), np.std(prof_res)))
        out_idx = np.argmax(np.squeeze(tvm_output))
    except Exception as err:
        print("\nExpect predict id {}, got {}".format(idx, err))
        print(model_name, ": FAIL")
        failed_list.append(model_name)
    else:
        print("\nExpect predict id {}, got {}".format(idx, out_idx))
        if (idx == out_idx):
            print(model_name, ": PASS")
            pass_cases += 1
        else:
            print(model_name, ": FAIL")
            failed_list.append(model_name)

print("\n\nTest", len(models_to_run), "cases: ", pass_cases, "Passed")
if len(failed_list) > 0:
    print("Failed list is:", failed_list)
