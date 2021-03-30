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
from PIL import Image
import numpy as np
import argparse

from tvm import relay, transform
from tvm.relay.op.contrib import vsi_npu
from tvm.contrib.download import download_testdata
from tflite_deeplab import *

SUPPORTED_MODELS = {}  # name to TFModel mapping


class TFModel:
    def __init__(self, name, where, is_quant, formats='tgz', suffix=''):

        # expect the name looks like "mobilenet_v1_0.25_128"
        fields = name.split("_")

        try:
            size = int(fields[-1])
        except Exception:
            size = 224

        if is_quant:
            name = "{}_quant".format(name)

        self.name = name
        self.formats = formats
        self.url = "{}/{}{}.{}".format(where, name, suffix, formats)

        self.is_quant = is_quant
        self.input_size = size
        self.inputs = 'input'


def add_supported_model(name, where, is_quant=False, formats='tgz',
                        suffix=''):
    m = TFModel(name, where, is_quant, formats, suffix)
    SUPPORTED_MODELS[m.name] = m

    return m


def init_supported_models():
    QUANT = True
    where = "https://storage.googleapis.com/tfhub-lite-models/tensorflow/lite-model/ssd_mobilenet_v1/1/metadata/1.tflite"
    m = add_supported_model("ssd_mobilenet_v1", "", QUANT, formats="tflite")
    m.url = where
    m.inputs = "normalized_input_image_tensor"
    m.input_size = 300

    where = "http://download.tensorflow.org/models/object_detection"
    m = add_supported_model("ssd_mobilenet_v3", where, formats='tar.gz', suffix="_small_coco_2020_01_14")
    m.input_size = 320
    m.is_quant = True
    m.inputs = 'normalized_input_image_tensor'

    where = "http://10.192.208.75/images/gf/tvm"
    m = add_supported_model("mobilenet_ssd_v1_trimmed_converted", where, formats='tflite')
    m.input_size = 300
    m.inputs = 'Preprocessor/sub'

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

    where = "https://storage.googleapis.com/download.tensorflow.org/"
    where += "models/tflite/gpu"
    m = add_supported_model("deeplabv3_257_mv_gpu", where, formats='tflite')
    m.input_size = 257
    m.inputs = "sub_7"

    where = "https://github.com/google-coral/edgetpu/raw/master/test_data"
    m = add_supported_model("deeplabv3_mnv2_pascal", where, QUANT,
                            formats='tflite')
    m.input_size = 513
    m.inputs = "MobilenetV2/MobilenetV2/input"

    where = "http://download.tensorflow.org/models/object_detection"
    m = add_supported_model("ssdlite_mobiledet_cpu_320x320_coco", where,
                            formats='tar.gz', suffix="_2020_05_19")
    m.input_size = 320
    m.inputs = 'normalized_input_image_tensor'

    return SUPPORTED_MODELS


def extract(path, model_name):
    import tarfile
    import zipfile

    dir_path = os.path.dirname(path)
    tmp_dir = os.path.join(dir_path, model_name)
    os.makedirs(tmp_dir, exist_ok=True)
    if path.endswith("tgz") or path.endswith("gz") or path.endswith("tar.gz"):
        tar = tarfile.open(path)
        tar.extractall(path=tmp_dir)
        tar.close()
    elif path.endswith("zip"):
        zf = zipfile.ZipFile(path, 'r')
        for f in zf.namelist():
            zf.extract(f, tmp_dir)
    else:
        raise RuntimeError("Could not decompress the file: " + path)

    for dir_path, subpaths, files in os.walk(tmp_dir):
        for f in files:
            if f.endswith("tflite"):
                return os.path.join(dir_path, f)

    raise RuntimeError("Could not find tflite model file.")

def get_tflite_model(model_name):

    m = SUPPORTED_MODELS[model_name]

    # Download model tar file and extract it to get tflite
    model_path = download_testdata(m.url,
                                   model_name + "." + m.formats,
                                   module=["tf", "official"])

    model_dir = os.path.dirname(model_path)
    if m.formats in ['tgz', 'zip', 'tar.gz']:
        model_name = extract(model_path, model_name)
    else:
        model_name = model_name + ".tflite"
    # Now we can open tflite model
    tflite_model_file = os.path.join(model_dir, model_name)
    tflite_model_buf = open(tflite_model_file, "rb").read()

    # Get TFLite model from buffer
    try:
        import tflite
        model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model
        model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    return model


def get_img_data(shape, is_quant):
    image_url =\
        "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"

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


def compile_tflite_model(shape, model_name, verbose=False, lib_path="./model.so"):
    m = SUPPORTED_MODELS[model_name]

    DTYPE = "uint8" if m.is_quant else "float32"

    model = get_tflite_model(model_name)
    # Parse TFLite model and convert it to a Relay module
    mod, params = relay.frontend.from_tflite(
        model, shape_dict={m.inputs: shape}, dtype_dict={m.inputs: DTYPE}
    )

    kwargs = {}
    kwargs["cc"] = "aarch64-linux-gnu-gcc"
    target = "llvm  -mtriple=aarch64-linux-gnu"
    with transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        mod = vsi_npu.partition_for_vsi_npu(mod, params)
        if verbose:
            print(mod.astext(show_meta_data=False))
        lib = relay.build(mod, target, params=params)
        lib.export_library(lib_path, fcompile=False, **kwargs)

    return lib_path
