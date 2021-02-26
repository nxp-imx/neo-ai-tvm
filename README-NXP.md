# Overview
[Apache TVM](https://github.com/apache/tvm) is a compiler stack for deep learning systems.

This repository introduces the codegen of NXP NPU support into the system. With this backend, the deep learning model can be compiled and accelerated with the NPU offloading library at runtime.

# How to build and install

This backend is using the [LLVM](https://llvm.org/) to cross compile the generted source as a deloyable dynamic library for device.
Please follow the [LLVM Doc](https://llvm.org/docs/) to install LLVM on the host. If installed successfully, llvm-config should be found under /usr/bin.

## Build tvm

Conceptually, TVM can be splitted into two parts.

`tvm build stack`: compiles the deep learing model at host
`tvm runtime`: loads and interprets the model at device

```bash
git clone --recursive {this git} tvm
cd tvm
mkdir build
cp cmake/config.cmake build
cd build
sed -i 's/USE_LLVM\ OFF/USE_LLVM\ \/usr\/bin\/llvm-config/' config.cmake  # turn on LLVM support
cmake ..
make tvm -j4  # make tvm build stack
```

To make the runtime for target device
```bash
setup the cross-compiling environment for the target device
make runtime -j4
```

Please refer [TVM Doc](https://tvm.apache.org/docs/) for more details.


## Install tvm build stack on host

```bash
EXPORT TVM_HOME=/workspace/tvm   # assume this tvm repo is put under /workspace
EXPORT PYTHONPATH=$TVM_HOME/python
python3 -c "import tvm"  # test if the tvm is installed
```

## Deploy tvm runtime to device



# Getting Started

## Compile a pre-trained model

```bash

import tvm
from tvm import relay, transform
from tvm import rpc
from tvm.contrib import graph_runtime
from tvm import te
from tvm.contrib import graph_runtime as runtime
from tvm.contrib import util
from tvm.relay.op.contrib import vsi_npu


# Load tflite model into buffer
tflite_model_file="mobilenet_v1_1.0_224_quant.tflite"
tflite_model_buf = open(tflite_model_file, "rb").read()

# Get tflite model from buffer
try:
    import tflite
    model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model
    model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

# Parse the tflite model
shape = shape = (1, 224, 224, 3)
inputs = 'input'
mod, params = relay.frontend.from_tflite(
    model, shape_dict={inputs: shape}, dtype_dict={inputs: "uint8"}
)

# Compile the model
lib_path = "./model.so"

kwargs = {}
kwargs["cc"] = "aarch64-linux-gnu-gcc"
target = "llvm  -mtriple=aarch64-linux-gnu"
with transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
    mod = vsi_npu.partition_for_vsi_npu(mod, params)
    lib = relay.build(mod, target, params=params)
    lib.export_library(lib_path, fcompile=False, **kwargs)

```

The generated model is saved as model.so

## Running the compiled model on device

See more examples at tvm/tests/python/contrib/test_vsi_npu/ with rpc verification

# Supported TFlite models

|model|float32|int8|input_size|
|:----|:----|:----|:----|
|mobilenet_v1_0.25_128|[mobilenet_v1_0.25_128](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128.tgz)|[mobilenet_v1_0.25_128_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128_quant.tgz)|128|
|mobilenet_v1_0.25_224|[mobilenet_v1_0.25_224](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224.tgz)|[mobilenet_v1_0.25_224_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_224_quant.tgz)|224|
|mobilenet_v1_0.5_128|[mobilenet_v1_0.5_128](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_128.tgz)|[mobilenet_v1_0.5_128_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_128_quant.tgz)|128|
|mobilenet_v1_0.5_224|[mobilenet_v1_0.5_224](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_224.tgz)|[mobilenet_v1_0.5_224_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.5_224_quant.tgz)|224|
|mobilenet_v1_0.75_128|[mobilenet_v1_0.75_128](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_128.tgz)|[mobilenet_v1_0.75_128_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_128_quant.tgz)|128|
|mobilenet_v1_0.75_224|[mobilenet_v1_0.75_224](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_224.tgz)|[mobilenet_v1_0.75_224_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.75_224_quant.tgz)|224|
|mobilenet_v1_1.0_128|[mobilenet_v1_1.0_128](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_128.tgz)|[mobilenet_v1_1.0_128_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_128_quant.tgz)|128|
|mobilenet_v1_1.0_224|[mobilenet_v1_1.0_224](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz)|[mobilenet_v1_1.0_224_quant](http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz)|224|
|mobilenet_v2_1.0_224|[mobilenet_v2_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz)|[mobilenet_v2_1.0_224_quant](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz)|224|
|inception_v1|NA|[inception_v1_224_quant](https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_224_quant_20181026.tgz)|224|
|inception_v2|NA|[inception_v2_224_quant](https://storage.googleapis.com/download.tensorflow.org/models/inception_v2_224_quant_20181026.tgz)|224|
|inception_v3|[inception_v3](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz)|[inception_v3_quant](https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/inception_v3_quant.tgz)|299|
|inception_v4|[inception_v4](https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz)|[inception_v4_299_quant](https://storage.googleapis.com/download.tensorflow.org/models/inception_v4_299_quant_20181026.tgz)|299|
|deeplab_v3_257_mv_gpu|[deeplab_v3_256_mv_gpu](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite)|NA|257|
|deeplab_v3_mnv2_pascal|NA|[deeplab_v3_mnv2_pascal](https://github.com/google-coral/edgetpu/raw/master/test_data/deeplabv3_mnv2_pascal_quant.tflite)|513|
|ssdlite_mobiledet|[ssdlite_mobiledet_cpu_320x320_coco](http://download.tensorflow.org/models/object_detection/ssdlite_mobiledet_cpu_320x320_coco_2020_05_19.tgz)"|NA|320|


# License
Licensed under an [Apache-2.0](http://www.apache.org/licenses/LICENSE-2.0) license.
