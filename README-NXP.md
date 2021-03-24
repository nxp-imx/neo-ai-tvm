# Overview
[Apache TVM](https://github.com/apache/tvm) is a compiler stack for deep learning systems.

This repository introduces the codegen of NXP NPU support into the system. With this backend, the deep learning model can be compiled and accelerated with the NPU offloading library at runtime.

# How to build and install

This backend is using the [LLVM](https://llvm.org/) to cross compile the generted source as a deloyable dynamic library for device.
Please follow the [LLVM Doc](https://llvm.org/docs/) to install LLVM on the host. If installed successfully, llvm-config should be found under /usr/bin.

## Build tvm

Conceptually, TVM can be splitted into two parts.

1. `tvm build stack`: compiles the deep learing model at host
2. `tvm runtime`: loads and interprets the model at device

#Build `tvm build stack` for host
```bash
export TOP_DIR=`pwd`
git clone --recursive {this git} tvm-host
cd tvm-host
mkdir build
cp cmake/config.cmake build
cd build
sed -i 's/USE_LLVM\ OFF/USE_LLVM\ \/usr\/bin\/llvm-config/' config.cmake  # turn on LLVM support
cmake ..
make tvm -j4  # make tvm build stack
```

#Build `tvm runtime` for target device

To build the runtime for target device, the cross-compiling environment should be installed first

Build the `tim-vx` library
```bash
cd ${TOP_DIR}
source {cross tool-chain directory}/environment-setup-aarch64-poky-linux
git clone {vsi-tim-vx repository}
cd vsi-tim-vx
cp -a include/tim {cross tool-chain directory}/sysroots/aarch64-poky-linux/usr/include/
mkdir build
cd build
cmake ..
make
```

Build the `tvm runtime`
```bash
cd ${TOP_DIR}
source {cross tool-chain directory}/environment-setup-aarch64-poky-linux
git clone --recursive {this git} tvm-runtime
cd tvm-runtime
mkdir build
cp cmake/config.cmake build
cd build
sed -i 's/USE_VSI_NPU_RUNTIME\ OFF/USE_VSI_NPU_RUNTIME\ ON/' config.cmake  # turn on npu runtime
make runtime -j4
```

Please refer [TVM Doc](https://tvm.apache.org/docs/) for more details.


## Install tvm build stack on host

```bash
EXPORT TVM_HOME=${TOP_DIR}/tvm-host
EXPORT PYTHONPATH=$TVM_HOME/python
python3 -c "import tvm"  # test if the tvm is installed
```

## Deploy tvm runtime to device
Copy `tim-vx` library,`tvm runtime`library and python package to device
```bash
scp -r ${TOP_DIR}/tvm-runtime/python/tvm root@{device ip}:/usr/lib/python3.7/site-packages/
scp ${TOP_DIR}/vsi-tim-vx/build/src/tim/vx/libtim-vx.so root@{device ip}:/usr/lib/
scp ${TOP_DIR}/tvm-runtime/build/libtvm_runtime.so root@{device ip}:/usr/lib/
```


# Getting Started

Run examples at tvm/tests/python/contrib/test_vsi_npu/ with rpc verification

# Bootup rpc server on device
```bash
python3 -m tvm.exec.rpc_server --host 0.0.0.0 --port=9090
```

# Run the test script on host
```bash
cd ${TOP_DIR}/tvm-host
python3 tests/python/contrib/test_vsi_npu/test_tflite_models.py -i {device ip}
```


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
