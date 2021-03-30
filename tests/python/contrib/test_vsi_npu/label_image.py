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

import tvm
from tvm.contrib import graph_runtime
import argparse
from PIL import Image
import numpy as np

def get_img_data(image_path, shape, dtype):
    resized_image = Image.open(image_path).resize(shape)
    image_data = np.asarray(resized_image).astype(dtype)
    # Add a dimension to the image so that we have NHWC format layout
    image_data = np.expand_dims(image_data, axis=0)

    if dtype == "float32":
        # Preprocess image as described here:
        # https://github.com/tensorflow/models/blob/edb6ed22a801665946c63d650ab9a0b23d98e1b1/research/slim/preprocessing/inception_preprocessing.py#L243

        image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
        image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
        image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1

    return image_data


def load_labels(label_file):
    # List of 1001 classes
    with open(label_file) as f:
        labels = f.readlines()
    return labels


parser = argparse.ArgumentParser(description='Image classification.')
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", required=True,
                     help="image to be processed")
parser.add_argument("-m", "--model", required=True,
                     help="model to be executed")
parser.add_argument("-l", "--labels", required=True,
                     help="name of file containing labels")
parser.add_argument("-t", "--input_tensor", default="input",
                     help="Input tensor for the model")
parser.add_argument("-s", "--input_size", type=int, default=224,
                     help="Input tensor size for the model")
parser.add_argument("-d", "--data_type", default="float32", choices=['float32', 'uint8'],
                     help="Input data type for the model")
args = parser.parse_args()
print(args)


ctx = tvm.cpu(0)
lib = tvm.runtime.load_module(args.model)
m = graph_runtime.GraphModule(lib["default"](ctx))
# set inputs
data = get_img_data(args.image, (args.input_size, args.input_size), args.data_type)
m.set_input(args.input_tensor, data)
# execute
m.run()
# get outputs
tvm_output = m.get_output(0)

labels = load_labels(args.labels)
top1 = np.argmax(tvm_output.asnumpy()[0])
print("TVM prediction top-1:", top1, labels[top1])
