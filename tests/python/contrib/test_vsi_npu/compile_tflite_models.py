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
import argparse
from tflite_models import *

VERBOSE = False

parser = argparse.ArgumentParser(description='VSI-NPU compile script for tflite models.')
parser.add_argument('-m', '--models', nargs='*', default=SUPPORTED_MODELS,
                    help='models list to test')
parser.add_argument('dir', type=str,
                    help='destination directory')
parser.add_argument('--verbos', action='store_true',
                    help='print more logs')

args = parser.parse_args()

init_supported_models()
models_to_run = {}

for m in args.models:
    if m not in SUPPORTED_MODELS.keys():
        print("{} is not supported!".format(m))
        print("Supported models: {}".format(list(SUPPORTED_MODELS.keys())))
        sys.exit(1)
    else:
        models_to_run[m] = SUPPORTED_MODELS[m]


print(f"\nCompiling {len(models_to_run)} model(s): {list(models_to_run.keys())}")

for model_name, m in models_to_run.items():
    print("\nCompiling {0: <50}".format(model_name.upper()))

    is_quant = m.is_quant
    input_size = m.input_size

    shape = (1, input_size, input_size, 3)
    target_models = os.path.join(args.dir, model_name + '.so')
    lib_path = compile_tflite_model(shape, model_name, args.verbos, target_models)
