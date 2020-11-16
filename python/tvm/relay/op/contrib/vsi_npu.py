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
# pylint: disable=invalid-name, unused-argument
"""vsiNPU library supported operators.
There are two ways to registering a function for an op to indicate if it is
supported by vsi_NPU.

- The first and simplest way is to use the helper so that
users only need to provide the operator name and a boolean value to indicate if
it is supported. For example:

    .. code-block:: python

      add = _register_external_op_helper("add")
      add = _register_external_op_helper("add", True)
      add = _register_external_op_helper("add", False)

- The other way is to implement the function by themselves to
check the attributes of the op and decide if it should be offloaded to vsiNPU.
"""
import tvm.ir
from ...dataflow_pattern import wildcard, is_op, is_constant
from .register import register_pattern_table
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import transform


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by vsiNPU.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by vsiNPU.
    """

    @tvm.ir.register_op_attr(op_name, "target.vsi_npu")
    def _func_wrapper(attrs, args):
        return supported

    return _func_wrapper


_register_external_op_helper("add")
_register_external_op_helper("nn.batch_flatten")
#_register_external_op_helper("nn.bias_add")
#_register_external_op_helper("nn.dense")
_register_external_op_helper("nn.relu")
_register_external_op_helper("nn.softmax")
_register_external_op_helper("nn.global_avg_pool2d")
_register_external_op_helper("nn.global_max_pool2d")
_register_external_op_helper("nn.batch_norm")


@register_pattern_table("vsi_npu")
def vsi_npu_pattern_table():
    """Get the VSI NPU pattern table."""

    def conv_pattern():
        """Create a convolution pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("nn.pad")(wildcard()) | wildcard()
        pattern = is_op("nn.conv2d")(pattern, is_constant())
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        return pattern

    def dense_pattern():
        """Create a dense (fully-connected) pattern.

        Returns
        -------
        pattern : dataflow_pattern.AltPattern
            Denotes the convolution pattern.
        """
        pattern = is_op("nn.dense")(wildcard(), is_constant())
        pattern = pattern.optional(lambda x: is_op("nn.bias_add")(x, is_constant()))
        return pattern

    vsi_npu_patterns = [
            ("vsi_npu.dense", dense_pattern()),
            ("vsi_npu.conv2d", conv_pattern()),
            ]
    return vsi_npu_patterns

def partition_for_vsi_npu(mod, params=None):
    """Partition the graph greedily offloading supported
    operators to VSI NPU.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.

    Returns
    -------
    ret : annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.MergeComposite(vsi_npu_pattern_table()),
            transform.AnnotateTarget("vsi_npu"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    return seq(mod)
