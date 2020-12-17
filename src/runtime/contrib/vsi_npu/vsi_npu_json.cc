/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/vsi_npu/dnnl_json_runtime.cc
 * \brief A simple JSON runtime for VsiNpu.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

#ifdef USE_VSI_NPU_RUNTIME
#include "ovxlibxx/context.h"
#include "ovxlibxx/graph.h"
#include "ovxlibxx/tensor.h"
#include "ovxlibxx/operation.h"
#include "ovxlibxx/operations/fullyconnected.h"
#include "ovxlibxx/operations/activations.h"
#include "ovxlibxx/operations/softmax.h"
#include "ovxlibxx/operations/reshape.h"
#include "ovxlibxx/operations/pool2d.h"
#include "ovxlibxx/operations/conv2d.h"
#include "ovxlibxx/operations/batchnorm.h"
#include "ovxlibxx/operations/add.h"
#include "ovxlibxx/operations/permute.h"
#include "ovxlibxx/operations/clip.h"
#include "ovxlibxx/operations/concat.h"
#include "ovxlibxx/operations/dropout.h"

#include "vsi_utils.h"
#endif

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

#ifdef USE_VSI_NPU_RUNTIME
class VsiNpuJSONRuntime : public JSONRuntimeBase {

 public:
  VsiNpuJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "vsi_npu_json"; }

  void Init(const Array<NDArray>& consts) override {
    // Setup constants entries for weights.
    SetupConstants(consts);

    CHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    BuildEngine();
  }

  void Run() override {
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto nid = input_nodes_[i];
      uint32_t eid = EntryID(nid, 0);
      if (nodes_[nid].GetOpType() == "input") {
	auto vsi_tensor = entry_out_tensor_[eid];

        void* data = data_entry_[eid]->data;
	int data_size = 1;
        for (int j = 0; j < data_entry_[eid]->ndim; j++) {
          data_size *= data_entry_[eid]->shape[j];
        }
        assert(vsi_tensor->CopyDataToTensor(data, data_size));
      }
    }

    assert(graph_->Run());

    for (size_t i = 0; i < outputs_.size(); ++i) {
      uint32_t eid = EntryID(outputs_[i]);
      void* data = data_entry_[eid]->data;

      auto vsi_tensor = entry_out_tensor_[eid];
      vsi_tensor->CopyDataFromTensor(data);
    }
  }
 private:

  void BuildEngine() {
    context_ = vsi::Context::Create();
    graph_ = context_->CreateGraph();

    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        CHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
	LOG(INFO) << "Build op: " << op_name;
        if ("nn.batch_flatten" == op_name or "reshape" == op_name) {
	  Reshape(nid);
        } else if ("nn.dense" == op_name or "qnn.dense" == op_name) {
	  Dense(nid);
        } else if ("nn.relu" == op_name) {
          Relu(nid);
        } else if ("nn.softmax" == op_name || "qnn.softmax" == op_name) {
          Softmax(nid);
        } else if ("nn.batch_norm" == op_name) {
          BatchNorm(nid);
        } else if ("nn.conv2d" == op_name || "qnn.conv2d" == op_name) {
          Conv2D(nid);
        } else if (("nn.global_avg_pool2d" == op_name) || ("nn.global_max_pool2d" == op_name)) {
          GlobalPool2d(nid);
        } else if ("nn.max_pool2d" == op_name || "nn.avg_pool2d" == op_name || "qnn.avg_pool2d" == op_name) {
          Pool2d(nid);
        } else if ("add" == op_name || "qnn.add" == op_name) {
          Add(nid);
        } else if ("clip" == op_name) {
          Clip(nid);
        } else if ("layout_transform" == op_name) {
          Permute(nid);
        } else if ("nn.dropout" == op_name) {
          Dropout(nid);
        } else if ("concatenate" == op_name || "qnn.concatenate" == op_name) {
          Concat(nid);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
    assert(graph_->Compile());
    LOG(INFO) << "Build graph successfully" << std::endl;
  }

  void Reshape(const size_t& nid) {
    auto node = nodes_[nid];
    JSONGraphNodeEntry out_entry(nid, 0);
    std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();

    CHECK(inputs.size() == 1U) << "Flatten layer requires 1 inputs.";

    auto vsi_input = MakeVSITensorFromJSONEntry(inputs[0]);
    auto vsi_output = MakeVSITensorFromJSONEntry(out_entry, vsi_input->GetQuantization());

    std::vector<uint32_t> output_shape = vsi_output->GetShape();

    auto reshape = graph_->CreateOperation<vsi::Reshape>(output_shape);
    (*reshape).BindInput(vsi_input).BindOutput(vsi_output);
    ops_.push_back(reshape);
  }

  void Dense(const size_t& nid) {
    auto node = nodes_[nid];
    // Collect inputs and outputs, handling both nn.dense and qnn.dense cases.
    std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();
    size_t num_inputs = inputs.size();
    JSONGraphNodeEntry out_entry(nid, 0);
    
    std::vector<std::shared_ptr<vsi::Tensor>> vsi_inputs;
    std::vector<std::shared_ptr<vsi::Tensor>> vsi_outputs;

    bool has_bias;
    if (node.GetOpName() == "qnn.dense") {
      //qnn.densn
      CHECK(num_inputs >= 10U && num_inputs <= 11U)
          << "Quantized convolution requires 11 inputs with a bias, 9 inputs without.";
      has_bias = num_inputs == 11;
      vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[0], &inputs[4], &inputs[2]));
      vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[1], &inputs[5], &inputs[3]));
      if (has_bias) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[6], &inputs[9], &inputs[10]));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry, &inputs[6 + has_bias], &inputs[7 + has_bias]));
    } else {
      CHECK(num_inputs >= 2U && num_inputs <= 3U)
          << "Fully connected (dense) layer requires 3 inputs with a bias, 2 inputs without.";
      for (const auto& i : inputs) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(i));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry));
    }

    auto weight_tensor = vsi_inputs[1];
    auto fc = graph_->CreateOperation<vsi::FullyConnected>(1, weight_tensor->GetShape()[1]);
    (*fc).BindInputs(vsi_inputs).BindOutputs(vsi_outputs);
    ops_.push_back(fc);
  }

  void Relu(const size_t& nid) {

    auto node = nodes_[nid];
    //JSONGraphNodeEntry input
    auto data_entry = node.GetInputs()[0];

    JSONGraphNodeEntry out_entry(nid, 0);

    std::shared_ptr<vsi::Tensor> vsi_input;
    std::shared_ptr<vsi::Tensor> vsi_output;

    vsi_input = MakeVSITensorFromJSONEntry(data_entry);

    vsi_output = MakeVSITensorFromJSONEntry(out_entry);

    auto _op = graph_->CreateOperation<vsi::Relu>();
    (*_op).BindInput(vsi_input).BindOutput(vsi_output);
    ops_.push_back(_op);
  }

  void Add(const size_t& nid) {
    auto node = nodes_[nid];

    auto inputs = node.GetInputs();

    CHECK(inputs.size() >= 2U) << "BatchNormal layer requires at least 2 inputs.";

    JSONGraphNodeEntry out_entry(nid, 0);

    std::vector<std::shared_ptr<vsi::Tensor>> vsi_inputs;
    std::vector<std::shared_ptr<vsi::Tensor>> vsi_outputs;

    auto input_cnt = inputs.size();

    if (node.GetOpName() == "qnn.add") {
      input_cnt = (input_cnt - 2) / 3; //Each input has 3 tensor(data, scale, offset)
      for (size_t j = 0; j < input_cnt; j ++) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[j],
                   &inputs[j + input_cnt], &inputs[j + input_cnt * 2]));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry,
                   &inputs[input_cnt * 3], &inputs[input_cnt * 3 + 1]));
    } else {
      for (const auto& i : inputs) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(i));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry));
    }

    auto add = graph_->CreateOperation<vsi::Add>();
    (*add).BindInputs(vsi_inputs).BindOutputs(vsi_outputs);
    ops_.push_back(add);
  }

  void Clip(const size_t& nid) {
    auto node = nodes_[nid];
    auto inputs = node.GetInputs();
    std::string min = node.GetAttr<std::vector<std::string>>("a_min")[0];
    std::string max = node.GetAttr<std::vector<std::string>>("a_max")[0];

    CHECK(inputs.size() == 1U) << "Clip layer requires 1 input.";

    JSONGraphNodeEntry out_entry(nid, 0);
    auto vsi_input = MakeVSITensorFromJSONEntry(inputs[0]);
    auto vsi_output = MakeVSITensorFromJSONEntry(out_entry);

    auto clip = graph_->CreateOperation<vsi::Clip>(std::stof(min), std::stof(max));
    (*clip).BindInput(vsi_input).BindOutput(vsi_output);
    ops_.push_back(clip);
  }

  void Permute(const size_t& nid) {
    auto node = nodes_[nid];
    auto inputs = node.GetInputs();
    std::string src_layout = node.GetAttr<std::vector<std::string>>("src_layout")[0];
    std::string dst_layout = node.GetAttr<std::vector<std::string>>("dst_layout")[0];
    std::vector<uint32_t> perm;

    if (src_layout == "NHWC" && dst_layout == "NCHW"){
        perm = {1, 2, 0, 3};
    } else if (src_layout == "NCHW" && dst_layout == "NHWC") {
        perm = {2, 0, 1, 3};
    } else {
        LOG(FATAL) << "Unsupported layout transform from " << src_layout << " to " << dst_layout;
    }

    JSONGraphNodeEntry out_entry(nid, 0);
    auto vsi_input = MakeVSITensorFromJSONEntry(inputs[0]);
    auto vsi_output =  MakeVSITensorFromJSONEntry(out_entry, vsi_input->GetQuantization());

    auto permute = graph_->CreateOperation<vsi::Permute>(perm);
    (*permute).BindInput(vsi_input).BindOutput(vsi_output);
    ops_.push_back(permute);
  }

  void BatchNorm(const size_t& nid) {
    auto node = nodes_[nid];

    auto inputs = node.GetInputs();

    /* inputs[0]: input,
     * inputs[1]: gamma,
     * inputs[2]: beta,
     * inputs[3]: moving_mean,
     * inputs[4]: moving_var
     */

    CHECK(inputs.size() == 5U) << "BatchNormal layer requires 5 inputs.";

    JSONGraphNodeEntry out_entry(nid, 0);

    std::vector<std::shared_ptr<vsi::Tensor>> vsi_inputs;
    std::vector<std::shared_ptr<vsi::Tensor>> vsi_outputs;

    vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[0]));
    vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[3]));
    vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[4]));
    vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[1]));
    vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[2]));

    vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry));

    float epsilon = std::stof(node.GetAttr<std::vector<std::string>>("epsilon")[0]);

    auto bn = graph_->CreateOperation<vsi::BatchNorm>(epsilon);
    (*bn).BindInputs(vsi_inputs).BindOutputs(vsi_outputs);
    ops_.push_back(bn);
  }

  void Pool2d(const size_t& nid) {

    vsi::PoolType pool_type;
    vsi::RoundType round_type;

    auto node = nodes_[nid];
    auto inputs = node.GetInputs();

    /* inputs[0]: input data,
     * attr: pool_size,
     * attr: strides,
     * attr: padding,
     */

    JSONGraphNodeEntry out_entry(nid, 0);

    std::vector<std::string> tvm_pool_size = node.GetAttr<std::vector<std::string>>("pool_size");
    std::vector<std::string> tvm_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> tvm_pad = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> tvm_ceil_mode = node.GetAttr<std::vector<std::string>>("ceil_mode");

    if (std::stoi(tvm_ceil_mode[0]) > 0) {
        round_type = vsi::RoundType::CEILING;
    } else {
        round_type = vsi::RoundType::FLOOR;
    }

    if (node.GetOpName() == "nn.max_pool2d") {
      pool_type = vsi::PoolType::MAX;
    } else if (node.GetOpName() == "nn.avg_pool2d" || node.GetOpName() == "qnn.avg_pool2d") {
      pool_type = vsi::PoolType::AVG;
    } else {
      LOG(FATAL) << "Pooling type not supported: " << node.GetOpName();
    }

    std::vector<uint32_t> vsi_ksize;
    for (const auto& i : tvm_pool_size) {
        vsi_ksize.push_back(std::stoi(i));
    }

    std::vector<uint32_t> vsi_stride;
    for (const auto& i : tvm_strides) {
        vsi_stride.push_back(std::stoi(i));
    }

    std::vector<uint32_t> vsi_pad;
    vsi_pad.push_back(std::stoi(tvm_pad[1]));
    vsi_pad.push_back(std::stoi(tvm_pad[3]));
    vsi_pad.push_back(std::stoi(tvm_pad[0]));
    vsi_pad.push_back(std::stoi(tvm_pad[2]));

    auto vsi_input = MakeVSITensorFromJSONEntry(inputs[0]);
    auto vsi_output = MakeVSITensorFromJSONEntry(out_entry, vsi_input->GetQuantization());

    auto pool = graph_->CreateOperation<vsi::Pool2d>(pool_type, vsi::PadType::AUTO, vsi_ksize, vsi_stride, vsi_pad, round_type);
    (*pool).BindInput(vsi_input).BindOutput(vsi_output);
    ops_.push_back(pool);
  }

  void GlobalPool2d(const size_t& nid) {

    vsi::PoolType pool_type;

    auto node = nodes_[nid];
    //JSONGraphNodeEntry input
    auto data_entry = node.GetInputs()[0];

    JSONGraphNodeEntry out_entry(nid, 0);

    if (node.GetOpName() == "nn.global_max_pool2d") {
      pool_type = vsi::PoolType::MAX;
    } else if (node.GetOpName() == "nn.global_avg_pool2d") {
      pool_type = vsi::PoolType::AVG;
    } else {
      LOG(FATAL) << "Pooling type not supported: " << node.GetOpName();
    }

    std::shared_ptr<vsi::Tensor> vsi_input;
    std::shared_ptr<vsi::Tensor> vsi_output;

    vsi_input = MakeVSITensorFromJSONEntry(data_entry);

    vsi_output = MakeVSITensorFromJSONEntry(out_entry);
    auto vsi_shap = vsi_input->GetShape();
    //layout is swapped, NCHW-->WHCN, [0]:W, [1]:H
    std::vector<uint32_t> ksize = {vsi_shap[0], vsi_shap[1]};
    //stride
    std::vector<uint32_t> stride = {1, 1};
    std::vector<uint32_t> pad = {0, 0, 0, 0};

    auto _op = graph_->CreateOperation<vsi::Pool2d>(pool_type, vsi::PadType::AUTO, ksize, stride, pad);
    (*_op).BindInput(vsi_input).BindOutput(vsi_output);
    ops_.push_back(_op);
  }

  void Softmax(const size_t& nid) {
    auto node = nodes_[nid];
    //JSONGraphNodeEntry input
    auto inputs = node.GetInputs();
    //softmax aixs
    auto axis_data_tvm = node.GetAttr<std::vector<std::string>>("axis")[0];
    auto shape_tvm = nodes_[inputs[0].id_].GetOpShape()[inputs[0].index_];
    uint32_t axis_data_vsi = 1;

    axis_data_vsi = ConvertAxis(std::stoi(axis_data_tvm), shape_tvm.size());

    JSONGraphNodeEntry out_entry(nid, 0);

    std::shared_ptr<vsi::Tensor> vsi_input;
    std::shared_ptr<vsi::Tensor> vsi_output;
    std::cout << "inputs.size" << inputs.size() << std::endl;
    CHECK(inputs.size() == 1U || inputs.size() == 5U)
          << "Softmax requires 5 inputs with quantization, 1 inputs without.";
    if (inputs.size() == 5) {
      vsi_input = MakeVSITensorFromJSONEntry(inputs[0], &inputs[1], &inputs[2]);
      vsi_output = MakeVSITensorFromJSONEntry(out_entry, &inputs[3], &inputs[4]);
    } else {
      vsi_input = MakeVSITensorFromJSONEntry(inputs[0]);
      vsi_output = MakeVSITensorFromJSONEntry(out_entry);
    }

    //set beta to 1.0
    auto _op = graph_->CreateOperation<vsi::Softmax>(1.0f, axis_data_vsi);
    (*_op).BindInput(vsi_input).BindOutput(vsi_output);
    ops_.push_back(_op);
  }

  void Dropout(const size_t& nid) {
    auto node = nodes_[nid];
    //JSONGraphNodeEntry input
    auto inputs = node.GetInputs();

    auto ratio_tvm = node.GetAttr<std::vector<std::string>>("rate")[0];
    //auto shape_tvm = nodes_[inputs[0].id_].GetOpShape()[inputs[0].index_];

    uint32_t ratio_vsi = 1;
    //ratio_vsi = ConvertAxis(std::stoi(ratio_tvm), shape_tvm.size());
    ratio_vsi = std::stoi(ratio_tvm);

    std::vector<std::shared_ptr<vsi::Tensor>> vsi_inputs;
    std::vector<std::shared_ptr<vsi::Tensor>> vsi_outputs;

    for (const auto& i : inputs) {
      vsi_inputs.push_back(MakeVSITensorFromJSONEntry(i));
    }

    JSONGraphNodeEntry out_entry(nid, 0);
    vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry));

    auto dropout = graph_->CreateOperation<vsi::Dropout>(ratio_vsi);
    (*dropout).BindInputs(vsi_inputs).BindOutputs(vsi_outputs);
    ops_.push_back(dropout);
  }

  void Concat(const size_t& nid) {
    auto node = nodes_[nid];
    //JSONGraphNodeEntry input
    auto inputs = node.GetInputs();

    auto axis_tvm = node.GetAttr<std::vector<std::string>>("axis")[0];
    auto shape_tvm = nodes_[inputs[0].id_].GetOpShape()[inputs[0].index_];

    uint32_t axis_vsi = 1;
    axis_vsi = ConvertAxis(std::stoi(axis_tvm), shape_tvm.size());

    auto input_cnt = inputs.size();

    std::vector<std::shared_ptr<vsi::Tensor>> vsi_inputs;
    std::vector<std::shared_ptr<vsi::Tensor>> vsi_outputs;

    JSONGraphNodeEntry out_entry(nid, 0);
    if (node.GetOpName() == "qnn.concatenate") {
      input_cnt = (input_cnt - 2) / 3; //Each input has 3 tensor(data, scale, offset)
      for (size_t j = 0; j < input_cnt; j ++) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[j],
			       	                        &inputs[j + input_cnt],
						       	&inputs[j + input_cnt * 2]));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry,
			                               &inputs[input_cnt * 3],
						       &inputs[input_cnt * 3 + 1]));
    } else {
      for (const auto& i : inputs) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(i));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry));
    }

    auto concat = graph_->CreateOperation<vsi::Concat>(axis_vsi, input_cnt);
    (*concat).BindInputs(vsi_inputs).BindOutputs(vsi_outputs);
    ops_.push_back(concat);
  }

  void Conv2D(const size_t& nid) {
    auto node = nodes_[nid];
    std::vector<std::string> pad = node.GetAttr<std::vector<std::string>>("padding");
    std::vector<std::string> strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> dilation = node.GetAttr<std::vector<std::string>>("dilation");
    auto is_depthwise = node.GetAttr<std::vector<std::string>>("is_depthwise")[0];

    int groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);

    // Collect inputs and outputs, handling both nn.conv2d and qnn.conv2d cases.
    std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();
    std::vector<std::shared_ptr<vsi::Tensor>> vsi_inputs;
    std::vector<std::shared_ptr<vsi::Tensor>> vsi_outputs;
    size_t num_inputs = inputs.size();
    JSONGraphNodeEntry out_entry(nid, 0);
    bool has_bias;
    if (node.GetOpName() == "qnn.conv2d") {
      CHECK(num_inputs >= 10U && num_inputs <= 11U)
          << "Quantized convolution requires 11 inputs with a bias, 9 inputs without.";
      has_bias = num_inputs == 11;
      vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[0], &inputs[4], &inputs[2]));
      vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[1], &inputs[5], &inputs[3]));
      if (has_bias) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[6], &inputs[9], &inputs[10]));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry, &inputs[6 + has_bias], &inputs[7 + has_bias]));
    } else {
      CHECK(num_inputs >= 2U && num_inputs <= 3U)
          << "Convolution requires 3 inputs with a bias, 2 inputs without.";
      has_bias = num_inputs == 3;
      for (const auto& i : inputs) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(i));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry));
    }

    // TVM: top, left, bottom, right -> VSI: left, right, top, bottom
    auto weight_tensor = vsi_inputs[1];
    std::vector<uint32_t> vsi_pad;
    vsi_pad.push_back(std::stoi(pad[1]));
    vsi_pad.push_back(std::stoi(pad[3]));
    vsi_pad.push_back(std::stoi(pad[0]));
    vsi_pad.push_back(std::stoi(pad[2]));

    std::vector<uint32_t> vsi_strides;
    vsi_strides.push_back(std::stoi(strides[0]));
    vsi_strides.push_back(std::stoi(strides[1]));

    std::vector<uint32_t> vsi_dilation;
    vsi_dilation.push_back(std::stoi(dilation[0]));
    vsi_dilation.push_back(std::stoi(dilation[1]));

    std::vector<uint32_t> vsi_ksize;
    vsi_ksize.push_back(weight_tensor->GetShape()[0]);
    vsi_ksize.push_back(weight_tensor->GetShape()[1]);

    if (!has_bias) {
      vsi_inputs.push_back(MakeDummyBiasTensor(vsi_inputs[0]->GetDataType(),
			      {weight_tensor->GetShape()[3]}));
    }
    int32_t vsi_multiplier = 0;
    if (std::stoi(is_depthwise) == 1) {
      vsi_multiplier = static_cast<int32_t>(weight_tensor->GetShape()[2]);
    }

    auto conv = graph_->CreateOperation<vsi::Conv2d>(static_cast<int32_t>(weight_tensor->GetShape()[3]),
		    vsi::PadType::AUTO, vsi_ksize, vsi_strides, vsi_dilation,
		    vsi_pad, groups, vsi_multiplier);
    (*conv).BindInputs(vsi_inputs).BindOutputs(vsi_outputs);
    ops_.push_back(conv);
  }


  void BatchFlatten(const size_t& nid) {
  }

  bool IsInputNode(const size_t& nid) {
    return std::find(input_nodes_.begin(), input_nodes_.end(), nid) != input_nodes_.end();
  }

  bool IsOutputNode(const size_t& nid) {
    int size = outputs_.size();
    for(int i = 0; i< size; i++) {
      if(outputs_[i].id_ == nid)
        return true;
    }
    return false;
  }

  /*!
   * \brief Create an VSI tensor given the JSON representation. If scale
   * and offset are given, then create a quantized VSI tensor.
   *
   * \param tensor The tensor to represent.
   * \param scale (optional) The scale of the tensor as an input.
   * \param offset (optional) The offset of the tensor as an input.
   * \return VSI Tensor.
   */
  std::shared_ptr<vsi::Tensor> MakeVSITensorFromJSONEntry(const JSONGraphNodeEntry& tensor,
                                                 JSONGraphNodeEntry* scale = nullptr,
                                                 JSONGraphNodeEntry* offset = nullptr) {
    vsi::Quantization vsi_quant;
    if (scale != nullptr && offset != nullptr) {
      auto scale_tensor = data_entry_[EntryID(*scale)];
      auto offset_tensor = data_entry_[EntryID(*offset)];
      std::vector<float> scale_data = GetVectorFromDLTensor<float>(scale_tensor);
      std::vector<int> offset_data = GetVectorFromDLTensor<int>(offset_tensor);
      CHECK(scale_data.size() == 1 && offset_data.size() == 1)
            << "Currently only per-layer quantization is supported in the VSI runtime.";
      vsi_quant = vsi::Quantization(vsi::QuantType::ASYMMETRIC, scale_data[0], offset_data[0]);
    }

    return MakeVSITensorFromJSONEntry(tensor, vsi_quant);
  }
  std::shared_ptr<vsi::Tensor> MakeVSITensorFromJSONEntry(const JSONGraphNodeEntry& tensor,
                                                 const vsi::Quantization vsi_quant) {
    auto eid = EntryID(tensor);

    if (entry_out_tensor_.count(eid) != 0) {
      //using the existed VSItensor
      return entry_out_tensor_[eid];
    }
    //create new VSItensor
    JSONGraphNode node = nodes_[tensor.id_];
    void* node_data = nullptr;
    vsi::TensorAttribute vsi_attr;

    if (node.GetOpType() == "const") {
      node_data = data_entry_[EntryID(tensor)]->data;
      vsi_attr = vsi::TensorAttribute::CONSTANT;
    } else if (IsInputNode(tensor.id_)) {
      vsi_attr = vsi::TensorAttribute::INPUT;
    } else if (IsOutputNode(tensor.id_)) {
      vsi_attr = vsi::TensorAttribute::OUTPUT;
    } else {
      vsi_attr = vsi::TensorAttribute::TRANSIENT;
    }

    auto vsi_tensor = MakeVSITensor(node, node_data, vsi_attr, vsi_quant);
    entry_out_tensor_.insert({eid, vsi_tensor});
    return entry_out_tensor_[eid];
  }

  std::shared_ptr<vsi::Tensor> MakeDummyBiasTensor(vsi::DataType dtype,
		 		 vsi::ShapeType bias_shape) {
    std::vector<float> bias_data(bias_shape[0], 0);

    vsi::TensorSpec bias_spec(dtype, bias_shape,
		    vsi::TensorAttribute::CONSTANT);
    auto bias = graph_->CreateTensor(bias_spec, bias_data.data());
    dummy_tensor_.push_back(bias);

    return bias;
  }

  std::shared_ptr<vsi::Tensor> MakeVSITensor(const JSONGraphNode& tensor_rep, void* data,
				  vsi::TensorAttribute vsi_attr,
				  const vsi::Quantization vsi_quant) {
    //VSI parameter
    vsi::ShapeType vsi_shape;
    vsi::DataType vsi_dtype;
    //TVM parameter
    std::vector<int64_t> tvm_shape = tensor_rep.GetOpShape()[0];
    DLDataType tvm_dtype = tensor_rep.GetOpDataType()[0];

    for (unsigned int i = 0; i < tvm_shape.size(); i ++) {
      vsi_shape.push_back(tvm_shape[tvm_shape.size() - i - 1]);
    }

    if (tvm_dtype.code == DLDataTypeCode::kDLFloat && tvm_dtype.bits == 32) {
      vsi_dtype = vsi::DataType::FLOAT32;
      std::cout << "vsi::DataType::FLOAT32" << std::endl;
    } else if (tvm_dtype.code == DLDataTypeCode::kDLUInt && tvm_dtype.bits == 8) {
      vsi_dtype = vsi::DataType::UINT8;
      std::cout << "vsi::DataType::UINT8" << std::endl;
    } else if (tvm_dtype.code == DLDataTypeCode::kDLInt && tvm_dtype.bits == 32) {
      vsi_dtype = vsi::DataType::INT32;
      std::cout << "vsi::DataType::INT32" << std::endl;
    } else {
      LOG(FATAL) << "Unsupported data type.";
    }

    auto input_spec = vsi::TensorSpec(vsi_dtype, vsi_shape, vsi_attr, vsi_quant);
    std::shared_ptr<vsi::Tensor> tensor;
    if (data != nullptr)
      tensor = graph_->CreateTensor(input_spec, data);
    else
      tensor = graph_->CreateTensor(input_spec);
    return tensor;
  }

  std::shared_ptr<vsi::Context> context_;
  std::shared_ptr<vsi::Graph> graph_;
  /* The entry ID to its corresponding output memory. */
  std::unordered_map<uint32_t, std::shared_ptr<vsi::Tensor>> entry_out_tensor_;
  std::vector<std::shared_ptr<vsi::Tensor>> dummy_tensor_;
  std::vector<std::shared_ptr<vsi::Operation>> ops_;
};

#else

class VsiNpuJSONRuntime : public JSONRuntimeBase {

 public:
  VsiNpuJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "vsi_npu_json"; }

  void Init(const Array<NDArray>& consts) override {

  }

  void Run() override {
  }
};
#endif



runtime::Module VsiNpuJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<VsiNpuJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.VsiNpuJSONRuntimeCreate").set_body_typed(VsiNpuJSONRuntimeCreate);

#ifdef USE_VSI_NPU_RUNTIME
TVM_REGISTER_GLOBAL("runtime.module.loadbinary_vsi_npu_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<VsiNpuJSONRuntime>);
#endif
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
