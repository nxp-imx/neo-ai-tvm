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
    std::cout << "func:"<<__FUNCTION__ << "   line:" << __LINE__ << std::endl;
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
        std::cout << "data_size = " << data_size << std::endl;
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
      std::cout << "nid = " << nid << "; input = " << IsInputNode(nid) << std::endl;
      std::cout << "nid = " << nid << "; output = " << IsOutputNode(nid) << std::endl;
      if (node.GetOpType() == "kernel") {
        CHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if ("nn.batch_flatten" == op_name) {
          LOG(INFO) << "Build op: " << op_name;
	  Flatten(nid);
        } else if ("nn.dense" == op_name) {
	  Dense(nid);
          LOG(INFO) << "Build op: " << op_name;
        } else if ("nn.relu" == op_name) {
          LOG(INFO) << "Build op: " << op_name;
          Relu(nid);
        } else if ("nn.softmax" == op_name) {
          LOG(INFO) << "Build op: " << op_name;
          Softmax(nid);
        } else if (("nn.global_avg_pool2d" == op_name) || ("nn.global_max_pool2d" == op_name)) {
          LOG(INFO) << "Build op: " << op_name;
          GlobalPool2d(nid);
        } else if ("add" == op_name) {
          LOG(INFO) << "Build op: " << op_name;
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
    assert(graph_->Compile());
    std::cout << "Pass" << std::endl;
  }

  void Flatten(const size_t& nid) {
    auto node = nodes_[nid];
    JSONGraphNodeEntry out_entry(nid, 0);
    std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();

    CHECK(inputs.size() == 1U) << "Flatten layer requires 1 inputs.";

    std::vector<std::shared_ptr<vsi::Tensor>> vsi_inputs;
    std::vector<std::shared_ptr<vsi::Tensor>> vsi_outputs;
    vsi_inputs.push_back(MakeVSITensorFromJSONEntry(inputs[0]));
    vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry));

    std::vector<int64_t> tvm_shape = nodes_[inputs[0].id_].GetOpShape()[0];
    uint32_t data_size = 1;
    for (unsigned int i = 0; i < tvm_shape.size(); i ++) {
      data_size *= tvm_shape[i];
    }

    std::vector<uint32_t> output_shape({data_size});
    auto flatten = graph_->CreateOperation<vsi::Reshape>(output_shape.data(), 1);
    (*flatten).BindInputs(vsi_inputs).BindOutputs(vsi_outputs);
  }

  void Dense(const size_t& nid) {
    auto node = nodes_[nid];
    // Collect inputs and outputs, handling both nn.dense and qnn.dense cases.
    std::vector<JSONGraphNodeEntry> inputs = node.GetInputs();
    size_t num_inputs = inputs.size();
    bool has_bias;
    JSONGraphNodeEntry out_entry(nid, 0);
    
    std::vector<std::shared_ptr<vsi::Tensor>> vsi_inputs;
    std::vector<std::shared_ptr<vsi::Tensor>> vsi_outputs;

    if (node.GetOpName() == "qnn.dense") {
	    //qnn.dense
	    
    } else {
      CHECK(num_inputs >= 2U && num_inputs <= 3U)
          << "Fully connected (dense) layer requires 3 inputs with a bias, 2 inputs without.";
      has_bias = num_inputs == 3;
      for (const auto& i : inputs) {
        vsi_inputs.push_back(MakeVSITensorFromJSONEntry(i));
      }
      vsi_outputs.push_back(MakeVSITensorFromJSONEntry(out_entry));
    }

    auto weight_tensor = vsi_inputs[1];
    auto fc = graph_->CreateOperation<vsi::FullyConnected>(1, weight_tensor->GetShape()[1]);
    (*fc).BindInputs(vsi_inputs).BindOutputs(vsi_outputs);
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

  }

  void Add(const size_t& nid) {
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

    auto _op = graph_->CreateOperation<vsi::Pool2d>(pool_type, vsi::PadType::AUTO, ksize, stride);
    (*_op).BindInput(vsi_input).BindOutput(vsi_output);

  }

  void Softmax(const size_t& nid) {
    auto node = nodes_[nid];

    //JSONGraphNodeEntry input
    auto data_entry = node.GetInputs()[0];

    //softmax aixs
    auto axis_data_tvm = node.GetAttr<std::vector<std::string>>("axis")[0];
    auto shape_tvm = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];

    uint32_t axis_data_vsi = 1;

    LOG(INFO) << "Softmax tvm_axis: " << axis_data_tvm;

    axis_data_vsi = ConvertAxis(std::stoi(axis_data_tvm), shape_tvm.size());

    LOG(INFO) << "Softmax vsi_axis: " << axis_data_vsi;

    JSONGraphNodeEntry out_entry(nid, 0);

    std::shared_ptr<vsi::Tensor> vsi_input;
    std::shared_ptr<vsi::Tensor> vsi_output;

    vsi_input = MakeVSITensorFromJSONEntry(data_entry);

    vsi_output = MakeVSITensorFromJSONEntry(out_entry);

    //set beta to 1.0
    auto _op = graph_->CreateOperation<vsi::Softmax>(1.0f, axis_data_vsi);
    (*_op).BindInput(vsi_input).BindOutput(vsi_output);
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
    auto eid = EntryID(tensor);
    std::cout << "In MakeVSITensorFromJSONEntry eid=" << eid << std::endl;

    if (entry_out_tensor_.count(eid) != 0) {
      //using the existed VSItensor
      std::cout << "###BindVSITensor using exting ID: " << eid <<std::endl;
      return entry_out_tensor_[eid];
    }
    std::cout << "func:"<<__FUNCTION__ << "   line:" << __LINE__ << std::endl;
    //create new VSItensor
    JSONGraphNode node = nodes_[tensor.id_];
    void* node_data = nullptr;
    vsi::TensorAttribute vsi_attr;
    std::cout << "func:"<<__FUNCTION__ << "   line:" << __LINE__ << std::endl;

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
    std::cout << "func:"<<__FUNCTION__ << "   line:" << __LINE__ << std::endl;

    std::cout << "###BindVSITensor bind new ID: " << eid <<std::endl;
    auto vsi_tensor = MakeVSITensor(node, node_data, vsi_attr, scale, offset);
    entry_out_tensor_.insert({eid, vsi_tensor});
    return entry_out_tensor_[eid];
  }

  std::shared_ptr<vsi::Tensor> MakeVSITensor(const JSONGraphNode& tensor_rep, void* data,
				  vsi::TensorAttribute vsi_attr,
                                  JSONGraphNodeEntry* scale = nullptr,
                                  JSONGraphNodeEntry* offset = nullptr) {
    std::cout << "In MakeVSITensor eid=" << std::endl;
    //VSI parameter
    vsi::ShapeType vsi_shape;
    vsi::DataType vsi_dtype;
    //TVM parameter
    std::vector<int64_t> tvm_shape = tensor_rep.GetOpShape()[0];
    DLDataType tvm_dtype = tensor_rep.GetOpDataType()[0];

    std::cout << "vsi::vsi_shape ={";
    for (unsigned int i = 0; i < tvm_shape.size(); i ++) {
      vsi_shape.push_back(tvm_shape[tvm_shape.size() - i - 1]);
      std::cout << vsi_shape[i] << ",";
    }
    std::cout << "}" << std::endl;

    if (tvm_dtype.code == DLDataTypeCode::kDLFloat && tvm_dtype.bits == 32) {
      vsi_dtype = vsi::DataType::FLOAT32;
      std::cout << "vsi::DataType::FLOAT32" << std::endl;
    } else if (tvm_dtype.code == DLDataTypeCode::kDLUInt && tvm_dtype.bits == 8) {
      vsi_dtype = vsi::DataType::UINT8;
      std::cout << "vsi::DataType::UINT8" << std::endl;
    } else {
      vsi_dtype = vsi::DataType::FLOAT32;
      LOG(FATAL) << "Datatype " << tvm_dtype << " unsupported by VSI runtime";
    }


    // If scale and offset provided create quantized ACL tensor.
    if (scale != nullptr && offset != nullptr) {
      //std::vector<float> scale_data = GetVectorFromDLTensor<float>(data_entry_[EntryID(*scale)]);
      //std::vector<int> offset_data = GetVectorFromDLTensor<int>(data_entry_[EntryID(*offset)]);
      //CHECK(scale_data.size() == 1 && offset_data.size() == 1)
      //    << "Currently only per-layer quantization is supported in the Arm Compute Library runtime.";
      //vsi::Quantization input_quant(vsi::QuantType::ASYMMETRIC, scale_data[0], offset_data[0]);
    }
    std::cout << "In MakeVSITensor 22222=" << std::endl;

    vsi::TensorSpec input_spec(vsi_dtype, vsi_shape, vsi_attr);
    std::shared_ptr<vsi::Tensor> tensor;
    if (data != nullptr)
      tensor = graph_->CreateTensor(input_spec, data);
    else
      tensor = graph_->CreateTensor(input_spec);
    return tensor;
  }

#if 0
  template <typename T>
  std::vector<T> GetVectorFromDLTensor(const DLTensor* tensor) {
    CHECK(tensor) << "Cannot convert a nullptr";
    int len = 1;
    for (int i = 0; i < tensor->ndim; i++) {
      len *= tensor->shape[i];
    }
    T* data = static_cast<T*>(tensor->data);
    return std::vector<T>(data, data + len);
  }
#endif


  std::shared_ptr<vsi::Context> context_;
  std::shared_ptr<vsi::Graph> graph_;
  /* The entry ID to its corresponding output memory. */
  std::unordered_map<uint32_t, std::shared_ptr<vsi::Tensor>> entry_out_tensor_;
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