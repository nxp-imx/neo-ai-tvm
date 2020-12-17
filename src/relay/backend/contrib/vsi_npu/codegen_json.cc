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
 * \file src/relay/backend/contrib/VsiNpu/codegen_json.cc
 * \brief Implementation of VsiNpu codegen APIs.
 */

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>

#include <fstream>
#include <numeric>
#include <sstream>

#include "../../utils.h"

#include "../../../../runtime/contrib/json/json_node.h"
#include "../codegen_json/codegen_json.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

class VsiNpuJSONSerializer : public backend::contrib::JSONSerializer {
  using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
  using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

 public:
  /*!
   * \brief A series of operators that form a composite
   * convolution. Supports both nn.conv2d and qnn.conv2d.
   */
  struct CompositeConvNode {
    const CallNode* pad = nullptr;
    const CallNode* conv = nullptr;
    const CallNode* bias = nullptr;
    const CallNode* requantize = nullptr;
  };

  /*!
   * \brief A series of operators that form a composite
   * dense layer. Supports both nn.dense and qnn.dense.
   */
  struct CompositeDenseNode {
    const CallNode* dense = nullptr;
    const CallNode* bias = nullptr;
    const CallNode* requantize = nullptr;
  };

  /*!
   * \brief A series of operators that form a composite
   * softmax layer. Supports both qnn.softmax.
   */
  struct CompositeQnnSoftmaxNode {
    const CallNode* dequantize = nullptr;
    const CallNode* softmax = nullptr;
    const CallNode* quantize = nullptr;
  };

  /*!
   * \brief A series of operators that form a composite
   * avg pool2d layer. Supports both qnn.avg_pool2d.
   */
  struct CompositeQnnAvgPool2DNode {
    const CallNode* pre_cast = nullptr;
    const CallNode* avg_pool2d = nullptr;
    const CallNode* post_cast = nullptr;
  };

  VsiNpuJSONSerializer(const std::string& symbol, const Expr& expr) : JSONSerializer(symbol, expr) {}

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* cn) override {

#if 0
    Expr expr = GetRef<Expr>(cn);
    std::string name;
    const CallNode* call = cn;
    if (const auto* op_node = cn->op.as<OpNode>()) {
      name = op_node->name;
    } else if (const auto* fn = cn->op.as<FunctionNode>()) {
      auto comp = fn->GetAttr<String>(attr::kComposite);
      CHECK(comp.defined()) << "VsiNpu JSON runtime only supports composite functions.";
      name = comp.value();

      if (name == "vsi_npu.dense") {
        call = GetRootCall(fn->body.as<CallNode>(), 1, {"nn.dense", "nn.bias_add"});
        CHECK(call->op.as<OpNode>()) << "Not op node";
      } else {
        LOG(FATAL) << "Unrecognized VsiNpu pattern: " << name;
      }
    } else {
      LOG(FATAL) << "VsiNpu JSON runtime does not support calls to " << cn->op->GetTypeKey();
    }

    std::vector<JSONGraphNodeEntry> inputs;
    for (const auto& arg : cn->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }
    auto node = std::make_shared<JSONGraphNode>(name,     /* name_ */
                                                "kernel", /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);
    SetCallNodeAttribute(node, call);
    return AddNode(node, GetRef<Expr>(cn));
#else
    if (cn->op.as<OpNode>()) {
      return JSONSerializer::VisitExpr_(cn);
    }
    if (!cn->op.as<FunctionNode>()) {
      LOG(FATAL) << "VSI NPU JSON runtime does not support calls to "
                 << cn->op->GetTypeKey();
    }
    auto fn = cn->op.as<FunctionNode>();
    auto comp = fn->GetAttr<String>(attr::kComposite);
    CHECK(comp.defined()) << "VSI NPU JSON runtime only supports composite functions.";
    const std::string name = comp.value();
    std::shared_ptr<JSONGraphNode> json_node;
    if (name == "vsi_npu.dense" or name == "vsi_npu.qnn_dense" ) {
      json_node = CreateCompositeDenseJSONNode(cn);
    } else if (name == "vsi_npu.conv2d" || name == "vsi_npu.qnn_conv2d") {
      json_node = CreateCompositeConvJSONNode(cn);
    } else if (name == "vsi_npu.qnn_softmax") {
      json_node = CreateCompositeQnnSoftmaxJSONNode(cn);
    } else if (name == "vsi_npu.qnn_avg_pool2d") {
      json_node = CreateCompositeAvgPool2DJSONNode(cn);
    } else {
      LOG(FATAL) << "Unrecognized VSI NPU pattern: " << name;
    }
    return AddNode(json_node, GetRef<Expr>(cn));

#endif
  }
 private:
  std::shared_ptr<JSONGraphNode> CreateCompositeAvgPool2DJSONNode(const CallNode* cn) {
    CompositeQnnAvgPool2DNode nodes = UnpackCompositeQnnAvgPool2D(cn);
    std::string name = "qnn.avg_pool2d";

    // Inputs must be added in the same order they appear in the relay graph.
    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);

    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.avg_pool2d);
    return json_node;
  }

  std::shared_ptr<JSONGraphNode> CreateCompositeQnnSoftmaxJSONNode(const CallNode* cn) {
    CompositeQnnSoftmaxNode nodes = UnpackCompositeQnnSoftmax(cn);
    std::string name = "qnn.softmax";

    // Inputs must be added in the same order they appear in the relay graph.
    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    inputs.push_back(VisitExpr(nodes.dequantize->args[1])[0]);  // input scale
    inputs.push_back(VisitExpr(nodes.dequantize->args[2])[0]);  // input zero-point
    inputs.push_back(VisitExpr(nodes.quantize->args[1])[0]);  // output scale
    inputs.push_back(VisitExpr(nodes.quantize->args[2])[0]);  // output zero-point

    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.softmax);
    return json_node;
  }

    std::shared_ptr<JSONGraphNode> CreateCompositeDenseJSONNode(const CallNode* cn) {
    CompositeDenseNode nodes = UnpackCompositeDense(cn);
    std::string name = "nn.dense";

    // Inputs must be added in the same order they appear in the relay graph.
    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    inputs.push_back(VisitExpr(nodes.dense->args[1])[0]);
    if (nodes.requantize) {
      name = "qnn.dense";
      inputs.push_back(VisitExpr(nodes.dense->args[2])[0]);  // input zero-point
      inputs.push_back(VisitExpr(nodes.dense->args[3])[0]);  // weight zero-point
      inputs.push_back(VisitExpr(nodes.dense->args[4])[0]);  // input scale
      inputs.push_back(VisitExpr(nodes.dense->args[5])[0]);  // weight scale
    }
    if (nodes.bias) {
      inputs.push_back(VisitExpr(nodes.bias->args[1])[0]);
    }
    if (nodes.requantize) {
      inputs.push_back(VisitExpr(nodes.requantize->args[3])[0]);  // output scale
      inputs.push_back(VisitExpr(nodes.requantize->args[4])[0]);  // output zero-point
      inputs.push_back(VisitExpr(nodes.requantize->args[1])[0]);  // bias scale
      inputs.push_back(VisitExpr(nodes.requantize->args[2])[0]);  // bias zero-point
    }

    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.dense);
    return json_node;
  }

  /*!
   * \brief Create a JSON representation of a composite convolution.
   *
   * \param cn The call to be represented.
   * \return A JSON representation of a specific operator.
   */
  std::shared_ptr<JSONGraphNode> CreateCompositeConvJSONNode(const CallNode* cn) {
    CompositeConvNode nodes = UnpackCompositeConvolution(cn);
    std::string name = "nn.conv2d";

    const auto* conv_attr = nodes.conv->attrs.as<Conv2DAttrs>();
    CHECK(conv_attr);
    CHECK(conv_attr->kernel_layout == "OIHW")
        << "Kernel layout must be OIHW, has the module been pre-processed correctly?";
    CHECK(conv_attr->data_layout == "NCHW")
        << "Input data layout must be NCHW, has the module been pre-processed correctly?";

    // Inputs must be added in the same order they appear in the relay graph.
    std::vector<JSONGraphNodeEntry> inputs;
    inputs.push_back(VisitExpr(cn->args[0])[0]);
    inputs.push_back(VisitExpr(nodes.conv->args[1])[0]);
    if (nodes.requantize) {
      name = "qnn.conv2d";
      inputs.push_back(VisitExpr(nodes.conv->args[2])[0]);  // input zero-point
      inputs.push_back(VisitExpr(nodes.conv->args[3])[0]);  // kernel zero-point
      inputs.push_back(VisitExpr(nodes.conv->args[4])[0]);  // input scale
      inputs.push_back(VisitExpr(nodes.conv->args[5])[0]);  // kernel scale
    }
    if (nodes.bias) {
      inputs.push_back(VisitExpr(nodes.bias->args[1])[0]);
    }
    if (nodes.requantize) {
      inputs.push_back(VisitExpr(nodes.requantize->args[3])[0]);  // output scale
      inputs.push_back(VisitExpr(nodes.requantize->args[4])[0]);  // output zero-point
      inputs.push_back(VisitExpr(nodes.requantize->args[1])[0]);  // bias scale
      inputs.push_back(VisitExpr(nodes.requantize->args[2])[0]);  // bias zero-point
    }

    auto json_node = std::make_shared<JSONGraphNode>(name, "kernel", inputs, 1);
    SetCallNodeAttribute(json_node, nodes.conv);

    // Override attributes
    if (nodes.pad) {
      const auto* pad_attr = nodes.pad->attrs.as<PadAttrs>();
      CHECK(pad_attr);
      auto p = pad_attr->pad_width;
      // Convert to TVM layout for now, conversion to VSI layout takes place in runtime.
      // Standard convolution pad layout for TVM: top, left, bottom, right.
      std::vector<std::string> padding = {std::to_string(p[1][0].as<IntImmNode>()->value),
                                          std::to_string(p[2][0].as<IntImmNode>()->value),
                                          std::to_string(p[1][1].as<IntImmNode>()->value),
                                          std::to_string(p[2][1].as<IntImmNode>()->value)};
      std::vector<dmlc::any> padding_attr;
      padding_attr.emplace_back(padding);
      json_node->SetAttr("padding", padding_attr);
    }

    // Create depthwise info
    bool is_depthwise = conv_attr->channels.defined() &&
                         tvm::tir::ExprDeepEqual()(conv_attr->channels, conv_attr->groups) &&
                         conv_attr->groups != 1;
    std::vector<dmlc::any> depthwise_attr;
    std::vector<std::string> depthwise = {std::to_string(is_depthwise)};
    depthwise_attr.emplace_back(depthwise);
    json_node->SetAttr("is_depthwise", depthwise_attr);

    return json_node;
  }
  /*!
   * \brief Extract qnn.avg_pool2d nodes from a composite function.
   *
   * \param cn The call node of the composite function.
   * \return Extracted composite convolution nodes.
   */
  static CompositeQnnAvgPool2DNode UnpackCompositeQnnAvgPool2D(const CallNode* cn) {
    CompositeQnnAvgPool2DNode nodes{};
    const auto* fn = cn->op.as<FunctionNode>();
    CHECK(fn);

    // Traverse composite dense function from child to parent
    const auto* current_call = fn->body.as<CallNode>();
    CHECK(backend::IsOp(current_call, "cast"));
    nodes.post_cast = current_call;
    current_call = current_call->args[0].as<CallNode>();
    CHECK(backend::IsOp(current_call, "nn.avg_pool2d"));
    nodes.avg_pool2d = current_call;
    current_call = current_call->args[0].as<CallNode>();
    CHECK(backend::IsOp(current_call, "cast"));
    nodes.pre_cast = current_call;
    return nodes;
  }

  /*!
   * \brief Extract qnn.softmax nodes from a composite function.
   *
   * \param cn The call node of the composite function.
   * \return Extracted composite convolution nodes.
   */
  static CompositeQnnSoftmaxNode UnpackCompositeQnnSoftmax(const CallNode* cn) {
    CompositeQnnSoftmaxNode nodes{};
    const auto* fn = cn->op.as<FunctionNode>();
    CHECK(fn);

    // Traverse composite dense function from child to parent
    const auto* current_call = fn->body.as<CallNode>();
    CHECK(backend::IsOp(current_call, "qnn.quantize"));
    nodes.quantize = current_call;
    current_call = current_call->args[0].as<CallNode>();
    CHECK(backend::IsOp(current_call, "nn.softmax"));
    nodes.softmax = current_call;
    current_call = current_call->args[0].as<CallNode>();
    CHECK(backend::IsOp(current_call, "qnn.dequantize"));
    nodes.dequantize = current_call;
    return nodes;
  }

  /*!
   * \brief Extract dense nodes from a composite function.
   *
   * \param cn The call node of the composite function.
   * \return Extracted composite convolution nodes.
   */
  static CompositeDenseNode UnpackCompositeDense(const CallNode* cn) {
    CompositeDenseNode nodes{};
    const auto* fn = cn->op.as<FunctionNode>();
    CHECK(fn);

    // Traverse composite dense function from child to parent
    const auto* current_call = fn->body.as<CallNode>();
    if (backend::IsOp(current_call, "qnn.requantize")) {
      nodes.requantize = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "nn.bias_add") or backend::IsOp(current_call, "add")) {
      nodes.bias = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    // Enforce a dense node exists at this point during traversal
    if (nodes.requantize) {
      CHECK(backend::IsOp(current_call, "qnn.dense"));
    } else {
      CHECK(backend::IsOp(current_call, "nn.dense"));
    }
    nodes.dense = current_call;
    return nodes;
  }

  /*!
   * \brief Extract convolution nodes from a composite function.
   *
   * \param cn The call node of the composite function.
   * \return Extracted composite convolution nodes.
   */
  static CompositeConvNode UnpackCompositeConvolution(const CallNode* cn) {
    CompositeConvNode nodes{};
    const auto* fn = cn->op.as<FunctionNode>();
    CHECK(fn);

    // Traverse composite convolution function from child to parent
    const auto* current_call = fn->body.as<CallNode>();
    if (backend::IsOp(current_call, "qnn.requantize")) {
      nodes.requantize = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    if (backend::IsOp(current_call, "nn.bias_add") || backend::IsOp(current_call, "add")) {
      nodes.bias = current_call;
      current_call = current_call->args[0].as<CallNode>();
    }
    // Enforce a convolution node exists at this point during traversal
    if (nodes.requantize) {
      CHECK(backend::IsOp(current_call, "qnn.conv2d"));
    } else {
      CHECK(backend::IsOp(current_call, "nn.conv2d"));
    }
    nodes.conv = current_call;
    if (!current_call->args.empty() && current_call->args[0]->IsInstance<CallNode>()) {
      current_call = current_call->args[0].as<CallNode>();
      if (backend::IsOp(current_call, "nn.pad")) {
        nodes.pad = current_call;
      }
    }
    return nodes;
  }

};

/*!
 * \brief The external compiler/codegen tool. It takes a Relay expression/module and
 * compile it into a runtime module.
 */
runtime::Module VsiNpuCompiler(const ObjectRef& ref) {
  CHECK(ref->IsInstance<FunctionNode>());
  auto func = Downcast<Function>(ref);
  auto func_name = GetExtSymbol(func);
  VsiNpuJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto params = serializer.GetParams();

  const auto* pf = runtime::Registry::Get("runtime.VsiNpuJSONRuntimeCreate");
  CHECK(pf != nullptr) << "Cannot find JSON runtime module to create";
  auto mod = (*pf)(func_name, graph_json, params);
  return mod;
}

TVM_REGISTER_GLOBAL("relay.ext.vsi_npu").set_body_typed(VsiNpuCompiler);
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
