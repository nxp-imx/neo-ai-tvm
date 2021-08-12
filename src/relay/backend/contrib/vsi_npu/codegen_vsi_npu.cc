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

#include "codegen_vsi_npu.h"

#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

class VsiNpuCodegen : public CodegenCBase {
 public:
  explicit VsiNpuCodegen(const std::string& id) { ext_func_id_ = id; }

  runtime::Module CreateVsiNpuModule(const ObjectRef& ref) { return runtime::Module(); }

  std::string JIT(const std::vector<Output>& out) {
    return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body_, const_array_name_, out);
  }

 private:
  std::string ext_func_id_ = {""};
  Array<Var> ext_func_args_;
  std::vector<std::string> buf_decl_;
  std::string const_array_name_;
  std::vector<std::string> ext_func_body_;
};

runtime::Module VsiNpuCompiler(const ObjectRef& ref) {
  VsiNpuCodegen vsi_npu_src{"VSI NPU Code Gen"};
  return vsi_npu_src.CreateVsiNpuModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.vsi_npu").set_body_typed(VsiNpuCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
