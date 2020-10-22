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
 * \file src/runtime/contrib/dnnl/dnnl.cc
 * \brief TVM compatible wrappers for dnnl kernels.
 */

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "vsi_kernel.h"
#include <VX/vx_khr_cnn.h>
#include "vsi_nn_pub.h"


namespace tvm {
namespace runtime {
namespace contrib {



extern "C" void vsi_npu_add(float* data1, float* data2, float* out, int n, int c, int h, int w)
{
#define _CHECK_TENSOR_ID( id, lbl )      do {\
    if( VSI_NN_TENSOR_ID_NA == id ) {\
        printf("CHECK TENSOR ID %d", __LINE__);\
        goto lbl;\
        }\
    } while(0)

#define _CHECK_PTR( ptr, lbl )      do {\
    if( NULL == ptr ) {\
        printf("CHECK PTR %d", __LINE__);\
        goto lbl;\
    }\
} while(0)

#define _CHECK_STATUS( stat, lbl )  do {\
    if( VX_SUCCESS != stat ) {\
        printf("CHECK STATUS %d", __LINE__);\
        goto lbl;\
    }\
} while(0)


#define _CONST_NODE_NUM     0
#define _TENSOR_NUM         (3)
#define _NODE_NUM           1
  vx_status             status;
  vsi_nn_graph_t      * graph;
  vsi_nn_context_t      ctx;
  vsi_nn_node_t       * node;
  vsi_nn_tensor_attr_t  attr;
  vsi_nn_tensor_id_t    input[2];
  vsi_nn_tensor_id_t    output[1];
  vsi_nn_tensor_t     * tensor;
  vx_float32          * vsi_out = NULL;

  status = VSI_FAILURE;

  ctx = vsi_nn_CreateContext();
  _CHECK_PTR( ctx, final );

  graph = vsi_nn_CreateGraph( ctx, _TENSOR_NUM, _NODE_NUM );
  _CHECK_PTR( graph, final );

  vsi_nn_SetGraphInputs( graph, NULL, 2 );
  vsi_nn_SetGraphOutputs( graph, NULL, 1 );

  node = vsi_nn_AppendNode( graph, VSI_NN_OP_ADD, NULL );
  _CHECK_PTR( node, final );

  memset( &attr, 0, sizeof( vsi_nn_tensor_attr_t ) );
  attr.size[0] = c;
  attr.size[1] = h;
  attr.size[2] = w;
  attr.dim_num = 3;
  attr.vtl = FALSE;
  attr.is_const = FALSE;
  attr.dtype.vx_type = VSI_NN_TYPE_FLOAT32;
  attr.dtype.qnt_type = VSI_NN_QNT_TYPE_NONE;

  input[0] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, &attr, (vx_uint8*)data1 );
  input[1] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, &attr, (vx_uint8*)data2 );
  output[0] = vsi_nn_AddTensor( graph, VSI_NN_TENSOR_ID_AUTO, &attr, NULL );
  _CHECK_TENSOR_ID( input[0], final );
  _CHECK_TENSOR_ID( input[1], final );
  _CHECK_TENSOR_ID( output[0], final );

  node->input.tensors[0] = input[0];
  node->input.tensors[1] = input[1];
  node->output.tensors[0] = output[0];

  graph->input.tensors[0] = input[0];
  graph->input.tensors[1] = input[1];
  graph->output.tensors[0] = output[0];

  status = vsi_nn_SetupGraph( graph, TRUE );
  _CHECK_STATUS( status, final );
  status = vsi_nn_VerifyGraph( graph );
  _CHECK_STATUS( status, final );
  status = vsi_nn_RunGraph( graph );
  _CHECK_STATUS( status, final );

  tensor = vsi_nn_GetTensor( graph, output[0] );
  _CHECK_PTR( tensor, final );
  vsi_out = (vx_float32*)vsi_nn_ConvertTensorToData( graph, tensor );
  _CHECK_PTR( vsi_out, final );

  memcpy(out, vsi_out, sizeof(float) * c * h *w);

  final:
  if(out != NULL)
  {
    free(vsi_out);
    out = NULL;
  }
  vsi_nn_ReleaseGraph( &graph );
  vsi_nn_ReleaseContext( &ctx );
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
