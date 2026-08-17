// Copyright 2026 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_TRANSPORT_ADAPTER_H_
#define THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_TRANSPORT_ADAPTER_H_

#include <cstddef>
#include <cstdint>

namespace tpu_raiden {
namespace transport {

// `Handle` uniquely identifies a transport request.
using Handle = uint32_t;

struct Request {
  uint8_t socket_opcode;
  uint8_t* laddr;
  uint8_t* raddr;
  size_t len;

  uint32_t count_or_size;
  uint32_t local_id;   // Block: layer_idx / local block ID |
                       // Buffer: dst/src shard index.
  uint32_t remote_id;  // Block: sender node_id / remote block ID | Buffer:
                       // dst/src byte offset.
  int layer_idx;
  uint32_t request_id;
  uint64_t uuid;
  int parallelism;
  uint8_t major_order;
};

}  // namespace transport
}  // namespace tpu_raiden

#endif  // THIRD_PARTY_TPU_RAIDEN_TPU_SYNC_TRANSPORT_TRANSPORT_ADAPTER_H_
