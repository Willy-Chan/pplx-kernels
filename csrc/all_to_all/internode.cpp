
#include <nvshmem.h>

#include <cassert>
#include <cstdint>

#include "all_to_all/internode.h"
#include "core/utils.h"




// TODO: DELETE THESE IMPORTS LATER
#include <vector>

using namespace pplx;

AllToAllInterNode::AllToAllInterNode(
    size_t maxNumTokens,
    size_t numExperts,
    size_t expertsPerToken,
    unsigned rank,
    unsigned worldSize,
    unsigned dpSize,
    size_t hiddenDim,
    size_t hiddenDimBytes,
    size_t hiddenDimScaleBytes,
    uint64_t* extNumTokensBuffer,
    uint64_t* extNumDispatchRecvBuffer,         // PART 5
    uint64_t * extCombineSignalBuffer,
    uint64_t * extCombineSyncBuffer
    // std::byte * xDispatchIn1,
    // std::byte * xDispatchOut1,
    // std::byte * xCombineIn1,
    // std::byte * xCombineOut1
)
    : AllToAll(
          maxNumTokens,
          numExperts,
          expertsPerToken,
          rank,
          worldSize,
          dpSize,
          hiddenDim,
          hiddenDimBytes,
          hiddenDimScaleBytes
      ),
      maxBatchTokens(numLocalExperts * numDPGroups * maxNumTokens) {
  // Buffers for token counts.
  numTokensPerDP = mallocZeroBuffer<uint32_t>(numLocalExperts * numDPGroups);

  if (extNumTokensBuffer) {
    // TODO: delete after, validate that the pointer is a valid device pointer.
    cudaPointerAttributes attr;
    auto err = cudaPointerGetAttributes(&attr, extNumTokensBuffer);
    PPLX_ASSERT(extNumTokensBuffer != nullptr && err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "numTokensBuffer is not a valid device pointer");
    printf("[AllToAllInterNode] numTokensBuffer is a valid device pointer at %p with memory type %d (0=Host, 1=Device, 2=Managed) and in device %d\n", extNumTokensBuffer, attr.type, attr.device);
    
    numTokensBuffer = extNumTokensBuffer;
  } else {
    numTokensBuffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * numLocalExperts * numDPGroups);
    PPLX_ASSERT(numTokensBuffer != nullptr, "failed to allocate numTokensBuffer");
    cudaMemset(numTokensBuffer, 0, sizeof(uint64_t) * numLocalExperts * numDPGroups);
  }

  if (extNumDispatchRecvBuffer) {
    cudaPointerAttributes attr;
    auto err = cudaPointerGetAttributes(&attr, extNumDispatchRecvBuffer);
    PPLX_ASSERT(extNumDispatchRecvBuffer != nullptr && err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "extNumDispatchRecvBuffer is not a valid device pointer");
    printf("[AllToAllInterNode] extNumDispatchRecvBuffer is a valid device pointer at %p with memory type %d (0=Host, 1=Device, 2=Managed) and in device %d\n", extNumDispatchRecvBuffer, attr.type, attr.device);
    
    numDispatchRecvBuffer = extNumDispatchRecvBuffer;
  } else {
    numDispatchRecvBuffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * numLocalExperts * numDPGroups);
    PPLX_ASSERT(numDispatchRecvBuffer != nullptr, "failed to allocate numDispatchRecvBuffer");
    cudaMemset(numDispatchRecvBuffer, 0, sizeof(uint64_t) * numLocalExperts * numDPGroups);
  }

  if (extCombineSignalBuffer) {
    cudaPointerAttributes attr;
    auto err = cudaPointerGetAttributes(&attr, extCombineSignalBuffer);
    PPLX_ASSERT(extCombineSignalBuffer != nullptr && err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "extCombineSignalBuffer is not a valid device pointer");
    printf("[AllToAllInterNode] extCombineSignalBuffer is a valid device pointer at %p with memory type %d (0=Host, 1=Device, 2=Managed) and in device %d\n", extCombineSignalBuffer, attr.type, attr.device);
    
    combineSignalBuffer = extCombineSignalBuffer;
  } else {
    combineSignalBuffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * maxNumTokens);
    PPLX_ASSERT(combineSignalBuffer != nullptr, "failed to allocate combineSignalBuffer");
    cudaMemset(combineSignalBuffer, 0, sizeof(uint64_t) * maxNumTokens);
  }
  
  if (extCombineSyncBuffer) {
    cudaPointerAttributes attr;
    auto err = cudaPointerGetAttributes(&attr, extCombineSyncBuffer);
    PPLX_ASSERT(extCombineSyncBuffer != nullptr && err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "extCombineSyncBuffer is not a valid device pointer");
    printf("[AllToAllInterNode] extCombineSyncBuffer is a valid device pointer at %p with memory type %d (0=Host, 1=Device, 2=Managed) and in device %d\n", extCombineSyncBuffer, attr.type, attr.device);
    
    combineSyncBuffer = extCombineSyncBuffer;
  } else {
    combineSyncBuffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * worldSize);
    PPLX_ASSERT(combineSyncBuffer != nullptr, "failed to allocate combineSyncBuffer");
    cudaMemset(combineSyncBuffer, 0, sizeof(uint64_t) * worldSize);
  }

  // Buffers for dispatch.
  const size_t perTokenBytes =
      round_up<size_t>(hiddenDimBytes + hiddenDimScaleBytes + sizeof(uint32_t), 16);
  xDispatchIn = (std::byte *)nvshmem_malloc(maxNumTokens * perTokenBytes);
  PPLX_ASSERT(xDispatchIn != nullptr, "failed to allocate xDispatchIn");
  xDispatchOut = (std::byte *)nvshmem_malloc(maxBatchTokens * perTokenBytes);
  PPLX_ASSERT(xDispatchOut != nullptr, "failed to allocate xDispatchOut");

  // Buffers for combine. The allocations are a bit wider to accommodate all
  // possible data types (primarily float for testing and bfloat16 for prod).
  xCombineIn = (std::byte *)nvshmem_malloc(maxBatchTokens * hiddenDim * sizeof(float));
  PPLX_ASSERT(xCombineIn != nullptr, "failed to allocate xCombineIn");
  xCombineOut = (std::byte *)nvshmem_malloc(maxNumTokens * numExperts * hiddenDim * sizeof(float));
  PPLX_ASSERT(xCombineOut != nullptr, "failed to allocate xCombineOut");

  // Buffers for token tracking.
  sourceIndex = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceExpert = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceOffset = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceGroup = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  sourceToken = mallocZeroBuffer<uint32_t>(maxBatchTokens);
  tokenIndex = mallocZeroBuffer<uint32_t>(1);
}

AllToAllInterNode::~AllToAllInterNode() {
  CUDACHECK(cudaFree(numTokensPerDP));
  nvshmem_free(numTokensBuffer);
  nvshmem_free(numDispatchRecvBuffer);
  nvshmem_free(combineSignalBuffer);
  nvshmem_free(combineSyncBuffer);
  nvshmem_free(xDispatchIn);
  nvshmem_free(xDispatchOut);
  nvshmem_free(xCombineIn);
  nvshmem_free(xCombineOut);

  CUDACHECK(cudaFree(sourceIndex));
  CUDACHECK(cudaFree(sourceExpert));
  CUDACHECK(cudaFree(sourceOffset));
  CUDACHECK(cudaFree(sourceGroup));
  CUDACHECK(cudaFree(sourceToken));
  CUDACHECK(cudaFree(tokenIndex));
}
