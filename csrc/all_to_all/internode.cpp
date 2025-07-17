
#include <nvshmem.h>

#include <cassert>
#include <cstdint>

#include "all_to_all/internode.h"
#include "core/utils.h"

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
    uint64_t* extNumDispatchRecvBuffer,
    uint64_t* extCombineSignalBuffer,
    uint64_t* extCombineSyncBuffer,
    std::byte* extXDispatchIn,
    std::byte* extXDispatchOut,
    std::byte* extXCombineIn,
    std::byte* extXCombineOut
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

    // Buffers for dispatch.
    const size_t perTokenBytes = round_up<size_t>(hiddenDimBytes + hiddenDimScaleBytes + sizeof(uint32_t), 16);

    // Buffers for token counts.
    numTokensPerDP = mallocZeroBuffer<uint32_t>(numLocalExperts * numDPGroups);

    cudaPointerAttributes attr;
    auto err = cudaPointerGetAttributes(&attr, extNumTokensBuffer);
    PPLX_ASSERT(extNumTokensBuffer != nullptr && err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "numTokensBuffer is not a valid device pointer");    
    numTokensBuffer = (uint64_t *)extNumTokensBuffer;
    cudaMemset(numTokensBuffer, 0, sizeof(uint64_t) * numLocalExperts * numDPGroups);

    err = cudaPointerGetAttributes(&attr, extNumDispatchRecvBuffer);
    PPLX_ASSERT(extNumDispatchRecvBuffer != nullptr && err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "extNumDispatchRecvBuffer is not a valid device pointer");
    numDispatchRecvBuffer = (uint64_t *)extNumDispatchRecvBuffer;
    cudaMemset(numDispatchRecvBuffer, 0, sizeof(uint64_t) * numLocalExperts * numDPGroups);

    err = cudaPointerGetAttributes(&attr, extCombineSignalBuffer);
    PPLX_ASSERT(extCombineSignalBuffer != nullptr && err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "extCombineSignalBuffer is not a valid device pointer");    
    combineSignalBuffer = (uint64_t *)extCombineSignalBuffer;
    cudaMemset(combineSignalBuffer, 0, sizeof(uint64_t) * maxNumTokens);
  
    err = cudaPointerGetAttributes(&attr, extCombineSyncBuffer);
    PPLX_ASSERT(extCombineSyncBuffer != nullptr && err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "extCombineSyncBuffer is not a valid device pointer");    
    combineSyncBuffer = (uint64_t *)extCombineSyncBuffer;
    cudaMemset(combineSyncBuffer, 0, sizeof(uint64_t) * worldSize);

    err = cudaPointerGetAttributes(&attr, extXDispatchIn);
    PPLX_ASSERT(extXDispatchIn != nullptr, "failed to allocate xDispatchIn");
    PPLX_ASSERT(err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "extXDispatchIn is not a valid device pointer");
    xDispatchIn = (std::byte *)extXDispatchIn;

    err = cudaPointerGetAttributes(&attr, extXDispatchOut);
    PPLX_ASSERT(extXDispatchOut != nullptr, "failed to allocate extXDispatchOut");
    PPLX_ASSERT(err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "extXDispatchOut is not a valid device pointer");
    xDispatchOut = (std::byte *)extXDispatchOut;

    err = cudaPointerGetAttributes(&attr, extXCombineIn);
    PPLX_ASSERT(extXCombineIn != nullptr, "failed to allocate extXCombineIn");
    PPLX_ASSERT(err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "extXCombineIn is not a valid device pointer");
    xCombineIn = (std::byte *)extXCombineIn;

    err = cudaPointerGetAttributes(&attr, extXCombineOut);
    PPLX_ASSERT(extXCombineOut != nullptr, "failed to allocate extXCombineOut");
    PPLX_ASSERT(err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "extXCombineOut is not a valid device pointer");
    xCombineOut = (std::byte *)extXCombineOut;

    /*
     * CRITICAL: Must call nvshmem_init() before nvshmemx_collective_launch() to avoid
     * CUDA illegal memory access errors.
     * 
     * ROOT CAUSE: NVSHMEM has two separate device state structures:
     * 1. nvshmemi_device_only_state (host-side scratch state) - initialized by check_state_and_init_d()
     * 2. nvshmemi_device_state_d (device-side constant memory) - populated by nvshmemi_update_device_state()
     * 
     * The collective launch only initializes the host-side scratch state (streams, events, device attributes)
     * but does NOT populate the device-side constant memory that all GPU NVSHMEM operations require.
     * 
     * When cudaLaunchCooperativeKernel() tries to access NVSHMEM memory, it dereferences the
     * __constant__ nvshmemi_device_state_d structure, which remains uninitialized until
     * nvshmem_init() calls nvshmemi_update_device_state() to copy the host state to device (with PE info, etc.).
     * 
     * DESIGN RATIONALE: This separation allows avoids unnecessary device memory transfers during bootstrap,
     * but the tradeoff is that it requires explicit initialization before any device-side NVSHMEM operations (we don't have to interact with the devuce if we don't need to).
     */
     nvshmem_init();


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
  CUDACHECK(cudaFree(sourceIndex));
  CUDACHECK(cudaFree(sourceExpert));
  CUDACHECK(cudaFree(sourceOffset));
  CUDACHECK(cudaFree(sourceGroup));
  CUDACHECK(cudaFree(sourceToken));
  CUDACHECK(cudaFree(tokenIndex));
}
