
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
    uint64_t* extNumDispatchRecvBuffer,         // PART 5
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
    const size_t perTokenBytes =
    round_up<size_t>(hiddenDimBytes + hiddenDimScaleBytes + sizeof(uint32_t), 16);

    // Buffers for token counts.
    numTokensPerDP = mallocZeroBuffer<uint32_t>(numLocalExperts * numDPGroups);

    // numTokensBuffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * numLocalExperts * numDPGroups);
    cudaPointerAttributes attr;
    auto err = cudaPointerGetAttributes(&attr, extNumTokensBuffer);
    PPLX_ASSERT(extNumTokensBuffer != nullptr && err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "numTokensBuffer is not a valid device pointer");    
    numTokensBuffer = (uint64_t *)extNumTokensBuffer;
    cudaMemset(numTokensBuffer, 0, sizeof(uint64_t) * numLocalExperts * numDPGroups);

    // numDispatchRecvBuffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * numLocalExperts * numDPGroups);
    err = cudaPointerGetAttributes(&attr, extNumDispatchRecvBuffer);
    PPLX_ASSERT(extNumDispatchRecvBuffer != nullptr && err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "extNumDispatchRecvBuffer is not a valid device pointer");
    numDispatchRecvBuffer = (uint64_t *)extNumDispatchRecvBuffer;
    cudaMemset(numDispatchRecvBuffer, 0, sizeof(uint64_t) * numLocalExperts * numDPGroups);


    // combineSignalBuffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * maxNumTokens);
    err = cudaPointerGetAttributes(&attr, extCombineSignalBuffer);
    PPLX_ASSERT(extCombineSignalBuffer != nullptr && err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "extCombineSignalBuffer is not a valid device pointer");    
    combineSignalBuffer = (uint64_t *)extCombineSignalBuffer;
    cudaMemset(combineSignalBuffer, 0, sizeof(uint64_t) * maxNumTokens);
  
    // combineSyncBuffer = (uint64_t *)nvshmem_malloc(sizeof(uint64_t) * worldSize);
    err = cudaPointerGetAttributes(&attr, extCombineSyncBuffer);
    PPLX_ASSERT(extCombineSyncBuffer != nullptr && err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "extCombineSyncBuffer is not a valid device pointer");    
    combineSyncBuffer = (uint64_t *)extCombineSyncBuffer;
    cudaMemset(combineSyncBuffer, 0, sizeof(uint64_t) * worldSize);

    err = cudaPointerGetAttributes(&attr, extXDispatchIn);
    PPLX_ASSERT(extXDispatchIn != nullptr, "failed to allocate xDispatchIn");
    PPLX_ASSERT(err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "extXDispatchIn is not a valid device pointer");
    // printf("[AllToAllInterNode] extXDispatchIn allocated at %p (memType %d, device %d)\n", extXDispatchIn, attr.type, attr.device);
    //   xDispatchIn = (std::byte *)nvshmem_malloc(maxNumTokens * perTokenBytes);
    xDispatchIn = (std::byte *)extXDispatchIn;

    err = cudaPointerGetAttributes(&attr, extXDispatchOut);
    PPLX_ASSERT(extXDispatchOut != nullptr, "failed to allocate extXDispatchOut");
    PPLX_ASSERT(err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "extXDispatchOut is not a valid device pointer");
    // printf("[AllToAllInterNode] extXDispatchOut allocated at %p (memType %d, device %d)\n", extXDispatchOut, attr.type, attr.device);
    //   xDispatchOut = (std::byte *)nvshmem_malloc(maxBatchTokens * perTokenBytes);
    xDispatchOut = (std::byte *)extXDispatchOut;

    err = cudaPointerGetAttributes(&attr, extXCombineIn);
    PPLX_ASSERT(extXCombineIn != nullptr, "failed to allocate extXCombineIn");
    PPLX_ASSERT(err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "extXCombineIn is not a valid device pointer");
    // printf("[AllToAllInterNode] extXCombineIn allocated at %p (memType %d, device %d)\n", extXCombineIn, attr.type, attr.device);
    //   xCombineIn = (std::byte *)nvshmem_malloc(maxBatchTokens * hiddenDim * sizeof(float));
    xCombineIn = (std::byte *)extXCombineIn;

    err = cudaPointerGetAttributes(&attr, extXCombineOut);
    PPLX_ASSERT(extXCombineOut != nullptr, "failed to allocate extXCombineOut");
    PPLX_ASSERT(err == cudaSuccess && (attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged),
                "extXCombineOut is not a valid device pointer");
    // printf("[AllToAllInterNode] extXCombineOut allocated at %p (memType %d, device %d)\n", extXCombineOut, attr.type, attr.device);
    //   xCombineOut = (std::byte *)nvshmem_malloc(maxNumTokens * numExperts * hiddenDim * sizeof(float));
    xCombineOut = (std::byte *)extXCombineOut;


    // ------------------------------------------------------------
        // Diagnostics: print expected byte sizes that the original
        // nvshmem_malloc calls would have allocated.  We do this once
        // on PE/rank 0 so logs are readable.

        if (rank == 0) {
            printf("===== Expected allocation sizes (bytes) =====\n");
            printf("numTokensBuffer       : %zu\n", sizeof(uint64_t) * numLocalExperts * numDPGroups);
            printf("numDispatchRecvBuffer : %zu\n", sizeof(uint64_t) * numLocalExperts * numDPGroups);
            printf("combineSignalBuffer   : %zu\n", sizeof(uint64_t) * maxNumTokens);
            printf("combineSyncBuffer     : %zu\n", sizeof(uint64_t) * worldSize);
            printf("xDispatchIn           : %zu\n", maxNumTokens  * perTokenBytes);
            printf("xDispatchOut          : %zu\n", maxBatchTokens * perTokenBytes);
            printf("xCombineIn            : %zu\n", maxBatchTokens * hiddenDim * sizeof(float));
            printf("xCombineOut           : %zu\n", maxNumTokens  * numExperts * hiddenDim * sizeof(float));
            printf("===========================================\n");

            printf("numTokensBuffer       -> %p\n", numTokensBuffer      );
            printf("numDispatchRecvBuffer -> %p\n", numDispatchRecvBuffer);
            printf("combineSignalBuffer   -> %p\n", combineSignalBuffer  );
            printf("combineSyncBuffer     -> %p\n", combineSyncBuffer    );
            printf("xDispatchIn           -> %p\n", xDispatchIn          );
            printf("xDispatchOut          -> %p\n", xDispatchOut         );
            printf("xCombineIn            -> %p\n", xCombineIn           );
            printf("xCombineOut           -> %p\n", xCombineOut  );        
        }
    // ------------------------------------------------------------



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
//   nvshmem_free(numTokensBuffer);
//   nvshmem_free(numDispatchRecvBuffer);
//   nvshmem_free(combineSignalBuffer);
//   nvshmem_free(combineSyncBuffer);
//   nvshmem_free(xDispatchIn);
//   nvshmem_free(xDispatchOut);
//   nvshmem_free(xCombineIn);
//   nvshmem_free(xCombineOut);

  CUDACHECK(cudaFree(sourceIndex));
  CUDACHECK(cudaFree(sourceExpert));
  CUDACHECK(cudaFree(sourceOffset));
  CUDACHECK(cudaFree(sourceGroup));
  CUDACHECK(cudaFree(sourceToken));
  CUDACHECK(cudaFree(tokenIndex));
}
