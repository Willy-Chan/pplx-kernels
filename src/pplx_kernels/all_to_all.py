# pyright: reportCallIssue=false

from collections.abc import Callable
from typing import Any

import torch

import nvshmem.core as nvshmem
import ctypes

from .ops import _ops


class AllToAll:
    def __init__(
        self,
        ptr: Any,
        combine_fn: Callable,
        dispatch_fn: Callable,
        has_scales: bool,
        *,
        numTokensBuffer = None,
        numDispatchRecvBuffer = None,
        combineSignalBuffer = None,
        combineSyncBuffer = None,
        xDispatchIn =  None,
        xDispatchOut = None,
        xCombineIn = None,
        xCombineOut = None
    ) -> None:
        self._ptr = ptr
        self._combine_fn = combine_fn
        self._dispatch_fn = dispatch_fn
        self._has_scales = has_scales

        # TODO: Added these extra attributes because we need a handle to the nvshmem4py device tensors, so that way we can free them when calling destroy()
        self.numTokensBuffer = numTokensBuffer
        self.numDispatchRecvBuffer = numDispatchRecvBuffer
        self.combineSignalBuffer = combineSignalBuffer
        self.combineSyncBuffer = combineSyncBuffer
        self.xDispatchIn = xDispatchIn 
        self.xDispatchOut = xDispatchOut
        self.xCombineIn = xCombineIn
        self.xCombineOut = xCombineOut

    def __del__(self) -> None:
        self.destroy()

        # freeing the externally-provided device buffers
        nvshmem.free_tensor(self.numTokensBuffer)
        nvshmem.free_tensor(self.numDispatchRecvBuffer)
        nvshmem.free_tensor(self.combineSignalBuffer)
        nvshmem.free_tensor(self.combineSyncBuffer)
        nvshmem.free_tensor(self.xDispatchIn)
        nvshmem.free_tensor(self.xDispatchOut)
        nvshmem.free_tensor(self.xCombineIn)
        nvshmem.free_tensor(self.xCombineOut)


    def dispatch(
        self,
        out_expert_num_tokens: torch.Tensor,
        out_expert_x: torch.Tensor,
        out_expert_x_scale: torch.Tensor | None,
        dp_x: torch.Tensor,
        dp_x_scale: torch.Tensor | None,
        indices: torch.Tensor,
        bound_m: torch.Tensor | None,
        do_send: bool = True,
        do_recv: bool = True,
    ) -> None:
        assert self._ptr is not None

        if self._has_scales:
            assert out_expert_x_scale is not None
            assert dp_x_scale is not None
        else:
            assert out_expert_x_scale is None
            assert dp_x_scale is None

        self._dispatch_fn(
            self._ptr,
            out_expert_num_tokens,
            out_expert_x,
            out_expert_x_scale,
            dp_x,
            dp_x_scale,
            indices,
            bound_m,
            do_send,
            do_recv,
        )

    def combine(
        self,
        out_tokens: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
        expert_y: torch.Tensor,
        bound_m: torch.Tensor | None,
        do_send: bool = True,
        do_recv: bool = True,
    ) -> None:
        assert self._ptr is not None
        self._combine_fn(
            self._ptr,
            out_tokens,
            indices,
            weights,
            expert_y,
            bound_m,
            do_send,
            do_recv,
        )

    def destroy(self) -> None:
        if self._ptr is not None:
            _ops.all_to_all_destroy(self._ptr)
            self._ptr = None

    @classmethod
    def intranode(
        cls,
        max_num_tokens: int,
        num_experts: int,
        experts_per_token: int,
        rank: int,
        world_size: int,
        dp_size: int,
        hidden_dim: int,
        hidden_dim_bytes: int,
        hidden_dim_scale_bytes: int,
        group_name: str = "default",
    ) -> "AllToAll":
        assert world_size % dp_size == 0
        assert world_size // dp_size > 1

        has_scales = hidden_dim_scale_bytes > 0

        ptr = _ops.all_to_all_intranode_create(
            max_num_tokens,
            num_experts,
            experts_per_token,
            rank,
            world_size,
            dp_size,
            hidden_dim,
            hidden_dim_bytes,
            hidden_dim_scale_bytes,
            group_name,
        )
        assert ptr != 0

        return cls(
            ptr,
            _ops.all_to_all_intranode_combine,
            _ops.all_to_all_intranode_dispatch,
            has_scales,
        )

    @classmethod
    def internode(
        cls,
        max_num_tokens: int,
        num_experts: int,
        experts_per_token: int,
        rank: int,
        world_size: int,
        dp_size: int,
        hidden_dim: int,
        hidden_dim_bytes: int,
        hidden_dim_scale_bytes: int,
    ) -> "AllToAll":
        assert world_size % dp_size == 0
        assert world_size // dp_size > 1

        has_scales = hidden_dim_scale_bytes > 0
        # numLocalExperts = num_experts // world_size
        # numDPGroups = world_size // dp_size
        def ceil_div(x: int, y: int) -> int:
            return (x + y - 1) // y

        numLocalExperts = ceil_div(num_experts, world_size)
        numDPGroups     = ceil_div(world_size,  dp_size)

        # COMMAND TO REINSTALL PPLX KERNELS AND RUN THE TESTS AGAIN
        # alias ben='pip uninstall pplx-kernels && TORCH_CUDA_ARCH_LIST=9.0a+PTX python3 setup.py bdist_wheel && pip install dist/*.whl && torchrun --nproc-per-node 4 /lustre/fs1/portfolios/coreai/projects/coreai_libraries_nvshmem/wilchan/pplx/bin/pytest -svx --tb=short tests tests/test_all_to_all.py::test_all_to_all_4_gpu'
        # alias ben_only_torchrun='torchrun --nproc-per-node 4 /lustre/fs1/portfolios/coreai/projects/coreai_libraries_nvshmem/wilchan/pplx/bin/pytest -svx --tb=short tests tests/test_all_to_all.py::test_all_to_all_4_gpu'
        # pip uninstall pplx-kernels && TORCH_CUDA_ARCH_LIST=9.0a+PTX python3 setup.py bdist_wheel && pip install dist/*.whl && torchrun --nproc-per-node 4 /lustre/fs1/portfolios/coreai/projects/coreai_libraries_nvshmem/wilchan/pplx/bin/pytest -svx --tb=short tests tests/test_all_to_all.py::test_all_to_all_4_gpu

        # TODO: convention to get element size/numbytes of torch.uint64?
        # torch_uint64_element_size = torch.tensor([], dtype=torch.uint64).element_size()
        # numTokensBuffer = nvshmem.memory.buffer(numLocalExperts * numDPGroups * torch_uint64_element_size)    # really should be uint64 which is just 8 bytes
        # numDispatchRecvBuffer = nvshmem.memory.buffer(numLocalExperts * numDPGroups * torch_uint64_element_size)
        # combineSignalBuffer = nvshmem.memory.buffer(max_num_tokens * torch_uint64_element_size)
        # combineSyncBuffer = nvshmem.memory.buffer(world_size * torch_uint64_element_size)

        numTokensBuffer = nvshmem.interop.torch.tensor((numLocalExperts * numDPGroups,), dtype=torch.int64) # TODO: SHOULD BE uint64!!!
        numTokensBuffer[:] = 0
        numDispatchRecvBuffer = nvshmem.interop.torch.tensor((numLocalExperts * numDPGroups,), dtype=torch.int64)
        numDispatchRecvBuffer[:] = 0
        combineSignalBuffer = nvshmem.interop.torch.tensor((max_num_tokens,), dtype=torch.int64)
        combineSignalBuffer[:] = 0
        combineSyncBuffer = nvshmem.interop.torch.tensor((world_size,), dtype=torch.int64)
        combineSyncBuffer[:] = 0

        def round_up(x: int, y: int) -> int:
            """Round up x to the nearest multiple of y."""
            return ((x + y - 1) // y) * y

        # PART 1
        per_token_bytes = round_up(hidden_dim_bytes + hidden_dim_scale_bytes + 4, 16)  # TODO: + 4 for uint32_t
        max_batch_tokens = numLocalExperts * numDPGroups * max_num_tokens


        # TODO: Should we be using torch.tensor for these std::byte arrays? Are there performance considerations?
        # xDispatchIn = nvshmem.memory.buffer(max_num_tokens * per_token_bytes * 1)   # TODO: + 1 for uint8_t
        # xDispatchOut = nvshmem.memory.buffer(max_batch_tokens * per_token_bytes * 1)

        xDispatchIn = nvshmem.interop.torch.tensor( (max_num_tokens * per_token_bytes,), dtype=torch.uint8 )
        xDispatchOut = nvshmem.interop.torch.tensor( (max_batch_tokens * per_token_bytes,), dtype=torch.uint8 )

        # xCombineIn = nvshmem.memory.buffer(max_batch_tokens * hidden_dim * 4)   # should be float32 right??
        # xCombineOut = nvshmem.memory.buffer(max_num_tokens * num_experts * hidden_dim * 4)
        xCombineIn = nvshmem.interop.torch.tensor( (max_batch_tokens * hidden_dim,), dtype=torch.float32 )
        xCombineOut = nvshmem.interop.torch.tensor( (max_num_tokens * num_experts * hidden_dim,), dtype=torch.float32 )

        # if rank == 1:
        #     print(f"[PYTHON] numTokensBuffer = 0x{numTokensBuffer.data_ptr():x}")
        #     print(f"[PYTHON] numDispatchRecvBuffer = 0x{numDispatchRecvBuffer.data_ptr():x}")
        #     print(f"[PYTHON] combineSignalBuffer = 0x{combineSignalBuffer.data_ptr():x}")
        #     print(f"[PYTHON] combineSyncBuffer = 0x{combineSyncBuffer.data_ptr():x}")
        #     print(f"[PYTHON] xDispatchIn = 0x{xDispatchIn.data_ptr():x}")
        #     print(f"[PYTHON] xDispatchOut = 0x{xDispatchOut.data_ptr():x}")
        #     print(f"[PYTHON] xCombineIn = 0x{xCombineIn.data_ptr():x}")
        #     print(f"[PYTHON] xCombineOut = 0x{xCombineOut.data_ptr():x}")

        ptr = _ops.all_to_all_internode_create(
            max_num_tokens,
            num_experts,
            experts_per_token,
            rank,
            world_size,
            dp_size,
            hidden_dim,
            hidden_dim_bytes,
            hidden_dim_scale_bytes,

            numTokensBuffer,
            numDispatchRecvBuffer,
            combineSignalBuffer,
            combineSyncBuffer,            # PART 2
            xDispatchIn,
            xDispatchOut,
            xCombineIn,
            xCombineOut
        )
        assert ptr != 0

        # ---------------- Size sanity checks -----------------
        # if rank == 0:
        #     print("===== NVSHMEM Buffer Sizes (bytes) =====")
        #     print(f"numTokensBuffer       : {torch.numel(numTokensBuffer) * numTokensBuffer.element_size()}")
        #     print(f"numDispatchRecvBuffer : {torch.numel(numDispatchRecvBuffer) * numDispatchRecvBuffer.element_size()}")
        #     print(f"combineSignalBuffer   : {torch.numel(combineSignalBuffer  ) * combineSignalBuffer  .element_size()}")
        #     print(f"combineSyncBuffer     : {torch.numel(combineSyncBuffer    ) * combineSyncBuffer    .element_size()}")
        #     print(f"xDispatchIn           : {torch.numel(xDispatchIn          ) * xDispatchIn          .element_size()}")
        #     print(f"xDispatchOut          : {torch.numel(xDispatchOut         ) * xDispatchOut         .element_size()}")
        #     print(f"xCombineIn            : {torch.numel(xCombineIn           ) * xCombineIn           .element_size()}")
        #     print(f"xCombineOut           : {torch.numel(xCombineOut          ) * xCombineOut          .element_size()}")
        #     print("========================================")

        # ------------------------------------------------------

        return cls(
            ptr,
            _ops.all_to_all_internode_combine,
            _ops.all_to_all_internode_dispatch,
            has_scales,
            numTokensBuffer = numTokensBuffer,
            numDispatchRecvBuffer = numDispatchRecvBuffer,
            combineSignalBuffer = combineSignalBuffer,
            combineSyncBuffer = combineSyncBuffer,
            xDispatchIn = xDispatchIn,
            xDispatchOut = xDispatchOut,
            xCombineIn = xCombineIn,
            xCombineOut = xCombineOut
        )

