# pyright: reportCallIssue=false

from collections.abc import Callable
from typing import Any

import torch

import nvshmem.core as nvshmem

from .ops import _ops


class AllToAll:
    def __init__(
        self,
        ptr: Any,
        combine_fn: Callable,
        dispatch_fn: Callable,
        has_scales: bool,
    ) -> None:
        self._ptr = ptr
        self._combine_fn = combine_fn
        self._dispatch_fn = dispatch_fn
        self._has_scales = has_scales

    def __del__(self) -> None:
        self.destroy()

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

        numLocalExperts = num_experts // world_size
        numDPGroups = world_size // dp_size

        # TODO: should really be uint64.....        NOW PASS IN THE POINTERS USING tensor.data_ptr()


        # TODO: just replace it with torch.buffer....
        numTokensBuffer = nvshmem.memory.buffer(numLocalExperts * numDPGroups * 8)    # really should be uint64 which is just 8 bytes
        numDispatchRecvBuffer = nvshmem.memory.buffer(numLocalExperts * numDPGroups * 8)
        combineSignalBuffer = nvshmem.memory.buffer(max_num_tokens * 8)
        combineSyncBuffer = nvshmem.memory.buffer(world_size * 8)


        numTokensBuffer = nvshmem.memory.buffer(numLocalExperts * numDPGroups * 8)    # really should be uint64 but we do *2 for the uint32 since uint64 not supported
        numDispatchRecvBuffer = nvshmem.memory.buffer(numLocalExperts * numDPGroups * 8)
        combineSignalBuffer = nvshmem.memory.buffer(max_num_tokens * 8)
        combineSyncBuffer = nvshmem.memory.buffer(world_size * 8)

        # COMMAND TO REINSTALL PPLX KERNELS AND RUN THE TESTS AGAIN
        # alias ben='pip uninstall pplx-kernels && TORCH_CUDA_ARCH_LIST=9.0a+PTX python3 setup.py bdist_wheel && pip install dist/*.whl && torchrun --nproc-per-node 4 /lustre/fs1/portfolios/coreai/projects/coreai_libraries_nvshmem/wilchan/pplx/bin/pytest -svx --tb=short tests tests/test_all_to_all.py::test_all_to_all_4_gpu'
        # pip uninstall pplx-kernels && TORCH_CUDA_ARCH_LIST=9.0a+PTX python3 setup.py bdist_wheel && pip install dist/*.whl && torchrun --nproc-per-node 4 /lustre/fs1/portfolios/coreai/projects/coreai_libraries_nvshmem/wilchan/pplx/bin/pytest -svx --tb=short tests tests/test_all_to_all.py::test_all_to_all_4_gpu
        
        align = 16
        per_token_bytes = (((hidden_dim_bytes + hidden_dim_scale_bytes + 4) + align - 1) // align) * align  # + 4 for uint32_t # round_up<size_t>(hiddenDimBytes + hiddenDimScaleBytes + sizeof(uint32_t), 16);
        max_batch_tokens = numLocalExperts * numDPGroups * max_num_tokens


        xDispatchIn = nvshmem.memory.buffer(max_num_tokens * per_token_bytes * 1)   # uint8 right???
        xDispatchOut = nvshmem.memory.buffer(max_batch_tokens * per_token_bytes * 1)

        xCombineIn = nvshmem.memory.buffer(max_batch_tokens * hidden_dim * 4)   # should be float32 right??
        xCombineOut = nvshmem.memory.buffer(max_num_tokens * num_experts * hidden_dim * 4)


        # numTokensBuffer[:] = 0
        # numDispatchRecvBuffer[:] = 0
        # combineSignalBuffer[:] = 0
        # combineSyncBuffer[:] = 0
        # xDispatchIn[:] = 0
        # xDispatchOut[:] = 0
        # xCombineIn[:] = 0
        # xCombineOut[:] = 0


        print("allocated nvshmem core tensors!")

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

            # numTokensBuffer.data_ptr,
            # numDispatchRecvBuffer,
            # combineSignalBuffer,
            # combineSyncBuffer,
            # xDispatchIn,
            # xDispatchOut,
            # xCombineIn,
            # xCombineOut,
        )
        assert ptr != 0

        return cls(
            ptr,
            _ops.all_to_all_internode_combine,
            _ops.all_to_all_internode_dispatch,
            has_scales,
        )
