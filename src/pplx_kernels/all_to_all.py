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
        xDispatchIn = None,
        xDispatchOut = None,
        xCombineIn = None,
        xCombineOut = None
    ) -> None:
        self._ptr = ptr
        self._combine_fn = combine_fn
        self._dispatch_fn = dispatch_fn
        self._has_scales = has_scales
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
        if self.numTokensBuffer is not None:
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
            group_name
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

        def ceil_div(x: int, y: int) -> int:
            return (x + y - 1) // y

        numLocalExperts = ceil_div(num_experts, world_size)
        numDPGroups     = ceil_div(world_size,  dp_size)

        # Note that this should be torch.uint64, but uint64 not supported on most systems, so we just zero it out here
        numTokensBuffer = nvshmem.interop.torch.tensor((numLocalExperts * numDPGroups,), dtype=torch.int64)
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

        per_token_bytes = round_up(hidden_dim_bytes + hidden_dim_scale_bytes + 4, 16)  # TODO: + 4 for uint32_t
        max_batch_tokens = numLocalExperts * numDPGroups * max_num_tokens

        xDispatchIn = nvshmem.interop.torch.tensor( (max_num_tokens * per_token_bytes,), dtype=torch.uint8 )
        xDispatchOut = nvshmem.interop.torch.tensor( (max_batch_tokens * per_token_bytes,), dtype=torch.uint8 )

        xCombineIn = nvshmem.interop.torch.tensor( (max_batch_tokens * hidden_dim,), dtype=torch.float32 )
        xCombineOut = nvshmem.interop.torch.tensor( (max_num_tokens * num_experts * hidden_dim,), dtype=torch.float32 )

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
            combineSyncBuffer,
            xDispatchIn,
            xDispatchOut,
            xCombineIn,
            xCombineOut
        )
        assert ptr != 0

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

