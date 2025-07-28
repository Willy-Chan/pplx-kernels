import pytest
import torch

from .distributed_utils import (
    ProcessGroupInfo,
    parallel_launch,
    parallel_launch_from_env,
    require_multi_node,
)

from cuda.core.experimental import Device
import nvshmem.core as nvshmem
import torch.distributed as dist
from nvshmem.core import Teams

def test_nvshmem_1_gpu() -> None:

    local_rank = 0
    world_size = 1

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dev = Device(local_rank)
    dev.set_current()

    uniqueid = nvshmem.get_unique_id()
    nvshmem.init(device=dev, uid=uniqueid, rank=0, nranks=1, initializer_method="uid")

    # Check host initialization status
    test_script_init_status = nvshmem.direct.init_status()
    if test_script_init_status < 2 and local_rank == 0:
        logger.warning(
            "NVSHMEM hostlib initialization incomplete - status: %d (rank: %d, local_rank: %d)",
            test_script_init_status, rank_id, local_rank
        )

    assert nvshmem.my_pe() == 0
    assert nvshmem.n_pes() == 1

    nvshmem.finalize()
    


def _worker_test_nvshmem_4_gpu(pgi: ProcessGroupInfo) -> None:
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()

    dev = Device(local_rank)
    dev.set_current()

    uniqueid = nvshmem.get_unique_id(empty=True)
    if local_rank == 0:
        uniqueid = nvshmem.get_unique_id()
        broadcast_objects = [uniqueid]
    else:
        broadcast_objects = [None]

    dist.broadcast_object_list(broadcast_objects, src=0)
    dist.barrier()

    nvshmem.init(device=dev, uid=broadcast_objects[0], rank=local_rank, nranks=world_size, initializer_method="uid")

    # Check host initialization status
    test_script_init_status = nvshmem.direct.init_status()
    if test_script_init_status < 2 and local_rank == 0:
        logger.warning(
            "NVSHMEM hostlib initialization incomplete - status: %d (rank: %d, local_rank: %d)",
            test_script_init_status, rank_id, local_rank
        )

    assert nvshmem.my_pe() == pgi.rank
    assert nvshmem.n_pes() == pgi.world_size

    nvshmem.finalize()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
def test_nvshmem_4_gpu() -> None:
    parallel_launch(4, _worker_test_nvshmem_4_gpu)

class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream
        self.handle = pt_stream.cuda_stream

    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)  # Return format required by CUDA Python

def _worker_test_all_to_all(pgi: ProcessGroupInfo) -> None:
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()

    dev = Device(local_rank)
    dev.set_current()
    stream = PyTorchStreamWrapper(torch.cuda.current_stream())
    
    num_ranks = dist.get_world_size()
    rank_id = dist.get_rank()

    uniqueid = nvshmem.get_unique_id(empty=True)
    if rank_id == 0:
        uniqueid = nvshmem.get_unique_id()
        broadcast_objects = [uniqueid]
    else:
        broadcast_objects = [None]

    dist.broadcast_object_list(broadcast_objects, src=0)
    dist.barrier()

    nvshmem.init(device=dev, uid=broadcast_objects[0], rank=rank_id, nranks=num_ranks, initializer_method="uid")

    # Check NVSHMEM host initialization status
    test_script_init_status = nvshmem.direct.init_status()
    if test_script_init_status < 2 and local_rank == 0:
        logger.warning(
            "NVSHMEM hostlib initialization incomplete - status: %d (rank: %d, local_rank: %d)",
            test_script_init_status, rank_id, local_rank
        )

    # all-to-all test
    try:
        # Allocate a PyTorch tensor backed by NVSHMEM symmetric memory
        t_in = nvshmem.tensor( (pgi.world_size,), dtype=torch.int32 ).fill_(pgi.rank)
        t_out = nvshmem.tensor( (pgi.world_size,), dtype=torch.int32 )

        team = Teams.TEAM_WORLD
        nvshmem.collective.alltoall(team, t_out, t_in, stream=stream)

        nvshmem.collective.barrier(team, stream=stream)
        torch.cuda.synchronize()

        assert t_out.tolist() == list(range(pgi.world_size))
    finally:
        nvshmem.free_tensor(t_in)
        nvshmem.free_tensor(t_out)
        nvshmem.finalize()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
def test_all_to_all() -> None:
    parallel_launch(4, _worker_test_all_to_all)


@require_multi_node
def test_all_to_all_multi_node() -> None:
    parallel_launch_from_env(_worker_test_all_to_all)