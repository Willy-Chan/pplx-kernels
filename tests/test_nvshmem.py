import pytest
import torch

from pplx_kernels.nvshmem import (
    nvshmem_alloc_empty_unique_id,
    nvshmem_alltoall,
    nvshmem_barrier_all_on_current_stream,
    nvshmem_finalize,
    nvshmem_get_unique_id,
    nvshmem_init,
    nvshmem_malloc,
    nvshmem_my_pe,
    nvshmem_n_pes,
)

from .distributed_utils import (
    ProcessGroupInfo,
    parallel_launch,
    parallel_launch_from_env,
    require_multi_node,
)

# from mpi4py import MPI
from cuda.core.experimental import Device, system
import nvshmem.core as nvshmem
import numpy as np
import os
import torch.distributed as dist
from nvshmem.core import Teams

def test_nvshmem_1_gpu() -> None:
    # """
    # uid = nvshmem_get_unique_id()
    # nvshmem_init(uid, 0, 1)
    # assert nvshmem_my_pe() == 0
    # assert nvshmem_n_pes() == 1
    # nvshmem_finalize()
    # """
    
    # [UID bootstrap with torchrun] torchrun --nproc-per-node 1 /lustre/fs1/portfolios/coreai/projects/coreai_libraries_nvshmem/wilchan/pplx/bin/pytest -svx --tb=short tests tests/test_nvshmem.py::test_nvshmem_1_gpu
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Set the device for PyTorch (ensures torch operations target the right GPU)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Set the device for custom CUDA code (cuda.core) (ensures NVSHMEM operations target the right GPU)
    dev = Device(local_rank)
    dev.set_current()

    # Create a unique NVSHMEM UID on rank 0
    uniqueid = nvshmem.get_unique_id()

    # Initialize NVSHMEM. No need to broadcast the uid since we're just using 1 GPU.
    nvshmem.init(device=dev, uid=uniqueid, rank=0, nranks=1, initializer_method="uid")

    # Sanity checks
    assert nvshmem.my_pe() == 0
    assert nvshmem.n_pes() == 1

    nvshmem.finalize()
    


def _worker_test_nvshmem_4_gpu(pgi: ProcessGroupInfo) -> None:
    # """
    # uid = nvshmem_get_unique_id() if pgi.rank == 0 else nvshmem_alloc_empty_unique_id()
    # torch.distributed.broadcast(uid, src=0)
    # nvshmem_init(uid, pgi.rank, pgi.world_size)
    # assert nvshmem_my_pe() == pgi.rank
    # assert nvshmem_n_pes() == pgi.world_size
    # nvshmem_finalize()
    # """
    
    # [UID bootstrap with torchrun] torchrun --nproc-per-node 4 /lustre/fs1/portfolios/coreai/projects/coreai_libraries_nvshmem/wilchan/pplx/bin/pytest -svx --tb=short tests tests/test_nvshmem.py::test_nvshmem_4_gpu
    # PGI given parameters that specify the communication group
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Set the device for PyTorch (ensures torch operations target the right GPU)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Set the device for custom CUDA code (cuda.core) (ensures NVSHMEM operations target the right GPU)
    dev = Device(local_rank)
    dev.set_current()

    # Set up torch.distributed (dist) backend - always use cuda:nccl
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=local_rank,
        world_size=world_size,
        device_id=device
    )
    
    num_ranks = dist.get_world_size()
    rank_id = dist.get_rank()

    # Create a unique NVSHMEM UID on rank 0, empty UID on others
    uniqueid = nvshmem.get_unique_id(empty=True)
    if rank_id == 0:
        uniqueid = nvshmem.get_unique_id()
        broadcast_objects = [uniqueid]
    else:
        broadcast_objects = [None]

    # Broadcast the UID from rank 0 to all other ranks
    dist.broadcast_object_list(broadcast_objects, src=0)
    dist.barrier()

    # Initialize NVSHMEM with the broadcasted UID
    nvshmem.init(device=dev, uid=broadcast_objects[0], rank=rank_id, nranks=num_ranks, initializer_method="uid")

    # Sanity checks
    assert nvshmem.my_pe() == pgi.rank
    assert nvshmem.n_pes() == pgi.world_size

    nvshmem.finalize()
    dist.destroy_process_group()


# TODO: Note from Willy - I'm not sure if there's a more elegant way to do this but right now I'm launching this test like so:
#   torchrun --nproc-per-node 4 /lustre/fs1/portfolios/coreai/projects/coreai_libraries_nvshmem/wilchan/pplx/bin/pytest -svx --tb=short tests tests/test_nvshmem.py::test_nvshmem_4_gpu
# But parallel_launch(n, _test_) seems to fork off n processes other processes so the total number is n * 4 if you're using torchrun.
# Currently working assuming the user wants torchrun + UID based nvshmem.
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
def test_nvshmem_4_gpu() -> None:

    # parallel_launch(4, _test_)

    if "LOCAL_RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Must be run with torchrun")

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    pgi = ProcessGroupInfo(
        world_size=world_size,
        world_local_size=int(os.environ.get("LOCAL_WORLD_SIZE", world_size)),
        rank=int(os.environ["RANK"]),
        node_rank=int(os.environ.get("NODE_RANK", 0)),
        local_rank=local_rank,
        device=torch.device("cuda", local_rank),
    )
    _worker_test_nvshmem_4_gpu(pgi)


def _worker_test_all_to_all(pgi: ProcessGroupInfo) -> None:
    # uid = nvshmem_get_unique_id() if pgi.rank == 0 else nvshmem_alloc_empty_unique_id()
    # torch.distributed.broadcast(uid, src=0)
    # nvshmem_init(uid, pgi.rank, pgi.world_size)
    # try:
    #     t_in = nvshmem_malloc([pgi.world_size], dtype=torch.int32, device=pgi.device)
    #     t_in.copy_(
    #         torch.full([pgi.world_size], pgi.rank, dtype=torch.int32, device=pgi.device)
    #     )

    #     t_out = nvshmem_malloc([pgi.world_size], dtype=torch.int32, device=pgi.device)

    #     nvshmem_alltoall(t_out, t_in)
    #     nvshmem_barrier_all_on_current_stream()
    #     torch.cuda.synchronize()

    #     assert t_out.tolist() == list(range(pgi.world_size))
    # finally:
    #     del t_in
    #     del t_out
    #     nvshmem_finalize()


    # First, we uid-bootstrap nvshmem with torchrun
    # PGI given parameters that specify the communication group
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Set the device for PyTorch (ensures torch operations target the right GPU)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Set the device for custom CUDA code (cuda.core) (ensures NVSHMEM operations target the right GPU)
    dev = Device(local_rank)
    dev.set_current()
    stream = dev.create_stream()

    # Set up torch.distributed (dist) backend
    # TODO: what's the correct uid-bootstrap backend?
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        rank=local_rank,
        world_size=world_size,
        device_id=device
    )
    
    num_ranks = dist.get_world_size()
    rank_id = dist.get_rank()

    # Create a unique NVSHMEM UID on rank 0, empty UID on others
    uniqueid = nvshmem.get_unique_id(empty=True)
    if rank_id == 0:
        uniqueid = nvshmem.get_unique_id()
        broadcast_objects = [uniqueid]
    else:
        broadcast_objects = [None]

    # Broadcast the UID from rank 0 to all other ranks
    dist.broadcast_object_list(broadcast_objects, src=0)
    dist.barrier()

    # Initialize NVSHMEM with the broadcasted UID
    nvshmem.init(device=dev, uid=broadcast_objects[0], rank=rank_id, nranks=num_ranks, initializer_method="uid")

    # all-to-all test
    try:
        # Allocate a PyTorch tensor backed by NVSHMEM symmetric memory
        t_in = nvshmem.interop.torch.tensor( (pgi.world_size,), dtype=torch.int32 )
        t_in.fill_(pgi.rank)
        t_out = nvshmem.interop.torch.tensor( (pgi.world_size,), dtype=torch.int32 )

        # perform the all-to-all operation with TEAM_WORLD and the specified stream
        team = Teams.TEAM_WORLD
        nvshmem.collective.alltoall(team, t_out, t_in, stream=stream)

        nvshmem.collective.barrier(team, stream=stream)
        torch.cuda.synchronize()

        assert t_out.tolist() == list(range(pgi.world_size))

    finally:
        nvshmem.interop.torch.free_tensor(t_in)
        nvshmem.interop.torch.free_tensor(t_out)
        nvshmem.finalize()
        dist.destroy_process_group()

    


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="Requires at least 4 GPUs")
def test_all_to_all() -> None:
    # parallel_launch(4, _worker_test_all_to_all)

    # TODO: use their parallel launcher later/ask them how it works

    if "LOCAL_RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Must be run with torchrun")

    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    pgi = ProcessGroupInfo(
        world_size=world_size,
        world_local_size=int(os.environ.get("LOCAL_WORLD_SIZE", world_size)),
        rank=int(os.environ["RANK"]),
        node_rank=int(os.environ.get("NODE_RANK", 0)),
        local_rank=local_rank,
        device=torch.device("cuda", local_rank),
    )
    _worker_test_all_to_all(pgi)


# TODO: need to make the all-to-all work for multinode environments, investigate parallel_launch_from_env() more.
@require_multi_node
def test_all_to_all_multi_node() -> None:
    parallel_launch_from_env(_worker_test_all_to_all)
