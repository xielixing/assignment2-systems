from __future__ import annotations

import argparse
import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from cs336_systems.ddp import NaiveDDP


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _setup(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(minutes=5))


def _worker(rank: int, world_size: int, port: int, steps: int, atol: float) -> None:
    _setup(rank, world_size, port)

    torch.manual_seed(0)
    baseline = TinyModel()

    torch.manual_seed(rank)
    ddp_model = NaiveDDP(TinyModel())

    torch.manual_seed(1234)
    batch_size = 16
    x = torch.randn(batch_size, 8)
    y = torch.randn(batch_size, 4)
    assert batch_size % world_size == 0
    local_batch = batch_size // world_size

    loss_fn = nn.MSELoss()
    baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05)
    ddp_optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.05)

    for step in range(steps):
        baseline_optimizer.zero_grad(set_to_none=True)
        ddp_optimizer.zero_grad(set_to_none=True)

        baseline_loss = loss_fn(baseline(x), y)
        baseline_loss.backward()
        baseline_optimizer.step()

        offset = rank * local_batch
        local_x = x[offset : offset + local_batch]
        local_y = y[offset : offset + local_batch]
        ddp_loss = loss_fn(ddp_model(local_x), local_y)
        ddp_loss.backward()
        ddp_model.finish_gradient_synchronization()
        ddp_optimizer.step()

        torch.manual_seed(42 + step)
        permutation = torch.randperm(batch_size)
        x = x[permutation]
        y = y[permutation]

    max_diff = 0.0
    if rank == 0:
        for baseline_parameter, ddp_parameter in zip(baseline.parameters(), ddp_model.parameters()):
            max_diff = max(max_diff, float((baseline_parameter - ddp_parameter).abs().max()))
        if max_diff > atol:
            raise AssertionError(f"DDP model diverged from single-process baseline: max_diff={max_diff:.3e}")
        print(f"Naive DDP verification passed with max parameter diff {max_diff:.3e}.")

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify naive DDP against single-process training on random data.")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    port = _free_port()
    mp.spawn(_worker, args=(args.world_size, port, args.steps, args.atol), nprocs=args.world_size, join=True)


if __name__ == "__main__":
    main()
