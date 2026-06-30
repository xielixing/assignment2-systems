from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


class NaiveDDP(nn.Module):
    """A small distributed data parallel wrapper for the assignment.

    This implementation intentionally performs gradient communication only
    after the backward pass has completed. Each parameter gradient is
    all-reduced independently, then averaged across ranks.
    """

    def __init__(self, module: nn.Module, process_group: dist.ProcessGroup | None = None):
        super().__init__()
        self.module = module
        self.process_group = process_group
        self._broadcast_module_state()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _distributed_is_ready(self) -> bool:
        return dist.is_available() and dist.is_initialized() and dist.get_world_size(self.process_group) > 1

    def _broadcast_module_state(self) -> None:
        if not self._distributed_is_ready():
            return

        with torch.no_grad():
            for parameter in self.module.parameters():
                dist.broadcast(parameter.data, src=0, group=self.process_group)
            for buffer in self.module.buffers():
                dist.broadcast(buffer.data, src=0, group=self.process_group)

    def finish_gradient_synchronization(self) -> None:
        if not self._distributed_is_ready():
            return

        world_size = dist.get_world_size(self.process_group)
        with torch.no_grad():
            for parameter in self.module.parameters():
                if parameter.grad is None:
                    continue
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=self.process_group)
                parameter.grad.div_(world_size)


DistributedDataParallel = NaiveDDP
