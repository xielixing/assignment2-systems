from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import statistics
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.ddp import NaiveDDP


MODEL_CONFIGS = {
    "xl": {
        "vocab_size": 10_000,
        "context_length": 512,
        "d_model": 2560,
        "num_layers": 32,
        "num_heads": 32,
        "d_ff": 10240,
    },
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _cuda_event_elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.synchronize()
    return float(start.elapsed_time(end))


def _training_step(
    model: NaiveDDP,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    use_bf16: bool,
) -> tuple[float, float, float]:
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    comm_start = torch.cuda.Event(enable_timing=True)
    comm_end = torch.cuda.Event(enable_timing=True)

    total_start.record()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    loss.backward()

    comm_start.record()
    model.finish_gradient_synchronization()
    comm_end.record()

    optimizer.step()
    total_end.record()

    return _cuda_event_elapsed_ms(total_start, total_end), _cuda_event_elapsed_ms(comm_start, comm_end), float(loss.detach())


def _worker(
    rank: int,
    world_size: int,
    config_name: str,
    global_batch_size: int,
    warmup_steps: int,
    measurement_steps: int,
    lr: float,
    use_bf16: bool,
    output_json: str,
    master_port: int,
) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{master_port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=30),
    )

    device = torch.device("cuda", rank)
    config = MODEL_CONFIGS[config_name]
    if global_batch_size % world_size != 0:
        raise ValueError(f"global_batch_size={global_batch_size} must be divisible by world_size={world_size}.")
    local_batch_size = global_batch_size // world_size

    torch.manual_seed(0)
    model = NaiveDDP(BasicsTransformerLM(**config).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    generator = torch.Generator(device=device)
    generator.manual_seed(10_000 + rank)
    input_ids = torch.randint(
        0,
        config["vocab_size"],
        (local_batch_size, config["context_length"]),
        device=device,
        generator=generator,
    )
    labels = torch.randint(
        0,
        config["vocab_size"],
        (local_batch_size, config["context_length"]),
        device=device,
        generator=generator,
    )

    for _ in range(warmup_steps):
        _training_step(model, optimizer, input_ids, labels, use_bf16)
    torch.cuda.synchronize(device)
    dist.barrier()

    total_times_ms: list[float] = []
    comm_times_ms: list[float] = []
    losses: list[float] = []
    for _ in range(measurement_steps):
        total_ms, comm_ms, loss = _training_step(model, optimizer, input_ids, labels, use_bf16)
        total_times_ms.append(total_ms)
        comm_times_ms.append(comm_ms)
        losses.append(loss)

    torch.cuda.synchronize(device)
    local_metrics = torch.tensor(
        [
            statistics.fmean(total_times_ms),
            statistics.fmean(comm_times_ms),
            max(total_times_ms),
            max(comm_times_ms),
        ],
        dtype=torch.float64,
        device=device,
    )
    max_metrics = local_metrics.clone()
    mean_metrics = local_metrics.clone()
    dist.all_reduce(max_metrics, op=dist.ReduceOp.MAX)
    dist.all_reduce(mean_metrics, op=dist.ReduceOp.SUM)
    mean_metrics /= world_size

    if rank == 0:
        total_mean_ms = float(max_metrics[0].item())
        comm_mean_ms = float(max_metrics[1].item())
        result = {
            "backend": "nccl",
            "world_size": world_size,
            "gpus": world_size,
            "model_size": config_name,
            **config,
            "global_batch_size": global_batch_size,
            "local_batch_size": local_batch_size,
            "precision": "bf16 autocast" if use_bf16 else "fp32",
            "optimizer": "torch.optim.AdamW",
            "warmup_steps": warmup_steps,
            "measurement_steps": measurement_steps,
            "max_rank_mean_total_ms": total_mean_ms,
            "max_rank_mean_comm_ms": comm_mean_ms,
            "mean_rank_mean_total_ms": float(mean_metrics[0].item()),
            "mean_rank_mean_comm_ms": float(mean_metrics[1].item()),
            "communication_fraction": comm_mean_ms / total_mean_ms,
            "rank0_final_loss": statistics.fmean(losses),
        }
        Path(output_json).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    dist.barrier()
    dist.destroy_process_group()


def _write_csv(result: dict[str, object], path: Path) -> None:
    fieldnames = [
        "backend",
        "world_size",
        "model_size",
        "context_length",
        "global_batch_size",
        "local_batch_size",
        "precision",
        "warmup_steps",
        "measurement_steps",
        "max_rank_mean_total_ms",
        "max_rank_mean_comm_ms",
        "communication_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({field: result[field] for field in fieldnames})


def _write_plot(result: dict[str, object], path: Path) -> None:
    total_ms = float(result["max_rank_mean_total_ms"])
    comm_ms = float(result["max_rank_mean_comm_ms"])
    comm_fraction = comm_ms / total_ms if total_ms > 0 else 0.0

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    labels = ["Total step", "Gradient communication"]
    values = [total_ms, comm_ms]
    colors = ["#4C78A8", "#F58518"]
    bars = ax.bar(labels, values, color=colors, width=0.55)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + total_ms * 0.025,
            f"{value:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.text(
        1,
        comm_ms + total_ms * 0.105,
        f"{comm_fraction * 100:.1f}% of step",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#7A3E00",
    )
    ax.set_ylabel("Time per training step (ms)")
    ax.set_title("Naive DDP: total step time vs communication time")
    ax.set_ylim(0, total_ms * 1.22)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_summary(result: dict[str, object], path: Path) -> None:
    text = "\n".join(
        [
            "# Naive DDP benchmark",
            "",
            "Setup: one node with 2 GPUs/processes, NCCL backend, one process per GPU. The model is the xl Transformer from Table 1: vocab size 10,000, context length 512, d_model 2560, d_ff 10240, 32 layers, and 32 attention heads. The global batch size is 4, so each rank trains on 2 examples per iteration; after backward, `NaiveDDP` all-reduces each parameter gradient individually and averages by world size before the AdamW step.",
            "",
            "| GPUs/processes | Precision | Total step time (ms) | Gradient communication time (ms) | Communication fraction |",
            "|---:|---|---:|---:|---:|",
            f"| {result['world_size']} | {result['precision']} | {float(result['max_rank_mean_total_ms']):.3f} | {float(result['max_rank_mean_comm_ms']):.3f} | {float(result['communication_fraction']) * 100:.1f}% |",
            "",
            "The reported runtime uses the max-rank mean because a distributed training iteration is gated by the slowest rank. In this naive implementation communication is on the critical path: all parameter gradients are reduced only after the full backward pass completes, so the communication fraction is direct overhead added before the optimizer step.",
            "",
            "Plot: `step_time_breakdown.png`",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark naive DDP training for the CS336 xl language model.")
    parser.add_argument("--model-size", choices=sorted(MODEL_CONFIGS), default="xl")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--global-batch-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measurement-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/naive_ddp"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the naive DDP benchmark.")
    if torch.cuda.device_count() < args.world_size:
        raise RuntimeError(f"Need at least {args.world_size} CUDA devices, found {torch.cuda.device_count()}.")

    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("NCCL_NVLS_ENABLE", "0")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_json = args.output_dir / "result.json"
    result_csv = args.output_dir / "results.csv"
    plot_path = args.output_dir / "step_time_breakdown.png"
    summary_path = args.output_dir / "summary.md"

    mp.spawn(
        _worker,
        args=(
            args.world_size,
            args.model_size,
            args.global_batch_size,
            args.warmup_steps,
            args.measurement_steps,
            args.lr,
            args.bf16,
            str(result_json),
            _free_port(),
        ),
        nprocs=args.world_size,
        join=True,
    )

    result = json.loads(result_json.read_text(encoding="utf-8"))
    _write_csv(result, result_csv)
    _write_plot(result, plot_path)
    _write_summary(result, summary_path)
    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
