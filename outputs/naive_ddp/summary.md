# Naive DDP benchmark

Setup: one node with 2 GPUs/processes, NCCL backend, one process per GPU. The model is the xl Transformer from Table 1: vocab size 10,000, context length 512, d_model 2560, d_ff 10240, 32 layers, and 32 attention heads. The global batch size is 4, so each rank trains on 2 examples per iteration; after backward, `NaiveDDP` all-reduces each parameter gradient individually and averages by world size before the AdamW step.

| GPUs/processes | Precision | Total step time (ms) | Gradient communication time (ms) | Communication fraction |
|---:|---|---:|---:|---:|
| 2 | fp32 | 1070.725 | 60.754 | 5.7% |

The reported runtime uses the max-rank mean because a distributed training iteration is gated by the slowest rank. In this naive implementation communication is on the critical path: all parameter gradients are reduced only after the full backward pass completes, so the communication fraction is direct overhead added before the optimizer step.

Plot: `step_time_breakdown.png`
