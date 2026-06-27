from __future__ import annotations

import torch


def _attention_scores(q: torch.Tensor, k: torch.Tensor, is_causal: bool) -> torch.Tensor:
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) * (d**-0.5)
    if is_causal:
        n_queries, n_keys = scores.shape[-2:]
        query_idx = torch.arange(n_queries, device=scores.device)
        key_idx = torch.arange(n_keys, device=scores.device)
        causal_mask = query_idx[:, None] >= key_idx[None, :]
        scores = torch.where(causal_mask, scores, scores.new_full((), -1e6))
    return scores


def _regular_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool) -> torch.Tensor:
    scores = _attention_scores(q, k, is_causal)
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def _flash_backward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    acc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q_work = q.to(acc_dtype)
    k_work = k.to(acc_dtype)
    v_work = v.to(acc_dtype)
    output_work = output.to(acc_dtype)
    grad_output_work = grad_output.to(acc_dtype)
    logsumexp_work = logsumexp.to(acc_dtype)

    n_queries = q.shape[-2]
    n_keys = k.shape[-2]
    scale = q.shape[-1] ** -0.5

    scores = q_work @ k_work.transpose(-2, -1) * scale
    if is_causal:
        query_idx = torch.arange(n_queries, device=q.device)
        key_idx = torch.arange(n_keys, device=q.device)
        causal_mask = query_idx[:, None] >= key_idx[None, :]
        scores = torch.where(causal_mask, scores, scores.new_full((), -1e6))

    probs = torch.exp(scores - logsumexp_work.unsqueeze(-1))
    d = torch.sum(grad_output_work * output_work, dim=-1)
    dp = grad_output_work @ v_work.transpose(-2, -1)
    ds = probs * (dp - d.unsqueeze(-1))

    dq = ds @ k_work * scale
    dk = ds.transpose(-2, -1) @ q_work * scale
    dv = probs.transpose(-2, -1) @ grad_output_work
    return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype)


try:
    flash_backward = torch.compile(_flash_backward_impl)
except RuntimeError as exc:
    if "Dynamo is not supported on Python 3.13" not in str(exc):
        raise
    flash_backward = _flash_backward_impl


class FlashAttentionPyTorch(torch.autograd.Function):
    Q_TILE_SIZE = 16
    K_TILE_SIZE = 16

    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        if q.ndim < 3 or k.ndim < 3 or v.ndim < 3:
            raise ValueError("q, k, and v must have shape (..., sequence_length, d_model).")
        if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
            raise ValueError("q, k, and v must have matching batch dimensions.")
        if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
            raise ValueError("q, k, and v must have the same embedding dimension.")
        if k.shape[-2] != v.shape[-2]:
            raise ValueError("k and v must have the same sequence length.")

        *batch_dims, n_queries, d = q.shape
        n_keys = k.shape[-2]
        flat_batch = 1
        for dim in batch_dims:
            flat_batch *= dim

        q_flat = q.reshape(flat_batch, n_queries, d)
        k_flat = k.reshape(flat_batch, n_keys, d)
        v_flat = v.reshape(flat_batch, n_keys, d)

        acc_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        q_work = q_flat.to(acc_dtype)
        k_work = k_flat.to(acc_dtype)
        v_work = v_flat.to(acc_dtype)

        output = torch.empty((flat_batch, n_queries, d), device=q.device, dtype=acc_dtype)
        logsumexp = torch.empty((flat_batch, n_queries), device=q.device, dtype=acc_dtype)
        scale = d**-0.5

        q_tile_size = FlashAttentionPyTorch.Q_TILE_SIZE
        k_tile_size = FlashAttentionPyTorch.K_TILE_SIZE

        for q_start in range(0, n_queries, q_tile_size):
            q_end = min(q_start + q_tile_size, n_queries)
            q_tile = q_work[:, q_start:q_end, :]
            tile_rows = q_end - q_start

            running_max = torch.full((flat_batch, tile_rows), -torch.inf, device=q.device, dtype=acc_dtype)
            running_sum = torch.zeros((flat_batch, tile_rows), device=q.device, dtype=acc_dtype)
            running_output = torch.zeros((flat_batch, tile_rows, d), device=q.device, dtype=acc_dtype)
            query_idx = torch.arange(q_start, q_end, device=q.device)

            for k_start in range(0, n_keys, k_tile_size):
                k_end = min(k_start + k_tile_size, n_keys)
                k_tile = k_work[:, k_start:k_end, :]
                v_tile = v_work[:, k_start:k_end, :]

                scores = q_tile @ k_tile.transpose(-2, -1) * scale
                if is_causal:
                    key_idx = torch.arange(k_start, k_end, device=q.device)
                    causal_mask = query_idx[:, None] >= key_idx[None, :]
                    scores = torch.where(causal_mask, scores, scores.new_full((), -1e6))

                new_max = torch.maximum(running_max, scores.max(dim=-1).values)
                old_scale = torch.exp(running_max - new_max)
                probs = torch.exp(scores - new_max.unsqueeze(-1))

                running_sum = running_sum * old_scale + probs.sum(dim=-1)
                running_output = running_output * old_scale.unsqueeze(-1) + probs @ v_tile
                running_max = new_max

            output[:, q_start:q_end, :] = running_output / running_sum.unsqueeze(-1)
            logsumexp[:, q_start:q_end] = torch.log(running_sum) + running_max

        output = output.reshape(*batch_dims, n_queries, d).to(q.dtype)
        logsumexp = logsumexp.reshape(*batch_dims, n_queries)

        ctx.save_for_backward(logsumexp, q, k, v, output)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logsumexp, q, k, v, output = ctx.saved_tensors
        is_causal = ctx.is_causal

        needs_q, needs_k, needs_v, _needs_causal = ctx.needs_input_grad
        if not (needs_q or needs_k or needs_v):
            return None, None, None, None

        dq, dk, dv = flash_backward(q, k, v, output, grad_output, logsumexp, is_causal)
        dq = None if not needs_q else dq
        dk = None if not needs_k else dk
        dv = None if not needs_v else dv
        return dq, dk, dv, None
