from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D_MODEL: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    q_offs = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    d_offs = tl.arange(0, D)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    q = tl.load(Q_block_ptr, boundary_check=(0, 1))

    m_i = tl.full([Q_TILE_SIZE], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([Q_TILE_SIZE], dtype=tl.float32)
    acc = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    for k_start in range(0, N_KEYS, K_TILE_SIZE):
        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        v = tl.load(V_block_ptr, boundary_check=(0, 1))

        scores = tl.dot(q, tl.trans(k)) * scale
        k_offs = k_start + tl.arange(0, K_TILE_SIZE)
        scores = tl.where(k_offs[None, :] < N_KEYS, scores, float("-inf"))
        if IS_CAUSAL:
            mask = q_offs[:, None] >= k_offs[None, :]
            scores = tl.where(mask, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, 1))
        old_scale = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = l_i * old_scale + tl.sum(p, 1)
        m_i = m_new

        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc=(acc * old_scale[:, None]).to(v.dtype))

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    acc = acc.to(tl.float32) / l_i[:, None]
    lse = tl.log(l_i) + m_i

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    tl.store(O_block_ptr, acc.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))

    l_ptrs = L_ptr + batch_index * stride_lb + q_offs * stride_lq
    tl.store(l_ptrs, lse, mask=q_offs < N_QUERIES)


@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, D_ptr, dO_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_dqb, stride_dqq, stride_dqd,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    scale,
    D_MODEL: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    q_offs = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
    d_offs = tl.arange(0, D)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    q = tl.load(Q_block_ptr, boundary_check=(0, 1))

    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_dob,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_doq, stride_dod),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    do_tile = tl.load(dO_block_ptr, boundary_check=(0, 1))

    l_ptrs = L_ptr + batch_index * stride_lb + q_offs * stride_lq
    lse = tl.load(l_ptrs, mask=q_offs < N_QUERIES, other=0.0)
    d_ptrs = D_ptr + batch_index * stride_db + q_offs * stride_dq
    d_val = tl.load(d_ptrs, mask=q_offs < N_QUERIES, other=0.0)

    dq = tl.zeros([Q_TILE_SIZE, D], dtype=tl.float32)

    for k_start in range(0, N_KEYS, K_TILE_SIZE):
        K_block_ptr = tl.make_block_ptr(
            K_ptr + batch_index * stride_kb,
            shape=(N_KEYS, D_MODEL),
            strides=(stride_kk, stride_kd),
            offsets=(k_start, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            V_ptr + batch_index * stride_vb,
            shape=(N_KEYS, D_MODEL),
            strides=(stride_vk, stride_vd),
            offsets=(k_start, 0),
            block_shape=(K_TILE_SIZE, D),
            order=(1, 0),
        )

        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        v = tl.load(V_block_ptr, boundary_check=(0, 1))

        scores = tl.dot(q, tl.trans(k)) * scale
        if IS_CAUSAL:
            k_offs = k_start + tl.arange(0, K_TILE_SIZE)
            mask = q_offs[:, None] >= k_offs[None, :]
            scores = tl.where(mask, scores, float("-inf"))

        p = tl.exp(scores - lse[:, None])
        dp = tl.dot(do_tile, tl.trans(v))
        ds = p * (dp - d_val[:, None])
        dq += tl.dot(ds, k) * scale

    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_dqb,
        shape=(N_QUERIES, D_MODEL),
        strides=(stride_dqq, stride_dqd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    tl.store(dQ_block_ptr, dq, boundary_check=(0, 1))


@triton.jit
def _bwd_dkv_kernel(
    Q_ptr, K_ptr, V_ptr, L_ptr, D_ptr, dO_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    N_QUERIES, N_KEYS,
    scale,
    D_MODEL: tl.constexpr,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    k_offs = key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
    d_offs = tl.arange(0, D)

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    k = tl.load(K_block_ptr, boundary_check=(0, 1))

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    v = tl.load(V_block_ptr, boundary_check=(0, 1))

    dk = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)
    dv = tl.zeros([K_TILE_SIZE, D], dtype=tl.float32)

    for q_start in range(0, N_QUERIES, Q_TILE_SIZE):
        q_offs = q_start + tl.arange(0, Q_TILE_SIZE)

        Q_block_ptr = tl.make_block_ptr(
            Q_ptr + batch_index * stride_qb,
            shape=(N_QUERIES, D_MODEL),
            strides=(stride_qq, stride_qd),
            offsets=(q_start, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        q = tl.load(Q_block_ptr, boundary_check=(0, 1))

        dO_block_ptr = tl.make_block_ptr(
            dO_ptr + batch_index * stride_dob,
            shape=(N_QUERIES, D_MODEL),
            strides=(stride_doq, stride_dod),
            offsets=(q_start, 0),
            block_shape=(Q_TILE_SIZE, D),
            order=(1, 0),
        )
        do_tile = tl.load(dO_block_ptr, boundary_check=(0, 1))

        l_ptrs = L_ptr + batch_index * stride_lb + q_offs * stride_lq
        lse = tl.load(l_ptrs, mask=q_offs < N_QUERIES, other=0.0)
        d_ptrs = D_ptr + batch_index * stride_db + q_offs * stride_dq
        d_val = tl.load(d_ptrs, mask=q_offs < N_QUERIES, other=0.0)

        scores = tl.dot(q, tl.trans(k)) * scale
        if IS_CAUSAL:
            mask = q_offs[:, None] >= k_offs[None, :]
            scores = tl.where(mask, scores, float("-inf"))

        p = tl.exp(scores - lse[:, None])
        dp = tl.dot(do_tile, tl.trans(v))
        ds = p * (dp - d_val[:, None])
        dk += tl.dot(tl.trans(ds), q) * scale
        dv += tl.dot(tl.trans(p), do_tile)

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_dkb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_dkk, stride_dkd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    tl.store(dK_block_ptr, dk, boundary_check=(0, 1))

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_dvb,
        shape=(N_KEYS, D_MODEL),
        strides=(stride_dvk, stride_dvd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    tl.store(dV_block_ptr, dv, boundary_check=(0, 1))


Q_TILE_SIZE = 16
K_TILE_SIZE = 16


class FlashAttentionTriton(torch.autograd.Function):

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

        device = q.device
        q_flat = q.reshape(flat_batch, n_queries, d).contiguous()
        k_flat = k.reshape(flat_batch, n_keys, d).contiguous()
        v_flat = v.reshape(flat_batch, n_keys, d).contiguous()

        output = torch.empty_like(q_flat, dtype=torch.float32)
        logsumexp = torch.empty(flat_batch, n_queries, device=device, dtype=torch.float32)
        scale = d ** -0.5

        D = triton.next_power_of_2(d)

        grid = (triton.cdiv(n_queries, Q_TILE_SIZE), flat_batch)
        flash_fwd_kernel[grid](
            q_flat, k_flat, v_flat,
            output, logsumexp,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            logsumexp.stride(0), logsumexp.stride(1),
            n_queries, n_keys,
            scale,
            D_MODEL=d, D=D, Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
            IS_CAUSAL=bool(is_causal),
        )

        output = output.reshape(*batch_dims, n_queries, d).to(q.dtype)
        logsumexp = logsumexp.reshape(*batch_dims, n_queries)

        ctx.save_for_backward(logsumexp, q, k, v, output)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logsumexp, q, k, v, output = ctx.saved_tensors
        is_causal = ctx.is_causal

        needs_q, needs_k, needs_v, _ = ctx.needs_input_grad
        if not (needs_q or needs_k or needs_v):
            return None, None, None, None

        *batch_dims, n_queries, d = q.shape
        n_keys = k.shape[-2]
        flat_batch = 1
        for dim in batch_dims:
            flat_batch *= dim

        device = q.device
        q_flat = q.reshape(flat_batch, n_queries, d).contiguous()
        k_flat = k.reshape(flat_batch, n_keys, d).contiguous()
        v_flat = v.reshape(flat_batch, n_keys, d).contiguous()
        o_flat = output.reshape(flat_batch, n_queries, d).contiguous()
        do_flat = grad_output.reshape(flat_batch, n_queries, d).contiguous()
        lse_flat = logsumexp.reshape(flat_batch, n_queries).contiguous()

        D_vals = (o_flat * do_flat).sum(dim=-1)

        scale = d ** -0.5
        D = triton.next_power_of_2(d)

        if needs_q:
            dq = torch.empty_like(q_flat)
            grid_dq = (triton.cdiv(n_queries, Q_TILE_SIZE), flat_batch)
            _bwd_dq_kernel[grid_dq](
                q_flat, k_flat, v_flat, lse_flat, D_vals, do_flat,
                dq,
                q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
                k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
                v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
                do_flat.stride(0), do_flat.stride(1), do_flat.stride(2),
                dq.stride(0), dq.stride(1), dq.stride(2),
                lse_flat.stride(0), lse_flat.stride(1),
                D_vals.stride(0), D_vals.stride(1),
                n_queries, n_keys,
                scale,
                D_MODEL=d, D=D, Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
                IS_CAUSAL=is_causal,
            )
        else:
            dq = None

        if needs_k or needs_v:
            dk = torch.empty_like(k_flat)
            dv = torch.empty_like(v_flat)
            s_dk = dk.stride()
            s_dv = dv.stride()
            grid_dkv = (triton.cdiv(n_keys, K_TILE_SIZE), flat_batch)
            _bwd_dkv_kernel[grid_dkv](
                q_flat, k_flat, v_flat, lse_flat, D_vals, do_flat,
                dk, dv,
                q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
                k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
                v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
                do_flat.stride(0), do_flat.stride(1), do_flat.stride(2),
                s_dk[0], s_dk[1], s_dk[2],
                s_dv[0], s_dv[1], s_dv[2],
                lse_flat.stride(0), lse_flat.stride(1),
                D_vals.stride(0), D_vals.stride(1),
                n_queries, n_keys,
                scale,
                D_MODEL=d, D=D, Q_TILE_SIZE=Q_TILE_SIZE, K_TILE_SIZE=K_TILE_SIZE,
                IS_CAUSAL=is_causal,
            )
            if not needs_k:
                dk = None
            if not needs_v:
                dv = None
        else:
            dk = None
            dv = None

        if dq is not None:
            dq = dq.reshape(*batch_dims, n_queries, d).to(q.dtype)
        if dk is not None:
            dk = dk.reshape(*batch_dims, n_keys, d).to(k.dtype)
        if dv is not None:
            dv = dv.reshape(*batch_dims, n_keys, d).to(v.dtype)

        return dq, dk, dv, None
