# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.cp import FLACPContext
from fla.ops.os_kda.chunk_bwd import chunk_kda_bwd
from fla.ops.os_kda.chunk_fwd import chunk_kda_fwd
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

class ChunkKDAFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx, q, k, v, g, beta, A_log, dt_bias, scale, initial_state,
        eta, use_denominator, d_min, d_max, initial_d, output_final_state=False, output_final_d=False,
        use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, cu_seqlens=None, cu_seqlens_cpu=None, safe_gate=False, lower_bound=None, disable_recompute=False, return_intermediate_states=False, cp_context=None, transpose_state_layout=False,
    ):
        chunk_size = 64
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size, cu_seqlens_cpu=cu_seqlens_cpu) if cu_seqlens is not None else None
        g_input = g

        ret = chunk_kda_fwd(
            q=q, k=k, v=v, g=g_input, beta=beta, scale=scale, eta=eta, use_denominator=use_denominator, d_min=d_min, d_max=d_max, initial_state=initial_state, output_final_state=output_final_state, initial_d=initial_d, output_final_d=output_final_d, cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, chunk_indices=chunk_indices, safe_gate=safe_gate, lower_bound=lower_bound, use_gate_in_kernel=use_gate_in_kernel, A_log=A_log, dt_bias=dt_bias, disable_recompute=disable_recompute, return_intermediate_states=return_intermediate_states, cp_context=cp_context, transpose_state_layout=transpose_state_layout,
        )
        
        o, final_state, g_cumsum, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state, d, final_d = ret

        if return_intermediate_states:
            assert torch.is_inference_mode_enabled()
            return o.type_as(q), final_state, final_d, h

        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g_cumsum, g_input, beta, A_log, dt_bias, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state, cu_seqlens, chunk_indices, d)
        ctx.chunk_size = chunk_size
        ctx.safe_gate = safe_gate
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.disable_recompute = disable_recompute
        ctx.cp_context = cp_context
        ctx.transpose_state_layout = transpose_state_layout
        
        ctx.eta = eta
        ctx.use_denominator = use_denominator
        ctx.d_min = d_min
        ctx.d_max = d_max

        return tuple([o.type_as(q), final_state, final_d])

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do, dht=None, dd_final=None):
        (q, q_rstd, k, k_rstd, v, g_cumsum, g_input, beta, A_log, dt_bias, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state, cu_seqlens, chunk_indices, d) = ctx.saved_tensors

        dq, dk, dv, db, dg, dh0, dA, dbias, dd_initial = chunk_kda_bwd(
            q=q, k=k, v=v, g=g_cumsum, beta=beta, Aqk=Aqk, Akk=Akk, scale=ctx.scale, initial_state=initial_state, do=do, dht=dht, d=d, eta=ctx.eta, use_denominator=ctx.use_denominator, d_min=ctx.d_min, d_max=ctx.d_max, dd_final=dd_final, output_initial_d_gradient=True, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_size=ctx.chunk_size, safe_gate=ctx.safe_gate, g_org=g_input if ctx.use_gate_in_kernel else None, lower_bound=ctx.lower_bound, use_gate_in_kernel=ctx.use_gate_in_kernel, A_log=A_log, dt_bias=dt_bias, disable_recompute=ctx.disable_recompute, w=w, u=u, qg=qg, kg=kg, v_new=v_new, h=h, cp_context=ctx.cp_context, transpose_state_layout=ctx.transpose_state_layout,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        return (dq.to(q), dk.to(k), dv.to(v), dg.to(g_input), db.to(beta), dA, dbias, None, dh0, None, None, None, None, dd_initial, None, None, None, None, None, None, None, None, None, None, None, None)

@torch.compiler.disable
def chunk_kda(
    q, k, v, g, beta, scale=None, initial_state=None,
    eta=1.0, use_denominator=False, d_min=None, d_max=None, initial_d=None, output_final_state=False, output_final_d=False,
    use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, cu_seqlens=None, cu_seqlens_cpu=None, safe_gate=False, lower_bound=None, disable_recompute=False, return_intermediate_states=False, cp_context=None, transpose_state_layout=False, **kwargs,
):
    if cp_context is not None:
        assert initial_state is None, "Initial state is not supported for CP"
        assert output_final_state is False, "Output final state is not supported for CP"
        assert cp_context.cu_seqlens is not None, "cu_seqlens is required for CP"
        cu_seqlens = cp_context.cu_seqlens
        if cp_context.cu_seqlens_cpu is not None: cu_seqlens_cpu = cp_context.cu_seqlens_cpu

    if cu_seqlens is not None:
        if q.shape[0] != 1: raise ValueError(f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`.")
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1: raise ValueError(f"Initial states mismatch.")
    if initial_state is not None: assert initial_state.dtype == torch.float32, "initial_state must be in float32."

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        assert "A_log" in kwargs, "A_log must be provided when use_gate_in_kernel=True."
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")

    if safe_gate and use_gate_in_kernel:
        if lower_bound is None: raise ValueError("`lower_bound` must be specified when `safe_gate=True` and `use_gate_in_kernel=True`.")

    if scale is None: scale = k.shape[-1] ** -0.5
    return ChunkKDAFunction.apply(
        q, k, v, g, beta, A_log, dt_bias, scale, initial_state, eta, use_denominator, d_min, d_max, initial_d, output_final_state, output_final_d, use_qk_l2norm_in_kernel, use_gate_in_kernel, cu_seqlens, cu_seqlens_cpu, safe_gate, lower_bound, disable_recompute, return_intermediate_states, cp_context, transpose_state_layout,
    )