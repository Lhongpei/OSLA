# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import math
import os

from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from fla.ops.cp import FLACPContext
from fla.ops.cp.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h_pre_process,
    compress_h0,
)
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from fla.ops.os_kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.os_kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2
from fla.ops.os_delta_rule.chunk_osgm_phase import (
    compute_osgm_phase1_fwd,
    compute_osgm_phase1_bwd,
    fused_osgm_bwd_mapping
)
from fla.ops.os_delta_rule.chunk_osgm_phase_dd_decay_beta import (
    compute_osgm_dd_decay_beta_phase1_fwd,
    compute_osgm_kda_gate_decay_beta_phase1_fwd,
)

_S_EFF_LOG_COUNTER = 0



def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    eta: float = 1.0,
    use_denominator: bool = False,
    d_min: float | None = None,
    d_max: float | None = None,
    beta_aware: bool = False,
    decay_mode: str = "none",
    decay_gamma: float = 1.0,
    g_decay: torch.Tensor | None = None,
    initial_d: torch.Tensor | None = None,
    output_final_d: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    cp_context: FLACPContext | None = None,
    transpose_state_layout: bool = False,
):
    g_org = None
    if use_gate_in_kernel:
        g_org = g
        g = kda_gate_chunk_cumsum(
            g=g_org, A_log=A_log, dt_bias=dt_bias, scale=RCP_LN2, chunk_size=chunk_size,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, lower_bound=lower_bound,
        )
    else:
        g = chunk_local_cumsum(g=g, scale=RCP_LN2, chunk_size=chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)

    if beta_aware:
        if os.environ.get("OSKDA_KDA_GATE_DDECAY", "0") == "1":
            if decay_mode not in ("none", "constant"):
                raise NotImplementedError(f"Unsupported OS-KDA d decay mode: {decay_mode}")
            d, final_d = compute_osgm_kda_gate_decay_beta_phase1_fwd(
                k=k, g=g, beta=beta, eta=eta, use_denominator=use_denominator,
                d_min=d_min, d_max=d_max,
                decay_gamma=float(decay_gamma) if decay_mode == "constant" else 1.0,
                cu_seqlens=cu_seqlens, initial_d=initial_d,
                output_final_state=output_final_d, chunk_size=chunk_size,
            )
        else:
            if decay_mode == "none":
                phase1_g_decay = torch.zeros_like(beta, dtype=torch.float32)
            elif decay_mode == "constant":
                phase1_g_decay = torch.full_like(beta, math.log(float(decay_gamma)), dtype=torch.float32)
            elif decay_mode == "data_dependent":
                if g_decay is None:
                    raise ValueError("decay_mode='data_dependent' requires g_decay")
                phase1_g_decay = g_decay.to(torch.float32)
            else:
                raise NotImplementedError(f"Unsupported OS-KDA d decay mode: {decay_mode}")
            d, final_d = compute_osgm_dd_decay_beta_phase1_fwd(
                k=k, g=phase1_g_decay, beta=beta, eta=eta, use_denominator=use_denominator,
                d_min=d_min, d_max=d_max, cu_seqlens=cu_seqlens,
                initial_d=initial_d, output_final_state=output_final_d,
            )
    else:
        d, final_d = compute_osgm_phase1_fwd(
            k=k, eta=eta, use_denominator=use_denominator,
            d_min=d_min, d_max=d_max, cu_seqlens=cu_seqlens,
            initial_d=initial_d, output_final_state=output_final_d
        )

    s_eff_max = os.environ.get("OSKDA_EFFECTIVE_STEP_MAX")
    if s_eff_max is not None:
        if os.environ.get("OSKDA_DETACH_PHASE1", "0") != "1":
            raise NotImplementedError("OSKDA_EFFECTIVE_STEP_MAX is only enabled for detached phase1 diagnostics.")
        s_eff_max = float(s_eff_max)
        k2 = k.to(torch.float32).square()
        s_eff = beta.to(torch.float32).unsqueeze(-1) * (d.to(torch.float32) * k2).sum(dim=-1, keepdim=True)
        scale_d = torch.clamp(s_eff_max / s_eff.clamp_min(1e-6), max=1.0)
        d = d * scale_d.to(d.dtype)

    if os.environ.get("OSKDA_LOG_S_EFF", "0") == "1":
        global _S_EFF_LOG_COUNTER
        log_limit = int(os.environ.get("OSKDA_LOG_S_EFF_LIMIT", "128"))
        log_every = max(1, int(os.environ.get("OSKDA_LOG_S_EFF_EVERY", "1")))
        if _S_EFF_LOG_COUNTER < log_limit and _S_EFF_LOG_COUNTER % log_every == 0:
            with torch.no_grad():
                vals = beta.to(torch.float32) * (d.to(torch.float32) * k.to(torch.float32).square()).sum(dim=-1)
                flat = vals.flatten()
                qs = torch.quantile(
                    flat,
                    torch.tensor([0.5, 0.95, 0.99, 0.999], device=flat.device, dtype=flat.dtype),
                )
                d_flat = d.to(torch.float32).flatten()
                print(
                    "[OSKDA_S_EFF]"
                    f" call={_S_EFF_LOG_COUNTER}"
                    f" p50={float(qs[0]):.6g} p95={float(qs[1]):.6g} p99={float(qs[2]):.6g} p999={float(qs[3]):.6g}"
                    f" max={float(flat.max()):.6g} min={float(flat.min()):.6g}"
                    f" frac_gt1p5={float((flat > 1.5).float().mean()):.6g} frac_gt2={float((flat > 2.0).float().mean()):.6g} frac_lt0={float((flat < 0).float().mean()):.6g}"
                    f" d_min={float(d_flat.min()):.6g} d_max={float(d_flat.max()):.6g}",
                    flush=True,
                )
        _S_EFF_LOG_COUNTER += 1

    w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
        q=q, k=k, v=v, gk=g, beta=beta, scale=scale, 
        d=d,
        cu_seqlens=cu_seqlens, chunk_size=chunk_size, chunk_indices=chunk_indices,
        safe_gate=safe_gate, disable_recompute=disable_recompute
    )

    if cp_context is not None:
        initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
            k=kg,
            w=w,
            u=u,
            gk=g,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            context=cp_context,
            use_exp2=True,
            transpose_state_layout=transpose_state_layout,
        )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=kg, w=w, u=u, gk=g, initial_state=initial_state, output_final_state=output_final_state,
        cu_seqlens=cu_seqlens, cu_seqlens_cpu=cu_seqlens_cpu, chunk_indices=chunk_indices,
        use_exp2=True, transpose_state_layout=transpose_state_layout,
    )

    if cp_context is not None:
        initial_state = compress_h0(initial_state, context=cp_context)

    o = chunk_gla_fwd_o_gk(
        q=q, v=v_new, g=g, A=Aqk, h=h, scale=scale, cu_seqlens=cu_seqlens, 
        chunk_size=chunk_size, chunk_indices=chunk_indices, use_exp2=True, transpose_state_layout=transpose_state_layout,
    )
    
    if disable_recompute is False:
        w, u, qg, kg, v_new = None, None, None, None, None
        if not return_intermediate_states:
            h = None
        if use_gate_in_kernel:
            g = None
            
    return o, final_state, g, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state, d, final_d
