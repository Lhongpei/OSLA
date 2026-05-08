#!/usr/bin/env python3
"""Measure per-token Delta-rule residual contraction.

This diagnostic records the theorem-facing quantity

    q_t = f_t(S_t) / f_t(S_{t-1}),
    f_t(S) = 0.5 * ||S k_t - v_t||^2,

inside trained DeltaNet / OSDN checkpoints.  It intentionally recomputes the
Delta-rule recurrence in PyTorch so it can record the write-side residuals
without modifying the Triton kernels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import fla  # noqa: F401 - registers FLA model/config classes with HF.

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_jrt import TASK_CONFIGS, truncate_context  # noqa: E402


def elu_p1(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x, 1.0, False) + 1.0


def sum_norm(x: torch.Tensor) -> torch.Tensor:
    return x / x.sum(-1, keepdim=True)


@dataclass
class ScalarStats:
    count: int = 0
    sum_q: float = 0.0
    sum_log_q: float = 0.0
    sum_pre: float = 0.0
    sum_post: float = 0.0
    sum_align: float = 0.0
    sum_beta_align: float = 0.0

    def add_tensor(
        self,
        q: torch.Tensor,
        pre: torch.Tensor,
        post: torch.Tensor,
        align: torch.Tensor,
        beta_align: torch.Tensor,
    ) -> None:
        q = q.detach().float().flatten().cpu()
        pre = pre.detach().float().flatten().cpu()
        post = post.detach().float().flatten().cpu()
        align = align.detach().float().flatten().cpu()
        beta_align = beta_align.detach().float().flatten().cpu()
        n = int(q.numel())
        if n == 0:
            return
        self.count += n
        self.sum_q += float(q.sum())
        self.sum_log_q += float(torch.log(q.clamp_min(1e-30)).sum())
        self.sum_pre += float(pre.sum())
        self.sum_post += float(post.sum())
        self.sum_align += float(align.sum())
        self.sum_beta_align += float(beta_align.sum())

    def as_dict(self) -> dict[str, float | int]:
        if self.count == 0:
            return {
                "count": 0,
                "q_mean": None,
                "q_geo_mean": None,
                "mean_log_q": None,
                "pre_loss_mean": None,
                "post_loss_mean": None,
                "align_mean": None,
                "beta_align_mean": None,
            }
        mean_log_q = self.sum_log_q / self.count
        return {
            "count": self.count,
            "q_mean": self.sum_q / self.count,
            "q_geo_mean": math.exp(mean_log_q),
            "mean_log_q": mean_log_q,
            "pre_loss_mean": self.sum_pre / self.count,
            "post_loss_mean": self.sum_post / self.count,
            "align_mean": self.sum_align / self.count,
            "beta_align_mean": self.sum_beta_align / self.count,
        }


@dataclass
class ActiveStats:
    label: str
    d_mode: str
    global_stats: ScalarStats
    task_stats: ScalarStats
    abs_curve: "CurveStats"
    rel_curve: "CurveStats"
    layer_stats: list[ScalarStats]


class CurveStats:
    def __init__(self, n: int):
        self.n = n
        self.stats = [ScalarStats() for _ in range(n)]

    def add(self, idx: int, q, pre, post, align, beta_align) -> None:
        if 0 <= idx < self.n:
            self.stats[idx].add_tensor(q, pre, post, align, beta_align)

    def rows(self, key: str, model_label: str, task: str | None = None) -> list[dict]:
        rows = []
        for i, stats in enumerate(self.stats):
            d = stats.as_dict()
            if d["count"] == 0:
                continue
            row = {"model": model_label, key: i}
            if task is not None:
                row["task"] = task
            row.update(d)
            rows.append(row)
        return rows


def parse_model_specs(values: Iterable[str]) -> list[tuple[str, str]]:
    specs = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--model must be label=path, got {value!r}")
        label, path = value.split("=", 1)
        specs.append((label.strip(), path.strip()))
    return specs


def build_prompts(
    tokenizer,
    tasks: list[str],
    max_samples_per_task: int,
    context_length: int,
    answer_length: int,
    seed: int,
) -> dict[str, list[str]]:
    from datasets import load_dataset

    rng = random.Random(seed)
    prompts_by_task: dict[str, list[str]] = {}
    for task_name in tasks:
        cfg = TASK_CONFIGS[task_name]
        ds = list(load_dataset(cfg["dataset"], split="validation"))
        rng.shuffle(ds)
        prompts = []
        is_twice = task_name.endswith("_twice")
        format_fn = cfg["format_fn"]
        ctx_key = cfg["context_key"]
        for doc in ds:
            answer = doc["value"]
            if not answer or len(answer) <= 1:
                continue
            raw_text = doc[ctx_key]
            if is_twice and task_name != "based_fda_twice":
                raw_text = raw_text + "\n" + raw_text
            cut_text = truncate_context(raw_text, answer, tokenizer, context_length, answer_length)
            prompts.append(format_fn(doc, cut_text))
            if len(prompts) >= max_samples_per_task:
                break
        prompts_by_task[task_name] = prompts
    return prompts_by_task


def resolve_osgm_defaults(attn, use_l2norm: bool) -> tuple[float, bool, float | None, float | None]:
    eta = attn.osgm_eta
    use_denominator = attn.osgm_use_denominator
    d_min = attn.osgm_d_min
    d_max = attn.osgm_d_max
    if use_l2norm:
        eta = 1.0 if eta is None else float(eta)
        use_denominator = False if use_denominator is None else bool(use_denominator)
        d_min = 0.0 if d_min is None else float(d_min)
        d_max = 1e9 if d_max is None else float(d_max)
    else:
        eta = 0.1 if eta is None else float(eta)
        use_denominator = True if use_denominator is None else bool(use_denominator)
    return eta, use_denominator, d_min, d_max


def attention_kind(attn) -> str:
    name = attn.__class__.__name__.lower()
    if "kimi" in name or name == "kda":
        return "kda"
    if "gateddelta" in name or "gated_delta" in name:
        return "gdn"
    return "delta"


def project_recurrent_inputs(attn, hidden_states: torch.Tensor):
    kind = attention_kind(attn)
    if kind == "delta":
        return project_delta_inputs(attn, hidden_states)
    if kind == "gdn":
        return project_gdn_inputs(attn, hidden_states)
    if kind == "kda":
        return project_kda_inputs(attn, hidden_states)
    raise NotImplementedError(f"Unsupported attention kind {kind!r}")


def project_delta_inputs(attn, hidden_states: torch.Tensor):
    if attn.use_short_conv:
        q, _ = attn.q_conv1d(
            x=attn.q_proj(hidden_states),
            cache=None,
            output_final_state=False,
            cu_seqlens=None,
        )
        k, _ = attn.k_conv1d(
            x=attn.k_proj(hidden_states),
            cache=None,
            output_final_state=False,
            cu_seqlens=None,
        )
        v, _ = attn.v_conv1d(
            x=attn.v_proj(hidden_states),
            cache=None,
            output_final_state=False,
            cu_seqlens=None,
        )
    else:
        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        if attn.qk_activation == "silu":
            q, k = F.silu(q), F.silu(k)
        v = F.silu(attn.v_proj(hidden_states))

    q, k = map(lambda x: rearrange(x, "... (h d) -> ... h d", d=attn.head_k_dim), (q, k))
    v = rearrange(v, "... (h d) -> ... h d", d=attn.head_v_dim)

    if attn.qk_activation != "silu":
        if attn.qk_activation == "relu":
            q, k = q.relu(), k.relu()
        elif attn.qk_activation == "elu":
            q, k = elu_p1(q), elu_p1(k)
        elif attn.qk_activation != "identity":
            raise NotImplementedError(f"Unsupported qk_activation={attn.qk_activation!r}")

    if attn.qk_norm == "sum":
        q = sum_norm(q)
        k = sum_norm(k)
    elif attn.qk_norm == "l2":
        q = F.normalize(q.float(), p=2, dim=-1).to(q.dtype)
        k = F.normalize(k.float(), p=2, dim=-1).to(k.dtype)

    if attn.use_beta:
        beta = attn.b_proj(hidden_states).sigmoid()
    else:
        beta = torch.ones_like(q[..., 0])
    if attn.allow_neg_eigval:
        beta = beta * 2.0
    return {
        "kind": "delta",
        "q": q,
        "k": k,
        "v": v,
        "beta": beta,
        "state_log_decay": None,
        "d_decay_log": osgm_decay_log(attn, hidden_states, None),
    }


def project_gdn_inputs(attn, hidden_states: torch.Tensor):
    from fla.ops.gated_delta_rule.gate import naive_gdn_gate

    if attn.use_short_conv:
        q, _ = attn.q_conv1d(
            x=attn.q_proj(hidden_states),
            cache=None,
            output_final_state=False,
            cu_seqlens=None,
        )
        k, _ = attn.k_conv1d(
            x=attn.k_proj(hidden_states),
            cache=None,
            output_final_state=False,
            cu_seqlens=None,
        )
        v, _ = attn.v_conv1d(
            x=attn.v_proj(hidden_states),
            cache=None,
            output_final_state=False,
            cu_seqlens=None,
        )
    else:
        q = F.silu(attn.q_proj(hidden_states))
        k = F.silu(attn.k_proj(hidden_states))
        v = F.silu(attn.v_proj(hidden_states))

    q, k = map(lambda x: rearrange(x, "... (h d) -> ... h d", d=attn.head_k_dim), (q, k))
    v = rearrange(v, "... (h d) -> ... h d", d=attn.head_v_dim)
    q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6).to(q.dtype)
    k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6).to(k.dtype)

    beta = attn.b_proj(hidden_states).sigmoid()
    if attn.allow_neg_eigval:
        beta = beta * 2.0
    state_log_decay = naive_gdn_gate(attn.a_proj(hidden_states), attn.A_log, attn.dt_bias)

    return {
        "kind": "gdn",
        "q": q,
        "k": k,
        "v": v,
        "beta": beta,
        "state_log_decay": state_log_decay,
        "d_decay_log": osgm_decay_log(attn, hidden_states, state_log_decay),
    }


def project_kda_inputs(attn, hidden_states: torch.Tensor):
    from einops import repeat
    from fla.ops.kda.gate import naive_kda_gate, naive_kda_lowerbound_gate

    if attn.use_short_conv:
        q, _ = attn.q_conv1d(
            x=attn.q_proj(hidden_states),
            cache=None,
            output_final_state=False,
            cu_seqlens=None,
        )
        k, _ = attn.k_conv1d(
            x=attn.k_proj(hidden_states),
            cache=None,
            output_final_state=False,
            cu_seqlens=None,
        )
        v, _ = attn.v_conv1d(
            x=attn.v_proj(hidden_states),
            cache=None,
            output_final_state=False,
            cu_seqlens=None,
        )
    else:
        q = F.silu(attn.q_proj(hidden_states))
        k = F.silu(attn.k_proj(hidden_states))
        v = F.silu(attn.v_proj(hidden_states))

    gate_raw = attn.f_proj(hidden_states)
    beta = attn.b_proj(hidden_states).sigmoid()
    q, k, gate_raw = (
        rearrange(x, "... (h d) -> ... h d", d=attn.head_k_dim)
        for x in (q, k, gate_raw)
    )
    v = rearrange(v, "... (h d) -> ... h d", d=attn.head_v_dim)

    if attn.num_v_heads > attn.num_heads:
        groups = attn.num_v_heads // attn.num_heads
        q, k, gate_raw = (
            repeat(x, "... h d -> ... (h g) d", g=groups)
            for x in (q, k, gate_raw)
        )
        beta = repeat(beta, "... h -> ... (h g)", g=groups)

    if attn.allow_neg_eigval:
        beta = beta * 2.0
    q = F.normalize(q.float(), p=2, dim=-1, eps=1e-6).to(q.dtype)
    k = F.normalize(k.float(), p=2, dim=-1, eps=1e-6).to(k.dtype)
    if attn.lower_bound is None:
        state_log_decay = naive_kda_gate(gate_raw, attn.A_log, attn.dt_bias)
    else:
        state_log_decay = naive_kda_lowerbound_gate(
            gate_raw,
            attn.A_log,
            attn.dt_bias,
            lower_bound=attn.lower_bound,
        )

    return {
        "kind": "kda",
        "q": q,
        "k": k,
        "v": v,
        "beta": beta,
        "state_log_decay": state_log_decay,
        "d_decay_log": osgm_decay_log(attn, hidden_states, None),
    }


def osgm_decay_log(attn, hidden_states: torch.Tensor, state_log_decay: torch.Tensor | None):
    if not bool(getattr(attn, "use_osgm", False)):
        return None
    mode = getattr(attn, "osgm_decay_mode", "none")
    if mode == "data_dependent":
        if getattr(attn, "osgm_d_decay_source", "osgm") == "gdn":
            return state_log_decay
        if hasattr(attn, "osgm_a_proj"):
            return F.logsigmoid(attn.osgm_a_proj(hidden_states)).float()
        return None
    return None


def initial_d_state(attn, batch_size: int, num_heads: int, key_dim: int, device) -> torch.Tensor:
    if bool(getattr(attn, "use_osgm", False)):
        return attn.initial_scale.unsqueeze(0).expand(batch_size, -1, -1).contiguous().float()
    return torch.ones(batch_size, num_heads, key_dim, device=device, dtype=torch.float32)


def effective_d(d_curr: torch.Tensor, mode: str) -> torch.Tensor:
    if mode in ("base", "original"):
        return d_curr
    if mode in ("ones", "d=1"):
        return torch.ones_like(d_curr)
    if mode in ("mean", "d=mean"):
        return d_curr.mean(dim=-1, keepdim=True).expand_as(d_curr)
    if mode in ("shuffle", "d=shuffle"):
        return torch.flip(d_curr, dims=[-1])
    raise ValueError(f"Unknown d ablation mode {mode!r}")


def apply_state_decay(state: torch.Tensor, log_decay: torch.Tensor | None) -> torch.Tensor:
    if log_decay is None:
        return state
    decay = log_decay.float().exp()
    if decay.ndim == 2:
        return state * decay[..., None, None]
    if decay.ndim == 3:
        return state * decay[..., None]
    raise ValueError(f"Unexpected decay shape {tuple(log_decay.shape)}")


def update_osgm_d(attn, d_curr: torch.Tensor, k_t: torch.Tensor, beta_t: torch.Tensor, d_decay_log_t):
    if not bool(getattr(attn, "use_osgm", False)):
        return d_curr

    eta, use_denominator, d_min, d_max = resolve_osgm_defaults(attn, True)
    k_sq = k_t.square()
    inner = (d_curr * k_sq).sum(dim=-1, keepdim=True)
    kind = attention_kind(attn)
    beta_aware = (
        kind in ("gdn", "kda")
        and bool(getattr(attn, "osgm_post_gate_regret", False) or getattr(attn, "osgm_beta_aware", False))
    )
    if beta_aware:
        beta_v = beta_t.unsqueeze(-1).float()
        term = beta_v * (1.0 - beta_v * inner)
    else:
        term = 1.0 - inner
        if use_denominator:
            term = term / (k_sq.sum(dim=-1, keepdim=True) + 1e-5)

    mode = getattr(attn, "osgm_decay_mode", "none")
    if mode == "none":
        d_base = d_curr
    elif mode in ("learnable", "constant"):
        gamma = torch.sigmoid(attn.osgm_gamma_log).view(1, -1, 1).float()
        d_base = gamma * d_curr
    elif mode == "data_dependent":
        if d_decay_log_t is None:
            d_base = d_curr
        else:
            d_base = d_decay_log_t.float().exp().unsqueeze(-1) * d_curr
    else:
        raise NotImplementedError(f"Unsupported OSGM decay mode {mode!r}")

    d_next = d_base + eta * term * k_sq
    if d_min is not None and d_max is not None:
        d_next = d_next.clamp(min=d_min, max=d_max)
    return d_next


def relative_bins(attention_mask: torch.Tensor, n_bins: int) -> torch.Tensor:
    lengths = attention_mask.long().sum(dim=1).clamp_min(1)
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    denom = (lengths - 1).clamp_min(1).unsqueeze(1)
    bins = torch.div(positions * n_bins, denom + 1, rounding_mode="floor")
    return bins.clamp(max=n_bins - 1)


@torch.no_grad()
def diagnose_attention(
    attn,
    hidden_states: torch.Tensor,
    token_mask: torch.Tensor,
    rel_bin_ids: torch.Tensor,
    active_stats: list[ActiveStats],
    layer_idx: int,
) -> torch.Tensor:
    projected = project_recurrent_inputs(attn, hidden_states)
    kind = projected["kind"]
    q = projected["q"]
    k = projected["k"]
    v = projected["v"]
    beta = projected["beta"]
    state_log_decay = projected["state_log_decay"]
    d_decay_log = projected["d_decay_log"]
    B, T, H, K = k.shape
    V = v.shape[-1]
    states = {
        stats.label: torch.zeros(B, H, K, V, device=k.device, dtype=torch.float32)
        for stats in active_stats
    }
    outputs = torch.empty(B, T, H, V, device=k.device, dtype=torch.float32)

    use_osgm = bool(getattr(attn, "use_osgm", False))
    d_curr = initial_d_state(attn, B, H, K, k.device)

    scale = K ** -0.5
    eps = 1e-20
    token_mask = token_mask.bool()

    for t in range(T):
        k_t = k[:, t].float()
        q_t = q[:, t].float() * scale
        v_t = v[:, t].float()
        beta_t = beta[:, t].float()
        log_decay_t = state_log_decay[:, t] if state_log_decay is not None else None

        for stats in active_stats:
            state = apply_state_decay(states[stats.label], log_decay_t)
            d_t = effective_d(d_curr, stats.d_mode)

            pred_pre = torch.einsum("bhk,bhkv->bhv", k_t, state)
            residual = v_t - pred_pre
            pre = 0.5 * residual.square().sum(dim=-1)

            align = (d_t * k_t.square()).sum(dim=-1)
            beta_align = beta_t * align
            factor = 1.0 - beta_align
            post = pre * factor.square()
            q_ratio = torch.where(pre > eps, post / pre.clamp_min(eps), torch.ones_like(pre))

            valid_b = token_mask[:, t]
            if valid_b.any():
                q_valid = q_ratio[valid_b]
                pre_valid = pre[valid_b]
                post_valid = post[valid_b]
                align_valid = align[valid_b]
                beta_align_valid = beta_align[valid_b]
                stats.global_stats.add_tensor(q_valid, pre_valid, post_valid, align_valid, beta_align_valid)
                stats.task_stats.add_tensor(q_valid, pre_valid, post_valid, align_valid, beta_align_valid)
                stats.layer_stats[layer_idx].add_tensor(q_valid, pre_valid, post_valid, align_valid, beta_align_valid)
                stats.abs_curve.add(t, q_valid, pre_valid, post_valid, align_valid, beta_align_valid)
                for b_idx in torch.where(valid_b)[0].tolist():
                    rb = int(rel_bin_ids[b_idx, t])
                    stats.rel_curve.add(
                        rb,
                        q_ratio[b_idx],
                        pre[b_idx],
                        post[b_idx],
                        align[b_idx],
                        beta_align[b_idx],
                    )

            state = state + (d_t * k_t).unsqueeze(-1) * (residual * beta_t.unsqueeze(-1)).unsqueeze(-2)
            states[stats.label] = state
            if stats.d_mode == "base":
                outputs[:, t] = torch.einsum("bhk,bhkv->bhv", q_t, state)

        if use_osgm:
            d_decay_log_t = d_decay_log[:, t] if d_decay_log is not None else None
            d_curr = update_osgm_d(attn, d_curr, k_t, beta_t, d_decay_log_t)

    o = outputs.to(hidden_states.dtype)
    if kind == "kda":
        g = rearrange(attn.g_proj(hidden_states), "... (h d) -> ... h d", d=attn.head_v_dim)
        o = attn.o_norm(o, g)
    elif attn.use_gate:
        g = rearrange(attn.g_proj(hidden_states), "... (h d) -> ... h d", d=attn.head_v_dim)
        o = attn.o_norm(o, g)
    else:
        o = attn.o_norm(o)
    o = rearrange(o, "b t h d -> b t (h d)")
    return attn.o_proj(o)


@torch.no_grad()
def diagnose_batch(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    active_stats: list[ActiveStats],
) -> None:
    backbone = model.model
    hidden_states = backbone.embeddings(input_ids)
    rel_bin_ids = relative_bins(attention_mask, active_stats[0].rel_curve.n)

    for layer_idx, layer in enumerate(backbone.layers):
        residual = hidden_states
        hidden_states = layer.attn_norm(hidden_states)
        hidden_states = diagnose_attention(
            layer.attn,
            hidden_states,
            attention_mask,
            rel_bin_ids,
            active_stats,
            layer_idx,
        )
        if layer.config.fuse_norm:
            hidden_states, residual = layer.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = layer.mlp_norm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(out_dir: Path, rel_rows: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is best-effort.
        print(f"Skipping plot: {exc}")
        return

    by_model: dict[str, list[dict]] = {}
    for row in rel_rows:
        if row.get("task") is not None:
            continue
        by_model.setdefault(row["model"], []).append(row)
    if not by_model:
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for model_label, rows in by_model.items():
        rows = sorted(rows, key=lambda r: int(r["relative_bin"]))
        xs = [int(r["relative_bin"]) for r in rows]
        ys = [float(r["q_geo_mean"]) for r in rows]
        ax.plot(xs, ys, marker="o", linewidth=1.7, markersize=2.6, label=model_label)
    ax.set_xlabel("Relative position bin")
    ax.set_ylabel("Geometric mean residual ratio q_t")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "residual_ratio_relative_curve.png", dpi=220)
    fig.savefig(out_dir / "residual_ratio_relative_curve.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True, help="label=checkpoint_path")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--tasks",
        default="based_fda_twice,based_swde_twice,based_squad_twice",
        help="comma-separated JRT task names",
    )
    parser.add_argument("--max_samples_per_task", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=1000)
    parser.add_argument("--answer_length", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=1064)
    parser.add_argument("--relative_bins", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--d_ablations",
        default="",
        help="comma-separated OSGM shadow ablations to add for use_osgm models: ones,mean,shuffle",
    )
    args = parser.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_specs = parse_model_specs(args.model)
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    d_ablations = [x.strip() for x in args.d_ablations.split(",") if x.strip()]

    tokenizer = AutoTokenizer.from_pretrained(model_specs[0][1])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    prompts_by_task = build_prompts(
        tokenizer=tokenizer,
        tasks=tasks,
        max_samples_per_task=args.max_samples_per_task,
        context_length=args.context_length,
        answer_length=args.answer_length,
        seed=args.seed,
    )
    with (out_dir / "prompts_meta.json").open("w") as f:
        json.dump({k: len(v) for k, v in prompts_by_task.items()}, f, indent=2)

    summary = {
        "tasks": tasks,
        "max_samples_per_task": args.max_samples_per_task,
        "context_length": args.context_length,
        "answer_length": args.answer_length,
        "max_length": args.max_length,
        "seed": args.seed,
        "models": {},
    }
    all_abs_rows = []
    all_rel_rows = []
    all_layer_rows = []
    all_task_rows = []

    for model_label, model_path in model_specs:
        print(f"\nLoading {model_label}: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(args.device).eval()
        model.config.use_cache = False
        n_layers = len(model.model.layers)
        has_osgm = any(bool(getattr(layer.attn, "use_osgm", False)) for layer in model.model.layers)
        variant_modes = [("base", model_label)]
        if has_osgm:
            for mode in d_ablations:
                variant_modes.append((mode, f"{model_label}/{mode}"))

        variant_state = {}
        for mode, label in variant_modes:
            variant_state[label] = {
                "mode": mode,
                "global_stats": ScalarStats(),
                "abs_curve": CurveStats(args.max_length),
                "rel_curve": CurveStats(args.relative_bins),
                "layer_stats": [ScalarStats() for _ in range(n_layers)],
                "task_stats": {},
            }

        for task in tasks:
            prompts = prompts_by_task[task]
            for state in variant_state.values():
                state["task_stats"][task] = ScalarStats()
            print(f"Diagnosing {model_label} on {task}: {len(prompts)} prompts")
            for i in tqdm(range(0, len(prompts), args.batch_size), desc=f"{model_label}:{task}"):
                batch_prompts = prompts[i : i + args.batch_size]
                encoded = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                )
                input_ids = encoded["input_ids"].to(args.device)
                attention_mask = encoded["attention_mask"].to(args.device)
                active_stats = [
                    ActiveStats(
                        label=label,
                        d_mode=state["mode"],
                        global_stats=state["global_stats"],
                        task_stats=state["task_stats"][task],
                        abs_curve=state["abs_curve"],
                        rel_curve=state["rel_curve"],
                        layer_stats=state["layer_stats"],
                    )
                    for label, state in variant_state.items()
                ]
                diagnose_batch(
                    model,
                    input_ids,
                    attention_mask,
                    active_stats,
                )

        for label, state in variant_state.items():
            model_summary = state["global_stats"].as_dict()
            model_summary["checkpoint"] = model_path
            model_summary["num_layers"] = n_layers
            model_summary["base_model"] = model_label
            model_summary["d_mode"] = state["mode"]
            summary["models"][label] = model_summary

            all_abs_rows.extend(state["abs_curve"].rows("token_index", label))
            all_rel_rows.extend(state["rel_curve"].rows("relative_bin", label))
            for layer_idx, stats in enumerate(state["layer_stats"]):
                row = {"model": label, "layer": layer_idx}
                row.update(stats.as_dict())
                all_layer_rows.append(row)
            for task, stats in state["task_stats"].items():
                row = {"model": label, "task": task}
                row.update(stats.as_dict())
                all_task_rows.append(row)

        del model
        torch.cuda.empty_cache()

    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    write_csv(out_dir / "token_curve.csv", all_abs_rows)
    write_csv(out_dir / "relative_curve.csv", all_rel_rows)
    write_csv(out_dir / "layer_summary.csv", all_layer_rows)
    write_csv(out_dir / "task_summary.csv", all_task_rows)
    plot_curves(out_dir, all_rel_rows)

    print("\nSummary")
    for model_label, model_summary in summary["models"].items():
        print(
            f"{model_label}: q_geo={model_summary['q_geo_mean']:.6g}, "
            f"q_mean={model_summary['q_mean']:.6g}, "
            f"pre={model_summary['pre_loss_mean']:.6g}, "
            f"post={model_summary['post_loss_mean']:.6g}, "
            f"align={model_summary['align_mean']:.6g}, "
            f"beta_align={model_summary['beta_align_mean']:.6g}"
        )
    print(f"Saved diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
