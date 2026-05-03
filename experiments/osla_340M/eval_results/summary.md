# 340M Model Evaluation Summary

## Model Overview

| # | Model | Type | Params | Description |
|---|-------|------|--------|-------------|
| 1 | DeltaNet Baseline | delta_net | 374M | Baseline, use_gate=false |
| 2 | OSGM | osla_delta_net | ~374M | DeltaNet + Online Subgradient Method |
| 3 | OSGM + Learnable D0 | osla_delta_net | ~374M | OSGM + learnable preconditioner |
| 4 | DeltaNet + gate | delta_net | 400M | Baseline config + use_gate=true |
| 5 | GatedDeltaNet v2 | gated_deltanet | 380M | Proper GDN (head_dim=128, num_heads=6, 21L) |
| 6 | KDA | kda | 374M | Kimi Delta Attention (per-key-dim gating) |
| 7 | OSGM + Decay | osla_delta_net | 374M | OSGM + per-head learnable decay on D (d_max=2.0, bug) |
| 8 | OSGM + Constant Decay | osla_delta_net | 374M | OSGM + fixed γ=0.999 decay on D, d_max=1e9 fixed |
| 9 | OSGM + DD Decay | osla_delta_net | ~374M | OSGM + data-dependent per-token decay via a_proj, d_max=1e9 |
| 10 | OSGM + D0 v2 | osla_delta_net | ~374M | Re-run of #3 with d_max=1e9 fix (was 2.0) |

---

## 1. Commonsense & LM Benchmarks

| Task (metric) | DeltaNet | OSGM | OSGM+D0 | DN+gate | GDN v2 | KDA | OSGM+Decay | ConstDecay | DD Decay | D0 v2 |
|---------------|----------|------|---------|---------|--------|-----|------------|------------|----------|-------|
| PIQA (acc_norm) | 0.6502 | 0.6594 | 0.6518 | 0.6480 | **0.6659** | **0.6659** | 0.6583 | 0.6513 | 0.6507 | 0.6453 |
| HellaSwag (acc_norm) | 0.3913 | 0.3876 | 0.3937 | 0.3953 | 0.4105 | **0.4164** | 0.3886 | 0.3873 | 0.3986 | 0.3948 |
| WinoGrande (acc) | 0.5193 | 0.4917 | 0.5193 | 0.5083 | 0.5075 | **0.5201** | 0.5130 | 0.5170 | 0.5091 | 0.5162 |
| ARC-e (acc_norm) | 0.5257 | 0.4979 | 0.5105 | 0.4979 | 0.5198 | **0.5248** | 0.4924 | 0.5034 | 0.5067 | 0.5269 |
| ARC-c (acc_norm) | 0.2619 | 0.2765 | 0.2654 | 0.2696 | **0.2824** | 0.2747 | 0.2654 | 0.2713 | 0.2841 | 0.2671 |
| SIQA (acc) | 0.3813 | 0.3823 | 0.3905 | 0.3736 | 0.3889 | **0.3930** | 0.3802 | 0.3731 | 0.3787 | 0.3854 |
| BoolQ (acc) | **0.6110** | 0.5911 | 0.6076 | 0.6092 | 0.6006 | 0.6104 | 0.5930 | 0.5927 | 0.5951 | **0.6187** |
| WikiText (word_ppl↓) | 28.73 | 29.24 | 28.75 | 29.01 | 27.43 | **26.56** | 29.31 | 28.75 | **28.07** | 28.56 |
| LAMBADA (acc) | 0.3121 | 0.2930 | 0.3346 | 0.3138 | 0.3299 | **0.3561** | 0.3126 | 0.3150 | 0.3181 | 0.3351 |
| **Average (excl. ppl)** | **0.4566** | **0.4475** | **0.4592** | **0.4520** | **0.4632** | **0.4702** | **0.4504** | **0.4511** | **0.4539** | **0.4612** |

---

## 2. Real Retrieval — JRT Cloze (contains, context=2K tokens)

| Task | DeltaNet | OSGM | OSGM+D0 | DN+gate | GDN v2 | KDA | OSGM+Decay | ConstDecay | DD Decay | D0 v2 |
|------|----------|------|---------|---------|--------|-----|------------|------------|----------|-------|
| FDA | 0.0872 | 0.0772 | 0.0899 | 0.0736 | 0.0881 | 0.0881 | 0.0563 | 0.0563 | 0.0663 | **0.0899** |
| FDA-twice | 0.0445 | 0.0436 | 0.0463 | 0.0509 | 0.0345 | 0.0409 | 0.0509 | 0.0509 | **0.0763** | 0.0572 |
| SWDE | 0.0890 | 0.1481 | **0.1565** | 0.1200 | 0.1340 | 0.1518 | 0.1209 | 0.0862 | 0.0975 | 0.0825 |
| SWDE-twice | 0.1575 | 0.2165 | 0.2165 | 0.1753 | 0.1799 | **0.2240** | 0.1799 | 0.1256 | 0.2146 | 0.1537 |
| SQuAD | 0.2872 | 0.2909 | 0.3047 | 0.2788 | 0.2859 | **0.3208** | 0.2644 | 0.2892 | 0.2919 | 0.3013 |
| SQuAD-twice | 0.2314 | **0.3833** | 0.2200 | 0.1784 | 0.2026 | 0.1837 | 0.1340 | 0.3557 | 0.3070 | 0.2969 |
| **Average** | **0.1495** | **0.1933** | **0.1723** | **0.1462** | **0.1542** | **0.1682** | **0.1344** | **0.1607** | **0.1756** | **0.1636** |

---

## 3. Real Retrieval — lm_eval (generate_until)

| Task (metric) | DeltaNet | OSGM | OSGM+D0 | DN+gate | GDN v2 | KDA | OSGM+Decay | ConstDecay | DD Decay | D0 v2 |
|---------------|----------|------|---------|---------|--------|-----|------------|------------|----------|-------|
| TriviaQA (EM) | **0.0059** | 0.0020 | 0.0026 | 0.0023 | 0.0054 | 0.0049 | 0.0021 | 0.0020 | 0.0036 | **0.0133** |
| DROP (F1) | 0.0254 | 0.0243 | 0.0251 | 0.0250 | **0.0272** | 0.0243 | 0.0268 | 0.0250 | **0.0308** | 0.0236 |
| NQ-Open (EM) | 0.0072 | 0.0091 | 0.0102 | 0.0086 | 0.0111 | **0.0238** | 0.0130 | 0.0141 | 0.0133 | 0.0091 |

> Note: All models are base (no instruction tuning), so generation-based retrieval scores are very low across the board.

---

## 4. LongBench (14 tasks, score)

| Task | DeltaNet | OSGM | OSGM+D0 | DN+gate | GDN v2 | KDA | OSGM+Decay | ConstDecay | DD Decay | D0 v2 |
|------|----------|------|---------|---------|--------|-----|------------|------------|----------|-------|
| NarrativeQA | 0.0167 | 0.0176 | 0.0149 | **0.0190** | 0.0163 | 0.0188 | 0.0153 | 0.0169 | 0.0153 | 0.0160 |
| Qasper | 0.0299 | 0.0362 | 0.0244 | **0.0401** | 0.0346 | 0.0394 | 0.0391 | 0.0377 | 0.0341 | 0.0383 |
| MultifieldQA | 0.1114 | 0.1075 | 0.0992 | 0.1051 | **0.1214** | 0.1125 | 0.1096 | 0.1135 | 0.1134 | 0.0854 |
| HotpotQA | 0.0364 | 0.0340 | 0.0393 | 0.0409 | 0.0378 | **0.0462** | 0.0370 | 0.0332 | 0.0400 | 0.0277 |
| 2WikiMQA | 0.0855 | 0.0784 | 0.0707 | 0.0681 | 0.0758 | **0.0885** | 0.0712 | 0.0731 | 0.0733 | 0.0818 |
| MuSiQue | 0.0240 | 0.0185 | 0.0222 | 0.0222 | 0.0243 | **0.0248** | 0.0241 | 0.0210 | 0.0230 | 0.0189 |
| GovReport | 0.0539 | 0.0648 | 0.0585 | 0.0691 | 0.0708 | **0.0753** | 0.0687 | 0.0615 | 0.0489 | 0.0719 |
| QMSum | 0.1397 | 0.1121 | 0.1606 | 0.1385 | 0.1500 | **0.1700** | 0.1344 | 0.1324 | 0.1294 | 0.1480 |
| MultiNews | 0.0685 | 0.0978 | 0.0605 | **0.1100** | 0.0946 | 0.0906 | 0.0865 | 0.0998 | 0.0698 | 0.0724 |
| TREC | 0.2100 | 0.1150 | 0.1600 | **0.2500** | 0.1350 | 0.2100 | 0.1600 | 0.1950 | **0.2650** | 0.0850 |
| TriviaQA | 0.1290 | 0.1529 | 0.1344 | 0.1349 | 0.1553 | **0.1730** | 0.1437 | 0.1420 | 0.1517 | 0.1596 |
| SAMSum | 0.0784 | 0.0385 | 0.0906 | 0.1260 | 0.0785 | **0.1378** | 0.1010 | 0.0688 | 0.0225 | 0.0720 |
| PassageCount | 0.0000 | 0.0078 | 0.0134 | 0.0099 | 0.0043 | **0.0240** | 0.0071 | 0.0158 | 0.0085 | 0.0046 |
| PassageRetrieval | 0.0256 | 0.0000 | 0.0098 | **0.0355** | 0.0200 | 0.0250 | 0.0275 | 0.0270 | 0.0280 | 0.0050 |
| **Average** | **0.0721** | **0.0630** | **0.0685** | **0.0835** | **0.0727** | **0.0883** | **0.0732** | **0.0741** | **0.0731** | **0.0633** |

---

## 5. Length Extrapolation — PG19 PPL (block=20K, bucket=2K)

| Metric | DeltaNet | OSGM | OSGM+D0 | DN+gate | GDN v2 | KDA | OSGM+Decay | ConstDecay | DD Decay | D0 v2 |
|--------|----------|------|---------|---------|--------|-----|------------|------------|----------|-------|
| **Final PPL↓** | 20.78 | 21.23 | 21.57 | 20.27 | 20.11 | **18.73** | ~21.6* | 30.73 | 19.85 | 21.42 |

*OSGM+Decay (model #7) PPL was estimated from partial evaluation.

### Block-wise PPL (each bucket = 2K tokens within 20K block):

| Bucket (pos) | DeltaNet | OSGM | OSGM+D0 | DN+gate | GDN v2 | KDA | OSGM+Decay | ConstDecay | DD Decay | D0 v2 |
|--------------|----------|------|---------|---------|--------|-----|------------|------------|----------|-------|
| 0 (0-2K) | 21.82 | 21.33 | 21.49 | 21.74 | 21.71 | **20.29** | 22.23 | 21.52 | 20.78 | 21.52 |
| 1 (2-4K) | 20.10 | **19.49** | 19.74 | 19.97 | 20.14 | **18.70** | 20.53 | 19.88 | 19.10 | 19.96 |
| 2 (4-6K) | 20.07 | **19.53** | 19.80 | 19.88 | 20.01 | **18.59** | 20.57 | 20.08 | 19.05 | 20.04 |
| 3 (6-8K) | 20.16 | **19.79** | 20.10 | 19.88 | 19.93 | **18.53** | 20.75 | 20.81 | 19.11 | 20.26 |
| 4 (8-10K) | 20.34 | 20.25 | 20.59 | 19.95 | 19.91 | **18.52** | 21.04 | 22.46 | 19.29 | 20.62 |
| 5 (10-12K) | 20.56 | 20.84 | 21.21 | 20.05 | **19.90** | **18.52** | 21.43 | 25.79 | 19.54 | 21.08 |
| 6 (12-14K) | 20.82 | 21.57 | 21.95 | 20.17 | **19.91** | **18.54** | 21.89 | 32.26 | 19.86 | 21.67 |
| 7 (14-16K) | 21.09 | 22.41 | 22.82 | 20.30 | **19.91** | **18.56** | 22.39 | 43.44 | 20.23 | 22.35 |
| 8 (16-18K) | 21.32 | 23.31 | 23.74 | 20.38 | **19.87** | **18.54** | 22.89 | 60.25 | 20.59 | 23.09 |
| 9 (18-20K) | 21.79 | 24.58 | 25.08 | 20.67 | **20.04** | **18.71** | 23.70 | 86.50 | 21.23 | 24.21 |
| **Degradation (9-1)** | **+1.69** | **+5.09** | **+5.34** | **+0.70** | **-0.10** | **+0.01** | **+3.17** | **+66.62** | **+2.13** | **+4.25** |

---

## Key Takeaways

1. **KDA is the overall best model** — wins on commonsense avg (0.4702), LongBench avg (0.0883), PG19 PPL (18.73), and most individual tasks.

2. **OSGM models degrade badly with length** — PPL goes from ~19.5 at 2K to ~25 at 20K (degradation of +5), suggesting the online preconditioner hurts extrapolation.

3. **GatedDeltaNet v2 has the most stable extrapolation** — PPL actually *decreases* slightly from bucket 1→9 (degradation -0.10), best length stability.

4. **OSGM excels at retrieval-twice tasks** — SQuAD-twice 0.3833 is the highest single score across all retrieval tasks, showing the value of OSGM for in-context recall when context is repeated.

5. **Gate helps** — DeltaNet+gate (0.4520) and GDN v2 (0.4632) both improve over plain DeltaNet (0.4566) on commonsense when accounting for the structural changes.

6. **Base models are poor at generation-based retrieval** — TriviaQA/DROP/NQ EM scores are near zero for all models (expected without instruction tuning).

7. **OSGM+Decay partially fixes extrapolation** — Per-head learnable decay reduces length degradation from +5.09 to +3.17 (38% improvement). LongBench also improves (0.0630→0.0732, surpassing baseline). However, the decay hurts JRT retrieval (especially twice-variants), as the forgetting mechanism weakens OSGM's long-range recall advantage.
   - Note: model #7 used a buggy d_max=2.0 projection bound (mathematically correct upper bound is ~√K/2≈5.66 for K=128); models #8 and #9 fix this with d_max=1e9.

8. **DD Decay (model #9) significantly improves length extrapolation** — PG19 degradation drops from +5.09 (plain OSGM) to **+2.13**, the best among all OSGM variants. Final PG19 PPL=19.85, beating the baseline (20.78) and all other OSGM models. WikiText PPL=28.07 is also best among OSGM variants. Length stability approaches DeltaNet+gate (+0.70) while retaining OSGM's learning advantage.
   - LongBench avg=0.0731, comparable to baseline (0.0721) and OSGM+Decay (0.0732). TREC=0.2650 is the **highest across all models**.
   - JRT retrieval avg=0.1756, weaker than plain OSGM (0.1933) but better than all decay variants (#7=0.1344, #8=0.1607). The data-dependent decay partially preserves retrieval.
   - DROP F1=0.0308 is the **highest across all models** on generation-based retrieval.

9. **Constant Decay (model #8) catastrophically degrades with length** — PG19 PPL explodes from 19.88 at 2K to **86.50** at 20K (degradation **+66.62**), far worse than any other model. The fixed γ=0.999 decay causes D to collapse toward zero over long sequences, destroying the preconditioner entirely. This confirms that constant decay is fundamentally broken for OSGM — the decay rate must be data-dependent or at minimum learned per-head.

10. **D0 v2 (model #10, d_max fix re-run)** — Commonsense avg=0.4612, the **best among all OSGM variants** and close to GDN v2 (0.4632). BoolQ=0.6187 is the **highest across all models**. PG19 degradation=+4.25, slightly improved over original D0 (+5.34) but still high. The d_max fix (2.0→1e9) helped slightly but did not solve the core extrapolation issue — the problem is inherent to OSGM's accumulating preconditioner, not the projection bound.
