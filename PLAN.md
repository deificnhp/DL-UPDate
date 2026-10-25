# PLAN.md — US100 4-Hour Quantile Forecasting Model

**Document type:** Project Management Plan, structured on PMBOK knowledge areas
**Version:** 1.0
**Date:** 2026-10-19
**Project codename:** NQ4H-Q

---

## 0. Executive Summary

This project will build a compact deep learning model that forecasts the distribution of the next 4-hour log-return of US100 (Nasdaq-100). It outputs seven quantiles, a directional signal derived from those quantiles, and a calibrated confidence score. The main design choices follow from the research below.

1. **Volatility is predictable. Direction mostly is not.** Most of the achievable skill at the 4-hour horizon will come from predicting the scale and shape of the return distribution: volatility clustering, intraday seasonality and fat tails. Very little will come from predicting the sign. The plan therefore treats calibrated quantiles as the primary deliverable and treats directional edge as secondary. That framing gives the project a high probability of success.
2. **No architecture reliably wins.** A large 2025 re-benchmark trained more than 3,500 networks on 14 datasets. It found that no recent model consistently beats the others, and that small changes to the experimental setup reverse the rankings [Source](https://arxiv.org/html/2502.14045v1). Simple linear models have also been shown to match or beat many complex Transformers [Source](https://arxiv.org/abs/2205.13504). The plan therefore uses one small, well-regularised primary model and holds it to strong non-deep-learning baselines.
3. **The primary model is a small PatchTST-style Transformer encoder with a non-crossing quantile head.** Each patch is one trading session (6 bars). The model has roughly 120k parameters, and an ensemble of K copies is trained in parallel on the GPU.
4. **The 6GB of VRAM goes to ensemble members, not a bigger model.** A dataset of about 25k samples with a low signal-to-noise ratio does not support a large model. The VRAM is instead filled with a vectorised deep ensemble of about 16–32 members, which reduces variance and gives a natural uncertainty signal for the confidence score. The target is a peak of 80–88% of 6GB, tuned automatically.
5. **Validation is an expanding walk-forward with purging and embargo.** Hyperparameters are tuned only on early development folds, and the most recent 9 months are kept as an untouched lockbox.

---

## 1. Integration Management

### 1.1 Project Charter

| Item | Definition |
|---|---|
| **Project purpose** | Produce a calibrated probabilistic forecast of the next US100 4H bar for risk sizing and directional decision support. |
| **Primary objective** | Forecast the quantiles τ ∈ {0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95} of r₍t+1₎ = ln(C₍t+1₎ / C₍t₎) with lower pinball loss than strong baselines, and with statistically valid coverage. |
| **Secondary objectives** | (a) A directional signal ∈ {−1, 0, +1} derived from the quantile-implied P(up). (b) A confidence score ∈ [0.5, 1.0] defined as the calibrated probability that the signal's direction is correct. |
| **Constraints** | A single GPU with 6GB VRAM. One primary engineer/researcher. 8-week timeline. Only OHLCV-derived features in the MVP. |
| **Sponsor** | Head of trading / research lead (decision authority at phase gates). |
| **Project manager / lead researcher** | Owner of this plan (accountable for delivery). |
| **High-level success criteria** | Defined in §12. In short, the model beats baselines on pinball loss, its coverage is valid, its confidence is monotonic, and it trains reliably within 6GB. |
| **Authorisation** | Proceed to Phase 1 when the sponsor approves this charter. |

### 1.2 Integrated Change Control

Any change to the feature set, the target definition, the validation protocol or the lockbox period requires a written change request (CR) in `CHANGELOG.md`. Each CR records its rationale and its expected effect on the success criteria. Changes to the lockbox definition after Phase 4 starts are prohibited. Every experiment is logged with its config hash, data hash, git commit and seed, so any result can be reproduced.

### 1.3 Assumptions

| ID | Assumption |
|---|---|
| A1 | Data comes from continuous E-mini Nasdaq-100 (NQ) futures, back-adjusted by ratio, with true exchange volume. CME Globex trades NQ from Sunday 6:00 p.m. to Friday 5:00 p.m. ET, with a daily one-hour maintenance break [Source](https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.html). A CFD "US100" feed may be used for deployment only if the same feed is used for training, because CFD tick volume and broker time zones differ. |
| A2 | 4H bars are anchored to the 18:00 ET session open (18–22, 22–02, 02–06, 06–10, 10–14, 14–17). That gives about 6 bars per session and about 1,500 bars per year, with DST handled in America/New_York time. |
| A3 | Predictions are made at the close of bar t and use only information available at that close. |
| A4 | Transaction cost assumption for economic evaluation: 1.0 bp per side for futures and a configurable 1.5–3 bp for CFDs. |
| A5 | The GPU is NVIDIA with compute capability ≥ 7.5. Ampere or newer uses bf16. Turing (e.g., RTX 2060 6GB) uses fp16 with GradScaler. |
| A6 | The return distribution is non-stationary. The model is retrained at least quarterly in production. |

---

## 2. Scope Management

### 2.1 In Scope

The project covers the following:

- A data pipeline from raw 1-minute or native 4H NQ data to clean, session-aligned 4H OHLCV.
- A causal feature pipeline (§9.3).
- A baseline suite (§9.5).
- The primary DL model with a vectorised ensemble and automatic VRAM tuning.
- Walk-forward evaluation, conformal calibration, the derivation of signal and confidence, and an evaluation report.
- A model card and an inference script that emits `{quantiles, p_up, signal, confidence}` per bar.

### 2.2 Out of Scope (MVP)

The MVP excludes order execution, portfolio construction, multi-asset models, news or sentiment and LLM features, tick or order-book data, multi-step horizons, reinforcement learning, and foundation-model fine-tuning. Cross-asset features (VIX/VXN, yields, DXY) and a macro-event calendar flag are **Phase-2 candidates**. They are admitted only through the ablation gate in §9.3.4.

### 2.3 Work Breakdown Structure (WBS)

| WBS | Work package | Deliverable |
|---|---|---|
| 1.0 | Project management | Charter, this plan, weekly status, CHANGELOG |
| 2.0 | Data | `data/nq_4h.parquet`, data-quality report |
| 3.0 | Features | `features.py` + leakage unit tests |
| 4.0 | Baselines | B0–B4 results on dev folds |
| 5.0 | Model | `model.py` (PatchTST-Q, batched ensemble), `vram_probe.py` |
| 6.0 | Training & HPO | `train.py`, Optuna study (≤ 40 trials) |
| 7.0 | Evaluation | Walk-forward report, DM tests, calibration plots |
| 8.0 | Calibration & signal | CQR offsets, isotonic confidence map, threshold θ |
| 9.0 | Lockbox & closure | Lockbox report, model card, lessons learned |

### 2.4 Scope Validation

The sponsor accepts each deliverable at the phase gates in §3. The final acceptance test is the lockbox evaluation against §12.

---

## 3. Schedule Management

### 3.1 Phased Timeline (8 weeks, single FTE)

| Phase | Weeks | Activities | Exit gate (must pass) |
|---|---|---|---|
| **P0 Initiation** | W1 (2 days) | Approve the charter, procure data, set up the environment | Charter signed; data access confirmed |
| **P1 Data** | W1–W2 | Ingest, session-align, roll-adjust, handle missing bars and outliers, build the DQ report | <0.1% missing bars after the session calendar; no unexplained gaps > 5σ |
| **P2 Features + Baselines** | W3 | Build the 11+2 features, leakage tests, baselines B0–B4 on dev folds | Leakage tests pass; baseline table produced |
| **P3 Model + VRAM** | W4–W5 | Implement PatchTST-Q, the batched ensemble, the VRAM probe, HPO on dev folds | Peak VRAM 80–88% with no OOM over 3 runs; beats B2 on dev folds |
| **P4 Walk-forward evaluation** | W6 | Run 7 evaluation folds (2019–2025), DM tests, ablations | Meets §12 S1–S2 on pooled OOS |
| **P5 Calibration + Signal** | W7 | CQR, isotonic confidence, θ selection, economic backtest | Confidence reliability monotone (S3) |
| **P6 Lockbox + Closure** | W8 | One-shot lockbox run (2026-01 → latest), model card, handover | §12 lockbox criteria; sponsor sign-off |

**Buffer:** 3 working days are held inside W8. If P4 fails its gate, the response is a single pre-defined iteration (§10, R6), not open-ended tuning.

### 3.2 Milestones

| Milestone | Date |
|---|---|
| M1 | Clean data frozen (end of W2) |
| M2 | Baselines locked (end of W3) |
| M3 | Hyperparameters frozen (end of W5); no further tuning after this point |
| M4 | Walk-forward report (end of W6) |
| M5 | Lockbox result and go/no-go (end of W8) |

---

## 4. Cost Management

| Item | Estimate | Note |
|---|---|---|
| GPU hardware | Already owned (6GB) | No cloud spend planned |
| Market data (NQ 1-min or 4H history 2007→present) | Vendor-dependent budget line | Confirm a licence that allows research use |
| Labour | 1 FTE × 8 weeks | Main cost driver |
| Contingency | 10% of schedule (the buffer) | |

Compute is not a cost driver. With all data resident on the GPU, one full walk-forward run is expected to take less than about an hour. This is an engineering estimate to be confirmed in P3.

---

## 5. Quality Management

### 5.1 Quality Policy

Every reported number is out-of-sample, produced by a causal pipeline, reproducible from a config hash, and compared against baselines with a significance test. The plan avoids "one lucky split" claims. The LTSF re-benchmark shows how fragile model rankings are and recommends statistical testing over leaderboard averages [Source](https://arxiv.org/html/2502.14045v1).

### 5.2 Evaluation Metrics

**Distributional (primary)**

- **Pinball loss.** The mean over quantiles and time of ρ_τ(y − q̂_τ), where ρ_τ(u) = u(τ − 1{u<0}). This is the proper scoring rule for quantiles [Source](https://www.lokad.com/pinball-loss-function-definition/). It is reported in raw return units (after de-scaling) and as a skill score versus each baseline.
- **CRPS approximation.** Computed as 2 × the average pinball loss over the 7 quantiles, as a single summary number.
- **Empirical coverage per quantile**, i.e. the hit rate P(y < q̂_τ). Also the width of the 50% and 90% central intervals, to measure sharpness.
- **Kupiec POF, Christoffersen conditional coverage and Engle–Manganelli DQ tests** on τ = 0.05 and 0.95. A recent study of deep quantile VaR on the S&P 500 stresses that lower pinball loss does not guarantee valid coverage or independent exceedances, so both must be tested [Source](https://www.mdpi.com/2227-7072/14/8/218).
- **Diebold–Mariano test** on the per-bar pinball-loss differential against each baseline, using HAC standard errors.

**Directional (secondary)**

- AUC and Brier score of p_up against the realised sign.
- Hit rate of the non-zero signals, and hit rate in the top 20% and top 10% confidence buckets.
- A reliability diagram of confidence against realised hit rate, plus the Spearman correlation across confidence deciles.

**Economic (reported, gating only for any future trading deployment)**

- Net-of-cost Sharpe, max drawdown and turnover of a one-bar-hold strategy that trades the sign signal.
- A Deflated Sharpe Ratio that accounts for the number of configurations tried. It corrects for selection bias under multiple testing and for non-normal returns [Source](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551).

### 5.3 Quality Control Activities

| Control | Method |
|---|---|
| Leakage test (unit) | For random t, recompute the features after deleting or shuffling all data after t. The features at t must be bit-identical. |
| Scaler leakage | Robust scalers (median/IQR) are fit on the training segment of each fold only. The cited VaR study uses the same train-only median/MAD protocol [Source](https://www.mdpi.com/2227-7072/14/8/218). |
| Label alignment test | Assert that target[t] equals ln(C[t+1]/C[t]) and that no feature uses index t+1. |
| Quantile crossing | Must be zero by construction (§9.4.3); asserted at inference. |
| Reproducibility | Fixed seeds, deterministic data order, results within ±0.3% pinball across 3 reruns. |
| Seed robustness | Evaluation metrics are reported as the ensemble result plus the spread across members. |

---

## 6. Resource Management

### 6.1 Human Resources

| Role | Allocation | Responsibility |
|---|---|---|
| Lead quant / ML engineer (PM) | 1.0 FTE | All technical work packages and the plan |
| Sponsor / trader | ~2 h/week | Gate decisions, cost assumptions, usability review |
| Peer reviewer (optional) | ~4 h total | Leakage and validation review before M4 |

### 6.2 Physical / Compute Resources

| Resource | Spec |
|---|---|
| GPU | 6GB VRAM (e.g., RTX 3060 Laptop / RTX 2060 / RTX 4050 Laptop) |
| CPU / RAM | ≥ 8 cores, ≥ 16GB RAM (32GB recommended), SSD |
| Software | Python 3.11/3.12, PyTorch 2.x, polars/pandas, LightGBM, statsmodels, `arch`, Optuna, MLflow (or CSV logging) |

### 6.3 GPU Resource Strategy (core constraint)

**Principle: fill the VRAM with useful parallelism, not with parameters.** The full training set (about 25k windows × 180 bars × 13 inputs) is only a few hundred MB even fully materialised, and a 120k-parameter model with Adam state needs only a few MB. A single model therefore cannot productively use 6GB. Making the model bigger to fill memory would increase overfitting on a low-SNR target. VRAM is allocated as follows.

| VRAM consumer | Planned share | Rationale |
|---|---|---|
| CUDA context + allocator headroom + fragmentation (+ display if attached) | ~0.7–1.0 GB (not allocatable) | Leaving this free prevents OOM |
| Entire dataset resident on GPU (raw feature matrix; windows built by indexing/`unfold`) | ~0.05–0.3 GB | Removes the dataloader bottleneck, so the GPU is never starved |
| **K-member vectorised ensemble: weights, grads, AdamW states, activations** | **~3.5–4.3 GB** | Main consumer. Each member is a different seed with block-bootstrap sampling. |
| Evaluation buffers | ~0.1–0.2 GB | |
| **Target peak `max_memory_reserved`** | **4.8–5.3 GB (80–88% of 6GB)** | "As close to the limit as safe" |

**Why an ensemble is the right use of the memory.**

- Deep ensembles are a simple, scalable and well-established way to improve predictive uncertainty [Source](https://proceedings.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles.pdf).
- Disagreement between members feeds directly into the confidence score (§9.6).
- Averaging member quantiles reduces seed variance, which is large in low-SNR financial models.

**Implementation.** Members are trained in one forward and backward pass using batched ("stacked") layers, where every weight tensor has a leading member dimension and the matmuls are done with `einsum`/`bmm`. PyTorch's `torch.func` `stack_module_state` + `vmap` ensembling pattern is a supported alternative [Source](https://docs.pytorch.org/tutorials/intermediate/ensembling.html).

**Automatic VRAM tuning (`vram_probe.py`)**

1. Fix the per-member batch size at B = 512. Generalisation matters more than throughput, so batch size is not increased just to use memory.
2. For K ∈ {8, 12, 16, 24, 32, 40, 48}, run 30 training steps plus one validation pass. Record `torch.cuda.max_memory_reserved()`.
3. Select the largest K whose peak is ≤ 5.2GB (≤ 4.9GB if the GPU drives a display).
4. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation.
5. Confirm stability with 3 full training runs at the chosen K and no OOM.

If K would exceed 48, stop increasing it. Gains in ensemble variance saturate, so leftover memory is used instead to run 2 Optuna trials concurrently during P3.

**Memory and speed techniques, and whether they apply**

| Technique | Use? | Reason |
|---|---|---|
| Mixed precision (bf16 autocast; fp16 + GradScaler on Turing) | **Yes** | Reduces activation memory and speeds up matmuls [Source](https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/). The pinball loss and quantile assembly stay in fp32. |
| `F.scaled_dot_product_attention` (fused/flash kernels) | **Yes** | Free memory and speed gain |
| `torch.compile` | Optional | Use if it compiles cleanly with the batched-ensemble layers; it can be skipped |
| Gradient checkpointing | **Fallback only** | Trades compute for memory [Source](https://machinelearningmastery.com/training-a-model-with-limited-memory-using-mixed-precision-and-gradient-checkpointing/). Only used if K < 12 fits, which is not expected. |
| 8-bit optimizers, LoRA, quantisation | **No** | Optimizer state is only MB; unnecessary complexity |
| Bigger d_model or more layers to fill VRAM | **No** | Overfitting risk with ~25k samples and low SNR |

---

## 7. Communications Management

| Artifact | Audience | Frequency |
|---|---|---|
| Weekly status (progress, risks, gate outlook) | Sponsor | Weekly |
| Experiment log (MLflow/CSV) | Team | Continuous |
| Phase-gate memo | Sponsor | At each gate |
| Final evaluation report + model card (intended use, limits, failure modes) | Sponsor, users | M5 |

---

## 8. Procurement and Stakeholder Management

**Procurement.** One data vendor provides NQ futures history from July 2007 to the present with volume. The contract must allow research use and must state the roll methodology, or supply raw contracts so the roll can be done in-house.

**Stakeholders.** The sponsor needs a usable risk and direction tool. The trader needs interpretable output, meaning quantile bands and confidence. The researcher needs a clean protocol. The plan manages expectations explicitly: the guaranteed value is calibrated risk bands, and any directional edge is a bonus that must prove itself out of sample.

---

## 9. Detailed Technical Design

### 9.1 Data

| Item | Specification |
|---|---|
| Instrument | NQ continuous futures, ratio back-adjusted. Returns at the roll are computed within a single contract. |
| Bar construction | Built from 1-min data into session-anchored 4H bars (A2). Partial bars around holidays are kept and flagged, not dropped. |
| History | Raw data from 2007-07 (feature warm-up). The modelling sample starts 2008-01. |
| **Volume** | 2008-01 → 2026-09 ≈ 18.7 years × ~1,500 bars/year ≈ **~28,000 bars**. Minimum acceptable is 10 years (~15k bars). Recommended is ≥ 15 years, so that the 2008 GFC, the 2020 COVID crash and the 2022 bear market are covered. These periods are needed for tail quantiles. |
| Cleaning | Remove bad ticks (\|1-min return\| > 15σ that revert within 3 minutes); align to the CME session calendar; forward-fill nothing. Missing bars stay missing, and windows that contain them are skipped. |

**Why not more data or a higher frequency?** Before about 2008, overnight NQ liquidity was much thinner, so those bars have different properties. Adding them increases non-stationarity more than it adds information. Using 1H data would quadruple the sample but change the task. That is out of scope.

### 9.2 Target

- **Raw target:** r₍t+1₎ = ln(C₍t+1₎/C₍t₎).
- **Training target, volatility-scaled:** z₍t+1₎ = r₍t+1₎ / σ̂₍t₎, where σ̂₍t₎ is a causal forecast of the next bar's volatility. σ̂₍t₎ is the EWMA of Garman–Klass variance (half-life 30 bars) multiplied by a bar-of-day seasonal factor, which is estimated on the training fold only.
- **Output:** the model predicts quantiles of z, and these are mapped back with q_τ(r) = σ̂₍t₎ · q_τ(z).

This explicit scaling handles the dominant heteroskedasticity. Intraday periodicity in volatility is pervasive in equity markets and strongly distorts dynamic volatility estimates if it is ignored [Source](https://www.sciencedirect.com/science/article/pii/S0927539897000042). Volatility scaling is also standard practice in time-series momentum research, where scaling materially changes results [Source](https://quantpedia.com/strategies/time-series-momentum-effect). The plan deliberately does **not** use RevIN-style per-window mean removal. RevIN is effective for distribution shift in general forecasting [Source](https://openreview.net/forum?id=cGDAkQo1C0p), but subtracting each window's mean return can erase the weak drift and momentum signal that matters here. Causal volatility scaling keeps the mean while removing the scale.

### 9.3 Feature Engineering (tight, evidence-based set)

**Design rule.** Every feature must be causal, stationary (or made stationary by scaling) and non-redundant, and must belong to a group with published or well-documented predictive relevance at intraday or short horizons. The MVP uses **11 continuous features + 2 calendar categoricals**. All continuous features are robust-scaled per fold (median/IQR, train only) and clipped to ±5.

#### 9.3.1 Feature Groups and Justification

| # | Feature | Definition (causal, at close of t) | Group |
|---|---|---|---|
| 1 | `r1_z` | r₍t₎ / σ̂₍t−1₎ | A. Return dynamics |
| 2 | `r6_z` | Σ last 6 bar returns / (σ̂√6) — 1 session | A |
| 3 | `r30_z` | Σ last 30 bar returns / (σ̂√30) — ~1 week | A |
| 4 | `gap_z` | ln(O₍t₎/C₍t−1₎) / σ̂ — non-zero mainly after maintenance or weekend breaks | A |
| 5 | `lgk_bar` | log Garman–Klass variance of bar t | B. Volatility state |
| 6 | `lrv_d` | log mean GK variance over 6 bars (daily) | B |
| 7 | `lrv_w` | log mean GK variance over 30 bars (weekly) | B |
| 8 | `lrv_m` | log mean GK variance over 120 bars (monthly) | B |
| 9 | `clv` | (2C − H − L)/(H − L), with 0 if H = L | C. Candle structure |
| 10 | `ema_dist` | (ln C − ln EMA₁₂₀(C)) / (σ̂_m · √120) | D. Trend / mean-reversion |
| 11 | `rvol` | log(V₍t₎ / median of V at the same bar slot over the previous 20 sessions) | E. Relative volume |
| 12 | `slot_next` | Bar-of-day slot of the **target** bar (0–5), embedded | F. Calendar (known future) |
| 13 | `dow_next` | Day of week of the target bar (0–4), embedded | F |

The per-bar return sequence is already present through feature 1 over the 180-bar lookback, so explicit lag features are unnecessary.

**A. Return dynamics (momentum and reversal at intraday scales).** Market intraday momentum is documented: the first half-hour return of the S&P 500 ETF predicts the last half-hour return with an R² of about 1.6% [Source](https://www.smallake.kr/wp-content/uploads/2015/01/SSRN-id2440866.pdf). Overnight returns have also been found to negatively predict the first half-hour return of US indices, which is a reversal effect across the session boundary [Source](https://www.sciencedirect.com/science/article/abs/pii/S1062940826001294). The 4H bars aligned to the session (A2) map onto exactly these regimes: overnight, pre-open, cash open, midday and close. Combined with the slot embedding, the return features let the model learn slot-conditional momentum and reversal. Scaling by volatility keeps them comparable across regimes. `gap_z` isolates break-time information.

**B. Volatility state (HAR on range-based variance).** This group carries most of the quantile skill. Range-based estimators use H and L, which have much lower variance than squared close-to-close returns. Garman–Klass has been found to be the range estimator closest to intraday realised volatility [Source](https://arxiv.org/html/2312.01426v2), and range estimators add information when forecasting realised volatility [Source](https://www.sciencedirect.com/science/article/pii/S1544612323003641). The daily/weekly/monthly HAR structure is a strong and durable out-of-sample benchmark for realised variance [Source](https://arxiv.org/html/2506.07928v1). The four log-variance features give the network that structure directly, and the gaps between them (term-structure slope) capture volatility mean-reversion.

**C. Candle structure.** The close-location value summarises where the bar closed within its range, which captures intrabar buying or selling pressure. It is orthogonal to the bar return in information content. One feature replaces a family of "shadow" or "wick" indicators.

**D. Trend / mean-reversion.** A single volatility-normalised distance from a ~1-month EMA captures medium-horizon trend state. Volatility-scaled time-series momentum is one of the most robust documented return anomalies [Source](https://alphaarchitect.com/time-series-momentum-volatility-scaling-and-crisis-alpha/). One feature is enough; RSI, MACD and stochastics are monotone transforms of the same price path.

**E. Relative volume.** Abnormal volume compared with the same intraday slot signals information arrival and moves with volatility. Normalising by slot removes the strong U-shaped intraday volume pattern [Source](https://www.nber.org/system/files/working_papers/w12413/w12413.pdf). This feature is only valid with exchange volume (A1).

**F. Calendar (known future covariates).** The volatility of the target bar depends heavily on which slot it is. The 10–14 ET bar contains the cash open, while the 22–02 bar is quiet. Recent work confirms economically significant time-of-day effects in intraday volatility [Source](https://www.oru.se/globalassets/oru-sv/institutioner/hh/workingpapers/workingpapers2025/wp-14-2025.pdf). Because the slot of the next bar is known in advance, it is fed to the head as a known-future covariate.

#### 9.3.2 Deliberately Excluded

- **Raw prices and levels** are non-stationary.
- **RSI, MACD, Stochastics, Bollinger, ADX and multiple moving averages** are redundant deterministic transforms of the price path already represented by groups A and D. They add collinearity and snooping risk.
- **Large indicator libraries** and **Fourier or wavelet stacks** are excluded for the same reasons.
- **Sentiment and news** add procurement and complexity cost.

Evidence that machine learning can extract intraday market predictability does exist, but it comes from large cross-sections of lagged high-frequency factor returns [Source](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4388560), not from technical indicators on a single series.

#### 9.3.3 Phase-2 Candidates (gated, not in MVP)

| Candidate | Evidence | Concern |
|---|---|---|
| VXN/VIX level and change, VIX term-structure slope | The VIX term structure predicts S&P 500 returns at horizons of 1 month to 1 year [Source](https://bauer.uh.edu/departments/finance/documents/seminars/Johnson_020712.pdf) | Not shown at a 4H horizon; adds a second data feed and session-alignment issues |
| Scheduled macro-event flag (CPI, FOMC, NFP in the target bar) | Known variance spikes | Needs a maintained calendar |

#### 9.3.4 Ablation Gate

Ablation is run group by group (remove one group at a time) on the **development folds only** (§9.7). A group is kept only if removing it worsens pooled pinball loss in at least 2 of 3 dev folds. A Phase-2 candidate is admitted only if it improves pinball loss in at least 2 of 3 dev folds and the DM test p-value is below 0.10. No ablation may touch the evaluation folds or the lockbox.

### 9.4 Model Architecture — "PatchTST-Q" (primary)

#### 9.4.1 Rationale

- **Patching.** PatchTST's patch-based tokens improve accuracy and efficiency, and the model remains a strong baseline across benchmarks [Source](https://arxiv.org/abs/2211.14730). Using a patch of 6 bars (one trading session) does three things: it denoises at the session level, it cuts the number of tokens by 6×, which makes attention memory trivial, and it lets attention operate over days, which is the natural scale of volatility persistence.
- **Channel mixing in the patch embedding.** The original PatchTST is channel-independent. Here the features are heterogeneous and informative jointly (for example, a return combined with the volatility state), so each patch's 6×F values are projected together. This is a small, documented deviation.
- **Why not TFT as primary?** TFT is strong and interpretable for multi-horizon quantile forecasting. In the cited S&P 500 VaR study it was also the slowest model to train, at about 180 minutes against 15–90 for the others [Source](https://www.mdpi.com/2227-7072/14/8/218). Its main advantage, the multi-horizon decoder, is unused for a one-step target, and its many components enlarge the tuning surface. It is kept only as an optional extra challenger if time allows.
- **Why not N-BEATS or N-HiTS?** These are excellent univariate and long-horizon architectures. They are built around decomposing the target series and are less natural for a one-step, covariate-driven quantile target.

#### 9.4.2 Configuration

| Component | Setting |
|---|---|
| Input | Lookback L = 180 bars (30 sessions), F = 11 continuous features |
| Patching | Patch length 6, stride 6 → 30 tokens; Linear(6·11 → d) + learnable positional embedding |
| Encoder | 3 pre-LayerNorm Transformer layers, d_model = 64, 4 heads, FFN 128, GELU, dropout 0.2, attention dropout 0.1 |
| Pooling | Concatenation of [last token, mean of tokens] → 128 |
| Known-future conditioning | Slot embedding (6→8) + day-of-week embedding (5→4), concatenated → 140 |
| Head | MLP 140 → 64 → 7 with a non-crossing parameterisation |
| Parameters | ≈ 110–130k per member |

#### 9.4.3 Non-Crossing Quantile Head

The head outputs a median m and six raw increments δ. The quantiles are then built monotonically:

- q₀.₅ = m
- q₀.₇₅ = m + softplus(δ₁), q₀.₉ = q₀.₇₅ + softplus(δ₂), q₀.₉₅ = q₀.₉ + softplus(δ₃)
- the lower quantiles are built symmetrically with their own δ₄–δ₆

This guarantees no quantile crossing by construction. Crossing is a known failure mode that dedicated monotone-network methods address [Source](https://eartharxiv.org/wg7sn). Asymmetry is preserved because the upper and lower increments are independent, which matters given the negative skew of equity returns.

### 9.5 Baselines (the DL model must earn its place)

| ID | Baseline | Purpose |
|---|---|---|
| B0 | Rolling unconditional empirical quantiles (last 1,500 bars) | Floor |
| B1 | Unconditional quantiles × seasonal factor × EWMA-GK volatility (i.e., quantiles of z, scaled by σ̂) | Isolates the value of volatility scaling |
| B2 | GJR-GARCH(1,1)-t on seasonally adjusted returns | Classic econometric benchmark |
| B3 | Linear quantile regression on the same 11 features (HAR-QR) | Strong linear benchmark |
| B4 | LightGBM quantile regression (one model per τ) on the same features plus 6 return lags | Strong tabular ML benchmark; also used in the cited VaR study [Source](https://www.mdpi.com/2227-7072/14/8/218) |

Challenger C1 is a 2-layer LSTM (hidden 64) with the same head and ensemble. It is included only to confirm that the choice of architecture is not driving the results.

### 9.6 Outputs: Signal and Confidence

1. **Quantiles.** Ensemble-mean quantiles of z (members averaged per τ), de-scaled by σ̂₍t₎ and then adjusted by CQR offsets (step 5).
2. **P(up).** Build a piecewise-linear CDF through the 7 calibrated quantiles, with tails extended linearly using the slope of the outer segment. Then p_up = 1 − F(0).
3. **Directional signal.**
   - The signal is +1 if p_up ≥ 0.5 + θ, −1 if p_up ≤ 0.5 − θ, and 0 otherwise.
   - It is also set to 0 if |q₀.₅| is smaller than the round-trip cost in return units.
   - θ ∈ {0.00, 0.02, 0.04, 0.06, 0.08} is selected on the calibration segment of each fold, never on the test segment.
4. **Confidence score.**
   - Raw score: s = |2·p_up − 1| × (1 − min(1, std_k(p_up,k) / 0.1)). The second factor uses the spread of p_up across ensemble members as a measure of disagreement.
   - s is mapped through an **isotonic regression** fitted on the calibration segment to P(sign correct).
   - The output confidence ∈ [0.5, 1.0] is therefore a calibrated probability that the signal's direction is right, which is directly interpretable.
5. **Conformal calibration.**
   - Conformalized quantile regression (CQR) offsets are computed per τ on the calibration segment of each fold. This corrects systematic under- or over-coverage [Source](https://medium.com/dataman-in-ai/conformalized-quantile-regression-for-time-series-probabilistic-forecasting-85a2a1047119).
   - In production, Adaptive Conformal Inference updates the miscoverage level online. ACI was designed for data whose distribution shifts over time in unknown ways [Source](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html).

### 9.7 Validation Strategy: Expanding Walk-Forward with Purging and Embargo

| Segment | Period | Use |
|---|---|---|
| **Development folds** (3) | Test years 2016, 2017, 2018 (train from 2008, expanding) | HPO, ablation, architecture sanity checks |
| **Evaluation folds** (7) | Test years 2019, 2020, 2021, 2022, 2023, 2024, 2025 | Reported performance; hyperparameters frozen at M3 |
| **Lockbox** | 2026-01-01 → latest available (~2026-09) | Run once at M5; never used for any decision before that |

**Inside each fold**, in time order:

- **Train:** from the start of the sample up to T_test − 12 months.
- **Embargo** of 30 bars.
- **Early-stopping validation:** 6 months.
- **Embargo** of 30 bars.
- **Calibration segment:** 6 months, used for CQR, isotonic confidence and θ.
- **Purge** of 1 bar.
- **Test:** 12 months.

**Purging and embargo rationale.** The label horizon is one bar, so only the last training sample's label overlaps the next segment; it is purged. Volatility persistence creates serial dependence well beyond the label horizon, so a 30-bar (about one week) embargo is added between segments. Purging and embargo are the standard remedies for leakage in financial cross-validation [Source](https://www.quantt.co.uk/resources/purged-cross-validation). An expanding walk-forward is preferred over combinatorial purged CV because it matches deployment exactly and keeps the analysis simple.

**Retraining.** The model is retrained for each fold, i.e. annually during evaluation. In production it is retrained quarterly, with ACI providing adjustment between retrains.

### 9.8 Training Configuration

| Setting | Value |
|---|---|
| Loss | Mean pinball loss over 7 τ on z (fp32) |
| Optimizer | AdamW, lr 1e-3, weight decay 0.05, β = (0.9, 0.95) |
| Schedule | 3% linear warm-up, then cosine decay |
| Per-member batch | 512 (fixed; not used as a memory lever) |
| Ensemble size K | Set automatically by `vram_probe.py` (expected 16–32) |
| Member diversity | Different seeds plus a stationary block-bootstrap of training windows (block = 30 bars) per member |
| Max epochs / early stopping | 60 / patience 8 on ensemble-mean validation pinball |
| Gradient clipping | 1.0 (global norm) |
| Precision | bf16 autocast (Ampere+) or fp16 + GradScaler (Turing) |
| Data path | Entire feature tensor on GPU; random window indices sampled on GPU |

**HPO.** Optuna TPE with a successive-halving pruner and **at most 40 trials**, run on the dev folds only. This mirrors the budget used in the rigorous LTSF re-benchmark [Source](https://arxiv.org/html/2502.14045v1). The search space is deliberately small:

| Parameter | Values |
|---|---|
| L | {90, 180, 360} |
| d_model | {32, 64, 96} |
| Layers | {2, 3} |
| Dropout | {0.1, 0.2, 0.3} |
| lr | [3e-4, 2e-3] (log) |
| Weight decay | [0.01, 0.1] (log) |

The number of trials is recorded for the Deflated Sharpe calculation.

### 9.9 Why This Model and Configuration Is Right for a 6GB GPU

1. **The memory profile is predictable and small per member.** With 30 tokens, attention activations are negligible. Activation memory scales linearly with K × B, so the probe can approach the memory limit smoothly and safely, with no cliff from sequence-length-squared attention.
2. **The capacity fits the data.** About 120k parameters per member is appropriate for about 25k noisy samples. Larger models are expected to fit noise.
3. **Unused memory becomes statistical value.** Ensemble members reduce variance and supply the disagreement signal for the confidence score, which a single larger model cannot provide.
4. **There are no I/O stalls.** Keeping the data on the GPU means utilisation is limited by compute rather than the CPU dataloader.
5. **Margins are safe.** An 80–88% peak target, expandable segments and a 3-run stability check make OOM unlikely. Gradient checkpointing remains available as a fallback.

---

## 10. Risk Management

### 10.1 Risk Register

| ID | Risk | Prob. | Impact | Response (strategy) | Trigger / owner |
|---|---|---|---|---|---|
| R1 | Look-ahead leakage (features, scalers, seasonal factors, calibration) | M | Critical | **Avoid:** causal pipeline, unit leakage tests, train-only fitting, peer review before M4 | Any test-period skill that looks too good (e.g., AUC > 0.58) → audit |
| R2 | Overfitting or data snooping via HPO and ablations | H | High | **Avoid:** HPO and ablation on dev folds only, ≤ 40 trials, frozen at M3, single-shot lockbox, Deflated Sharpe | Dev–eval gap > 3% pinball |
| R3 | No directional edge (low SNR at 4H) | H | Medium | **Accept / mitigate:** primary success is defined on distributional skill; the signal is gated by θ and cost; the project still delivers risk bands | S3 fails |
| R4 | Data quality: roll gaps, session misalignment, CFD/futures mismatch | M | High | **Mitigate:** DQ report, single-contract roll returns, same feed for train and deploy (A1) | DQ gate at P1 |
| R5 | Regime shift / non-stationarity (e.g., 2020, 2022) | H | Medium | **Mitigate:** volatility-scaled target, expanding window, quarterly retrain, ACI online recalibration, per-fold reporting | Rolling coverage leaves the ±3pp band |
| R6 | DL fails to beat B3/B4 | M | Medium | **Mitigate:** one pre-defined iteration (L=360, or adding the C1 LSTM to the ensemble). If it still fails, **ship the best baseline** with the same signal/confidence layer and record the outcome. | P4 gate |
| R7 | OOM / VRAM instability (driver, display, fragmentation) | L | Medium | **Mitigate:** auto-probe, 0.7–1.0GB headroom, expandable segments, checkpointing fallback | Any OOM |
| R8 | Tail miscalibration at 5%/95% (exceedances cluster in crises) | M | High | **Mitigate:** CQR + ACI, Christoffersen/DQ tests, include GFC/COVID in training | Kupiec or CC rejection |
| R9 | Scope creep (more features, exotic architectures) | M | Medium | **Avoid:** change control (§1.2), Phase-2 gate | Unplanned CRs |
| R10 | Non-reproducibility (GPU non-determinism) | M | Low | **Mitigate:** seeds, deterministic flags, report spread across 3 reruns | Spread > 0.3% |

### 10.2 Risk Monitoring

The register is reviewed weekly in the status report. Triggers are checked automatically in the evaluation scripts.

---

## 11. Monitoring and Controlling (production readiness)

| Monitor | Threshold | Action |
|---|---|---|
| Rolling 250-bar coverage per τ | Outside ±3pp | ACI absorbs it; if it persists for 500 bars, retrain early |
| Rolling pinball versus B1 | Skill < 0 for 500 bars | Review and retrain |
| Input drift | PSI > 0.25 on any feature | Investigate |
| Confidence reliability | Reviewed monthly | Refit the isotonic map at retrain |

---

## 12. Success Criteria (acceptance)

All criteria are measured on the **pooled evaluation folds (2019–2025)**, and the critical ones are then confirmed on the **lockbox**.

| ID | Criterion | Target | Minimum acceptable |
|---|---|---|---|
| **S1** Distributional skill | Pinball-loss reduction vs B0 / vs B2 / vs best of B3–B4 | ≥ 8% / ≥ 2% / ≥ 1% with DM p < 0.05 | ≥ 5% / ≥ 1% / not worse (DM p > 0.10), winning ≥ 4 of 7 folds |
| **S2** Calibration | \|coverage − τ\|: inner quantiles / 0.05 & 0.95. Kupiec and Christoffersen CC at 5%. | ≤ 1.5pp / ≤ 1.0pp; not rejected | ≤ 2.5pp / ≤ 1.5pp; Kupiec not rejected |
| **S2b** Integrity | Quantile crossings | 0 | 0 |
| **S3** Direction and confidence | AUC of p_up; top-20%-confidence hit rate; Spearman ρ(confidence decile, hit rate) | ≥ 0.52; ≥ 54%; ≥ 0.6 | ≥ 0.505; ≥ 52%; > 0 |
| **S4** Economic (informational) | Net-of-cost Sharpe of the signal strategy; DSR | > 0.5; DSR prob ≥ 0.90 | Reported only; required solely for any future live-trading decision |
| **S5** Engineering | Peak VRAM; OOM; reproducibility; full walk-forward runtime | 80–88% of 6GB; none in 3 runs; ±0.3%; < 2h | ≤ 90%; none; ±0.5%; < 6h |
| **S6** Lockbox | S1 vs B0 and S2 hold on 2026 data | Same as targets | Pinball skill vs B0 > 0 and coverage within ±3pp |

**Go/no-go at M5**

- **Full success:** S1, S2, S5 and S6 met, plus S3 at minimum. The full product (quantiles + signal + confidence) is approved for paper trading.
- **Partial success:** S1, S2, S5 and S6 met, but S3 not met. The model is approved as a **risk and position-sizing model**. The directional signal is disabled, and that outcome is documented. This is still a valuable result.
- **Fail:** S1 or S2 not met. Per R6, ship the best baseline with the calibration layer and record the lessons learned.

---

## 13. Project Closure

The closure deliverables are:

- The frozen code, config and data hashes.
- The final evaluation report: per-fold tables, reliability diagrams, coverage plots and DM tests.
- A model card covering intended use, known limits (low directional SNR, regime sensitivity, feed dependence), the retraining SOP and monitoring thresholds.
- A lessons-learned log, and archived experiment-tracking records.

The sponsor signs off against §12.