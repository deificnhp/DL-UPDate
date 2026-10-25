# DL-Layer — live inference for NQ4H-Q

This directory contains the **live inference system** for the NQ4H-Q model, and the
frozen artifacts it runs on. It has no dependency on the old training project.

```
DL-Layer/
├── CKPTs/                <- the frozen artifacts (copied from the trained project)
│   ├── checkpoints/      <- 2026_patchtst.pt, 2026_lstm.pt (weights + frozen transform)
│   ├── provenance/       <- 2026_patchtst.json, 2026_lstm.json (calibration)
│   └── history_seed.parquet  <- 419 warm-up bars (the default seed; required to
│                                 predict on the first run)
├── SRCs/                 <- scratch / notes
└── inference_live/       <- the live system
    ├── README.md         <- the operator guide: setup, running, verification, risks
    └── DESIGN_NOTE.md    <- what was read out of the trained project, and why
```

## Start here

```bash
pip install numpy pandas pyarrow torch MetaTrader5

python -m inference_live.cli check          # environment + artifacts
python -m inference_live.cli predict --symbol NDXUSD
python -m inference_live.cli selftest --training-root "D:\DataScience\interm\RED SPRITE\DL-Layer"
```

**Full documentation: [`inference_live/README.md`](inference_live/README.md).**
It covers what must be copied from the old project, how to run the live inference,
the output schema, **the clock contract (MT5 timestamps are broker server time,
UTC+3)**, **the relaxed live coverage gates**, the warm-up contract, the
verification results, and the risks. The live-run specifics of this deployment are
in [`LIVE_FIX_REPORT.md`](LIVE_FIX_REPORT.md).

## What this does

Takes a live 1-minute feed, rebuilds the **exact custom 4-hour session bars** the
model was trained on (5 ET-anchored slots per session, not UTC 4-hour buckets),
computes the **exact features**, runs the **trained ensemble**, and emits **only
calibrated quantiles** for the next 4-hour bar's log return.

There is no signal generation, no sizing, and no order execution here. It produces a
distribution; what you do with it is out of scope.

## Fidelity

Every layer has been checked against the training project's own artifacts:

| Layer | Result |
|---|---|
| Bars (on the real 5.2 M-minute vendor feed) | 13,586 accepted bars + 6,104 exclusions, identical prices and identical rejection reasons |
| Features (all 11 + 2 covariates) | max abs difference **0.0** |
| Model (saved member predictions, bf16/CUDA) | max abs difference **0.0** (bit-exact) |
| Warm-up (a 419-bar seed vs the full history) | agrees to **2e-6** |

Reproduce with `python -m inference_live.cli selftest --training-root <old project>`.
