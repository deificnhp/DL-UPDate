"""Command line interface for the live inference system.

    python -m inference_live.cli --help

Sub-commands:
    check     verify the artifacts and the environment, then exit
    predict   produce one prediction from the live feed (or a replay file)
    replay    produce one prediction from a recorded minute file (deterministic)
    selftest  run the offline parity checks against the training project
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from .config import CKPT_DIR, Config, ModelConfig, MT5Config
from .engine import LiveEngine, QuantilePrediction
from .logging_setup import get_logger, setup_logging

log = get_logger("cli")

# The shipped history seed: 419 bars of the trained 4-hour history, which is the
# minimum the model needs for a finite L=300 window.
DEFAULT_SEED = CKPT_DIR / "history_seed.parquet"


def default_seed_path() -> Path | None:
    """The seed artifact to use when none is given on the command line."""
    return DEFAULT_SEED if DEFAULT_SEED.exists() else None


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--symbol", default="NDXUSD",
                   help="MT5 symbol name (default: %(default)s)")
    p.add_argument("--ckpt-dir", type=Path, default=CKPT_DIR,
                   help="directory holding checkpoints/ and provenance/ (default: %(default)s)")
    p.add_argument("--fold", default="2026",
                   help="which trained fold to use (default: %(default)s)")
    p.add_argument("--arch", default="patchtst", choices=["patchtst", "lstm"],
                   help="which architecture to run (default: %(default)s)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="torch device (default: %(default)s)")
    p.add_argument("--no-bf16", action="store_true",
                   help="disable bf16 autocast (results differ from the shipped "
                        "predictions by ~2e-5 per member)")
    p.add_argument("--seed-bars", type=Path, default=default_seed_path(),
                   help="parquet of historical 4h bars used to warm the feature state "
                        "(default: %(default)s)")
    p.add_argument("--server-utc-offset", type=int, default=3,
                   help="the broker's server offset from UTC. MT5 timestamps are "
                        "broker SERVER time, not UTC (default: %(default)s)")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--log-file", default=None)


def _build_config(args) -> Config:
    mcfg = ModelConfig(device=args.device, use_bf16_autocast=not args.no_bf16)
    mt5cfg = MT5Config(symbol=args.symbol,
                       server_utc_offset_hours=args.server_utc_offset)
    return Config(symbol=args.symbol, model=mcfg, mt5=mt5cfg,
                   checkpoint_dir=Path(args.ckpt_dir), fold=args.fold,
                   architectures=(args.arch,), seed_bars_path=args.seed_bars)


def _print_prediction(pred: QuantilePrediction, as_json: bool) -> None:
    if as_json:
        print(json.dumps(pred.to_dict(), indent=2))
        return
    d = pred.to_dict()
    print("=" * 72)
    print(f"  {pred.model}  fold {pred.fold}   calibrated={pred.calibrated}")
    print("=" * 72)
    print(f"  input bar   : {d['input_bar']['bar_start_utc']}  ->  "
          f"{d['input_bar']['bar_end_utc']}")
    print(f"               trade_date {d['input_bar']['trade_date']}  "
          f"slot {d['input_bar']['slot']}  close {d['input_bar']['close']:.2f}")
    tgt = d["target"]
    if tgt["bar_start_utc"]:
        print(f"  target bar  : {tgt['bar_start_utc']}  ->  {tgt['bar_end_utc']}")
        print(f"               trade_date {tgt['trade_date']}  slot {tgt['slot']}  "
              f"horizon {tgt['horizon_hours']:.0f}h")
    else:
        print(f"  target bar  : (not yet in history)  horizon {tgt['horizon_hours']:.0f}h")
    print("-" * 72)
    print("  quantiles of the next 4h log return (raw units):")
    for k, v in d["quantiles"].items():
        print(f"    {k}  {v:+.6f}   ({v * 100:+.3f}%)")
    print("-" * 72)
    print(f"  scale {pred.scale:.6e}   sigma_ewma {pred.sigma_ewma:.6e}   "
          f"slot_next {pred.slot_next}  dow_next {pred.dow_next}")
    print()


# ---------------------------------------------------------------------------
def cmd_check(args) -> int:
    cfg = _build_config(args)
    print(cfg.describe())
    print()
    ok = True
    try:
        eng = LiveEngine(cfg)
        eng.load_models()
        print("artifacts: OK")
    except Exception as exc:  # noqa: BLE001
        print(f"artifacts: FAIL -- {type(exc).__name__}: {exc}")
        ok = False
    try:
        import torch
        print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
    except Exception as exc:  # noqa: BLE001
        print(f"torch: FAIL -- {exc}")
        ok = False
    try:
        import MetaTrader5  # noqa: F401
        print("MetaTrader5 package: installed")
    except ImportError:
        print("MetaTrader5 package: NOT installed (needed for the live feed)")
    return 0 if ok else 1


def cmd_predict(args) -> int:
    from .data import mt5_source
    cfg = _build_config(args)
    eng = LiveEngine(cfg)
    eng.load_models()

    # --- seed ----------------------------------------------------------------
    seed = args.seed_bars or cfg.seed_bars_path
    if seed is not None and Path(seed).exists():
        bars = pd.read_parquet(seed)
        eng.seed_history_from_bars(bars)
        log.info("seeded with %d bars from %s (the effective minimum is %d bars)",
                 len(bars), seed, eng.effective_min_history())
    else:
        log.warning("no history seed found (%s). The engine needs at least %d bars "
                    "before it can predict; without a seed it will decline and say so. "
                    "Create one with: python -m inference_live.cli make-seed",
                    seed, eng.effective_min_history())

    mt5_source.initialize(cfg.mt5)
    try:
        minutes = mt5_source.fetch_minutes(cfg.mt5)
        mt5_source.clock_diagnostic(minutes, cfg.mt5, cfg.slots)
    finally:
        mt5_source.shutdown()
    eng.update_from_minutes(minutes, contract_id=cfg.symbol)

    now = pd.Timestamp.now(tz="UTC")
    pred = eng.predict(args.arch, now_utc=now)
    if pred is None:
        print("NO PREDICTION: the engine declined to emit one (see the log for the reason).")
        return 2
    _print_prediction(pred, args.json)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(pred.to_dict(), indent=2), encoding="utf-8")
        print(f"written: {args.out}")
    return 0


def cmd_make_seed(args) -> int:
    """Build a history seed from a clean 4-hour bar file.

    The seed is the tail of the training project's clean bar history: the last N
    bars (N >= 419) in the CLEAN_COLUMNS schema. It exists to warm the recursive
    feature state (EWMA, EMA-120 on the price index, the 20-same-slot volume
    median) so that the model's 300-bar window contains no non-finite feature.
    """
    import numpy as np

    from .core.bar_builder import CLEAN_COLUMNS

    src = Path(args.clean_bars)
    if not src.exists():
        print(f"clean bar file not found: {src}")
        print("The seed is the tail of the training project's clean 4h bars. Either")
        print("  (a) keep the shipped artifact CKPTs/history_seed.parquet, or")
        print("  (b) rebuild it with:")
        print("      python -m inference_live.cli make-seed "
              "--clean-bars <path to nq_4h_clean.parquet>")
        return 2
    bars = pd.read_parquet(src)
    for c in ("bar_start_utc", "bar_end_utc"):
        bars[c] = pd.to_datetime(bars[c], utc=True)
    bars["trade_date"] = pd.to_datetime(bars["trade_date"]).dt.date
    bars = bars.sort_values("bar_start_utc", ignore_index=True)
    n = min(args.bars, len(bars))
    seed = bars.tail(n)
    missing = [c for c in CLEAN_COLUMNS if c not in seed.columns]
    if missing:
        print(f"the clean bar file is missing the seed columns: {missing}")
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seed[CLEAN_COLUMNS].to_parquet(out, index=False)
    print(f"wrote {out}: {len(seed)} bars "
          f"({seed['bar_start_utc'].iloc[0]} .. {seed['bar_start_utc'].iloc[-1]})")
    if len(seed) < 419:
        print(f"WARNING: {len(seed)} bars is below the model's effective minimum of "
              f"419; the engine will still decline to predict.")
    return 0


def cmd_replay(args) -> int:
    from .data import replay_source
    if not args.minutes and not args.bars:
        print("replay needs either --minutes (a recorded 1-minute feed) or --bars "
              "(4-hour bars to synthesize a minute feed from)")
        return 2
    cfg = _build_config(args)
    eng = LiveEngine(cfg)
    eng.load_models()

    minutes = (replay_source.read_minutes(args.minutes) if args.minutes
               else replay_source.replay_from_bar_file(args.bars))
    eng.update_from_minutes(minutes, contract_id=cfg.symbol)

    # --now simulates the wall clock, so the "a bar has just closed" gate can be
    # exercised on a recorded feed
    now = pd.Timestamp(args.now) if args.now else None
    if now is not None and now.tzinfo is None:
        now = now.tz_localize("UTC")
    pred = eng.predict(args.arch, now_utc=now)
    if pred is None:
        print("NO PREDICTION: the engine declined to emit one (see the log for the reason).")
        return 2
    _print_prediction(pred, args.json)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(pred.to_dict(), indent=2), encoding="utf-8")
        print(f"written: {args.out}")
    return 0


def cmd_selftest(args) -> int:
    from .tests.run_selftest import run_selftest
    return run_selftest(args.training_root, args.keep_going)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m inference_live.cli",
        description="Live inference for the NQ4H-Q 4-hour quantile model.")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="verify artifacts and environment")
    _add_common(c)
    c.set_defaults(func=cmd_check)

    c = sub.add_parser("predict", help="one prediction from the live MT5 feed")
    _add_common(c)
    c.add_argument("--out", type=Path, default=None, help="write the prediction as JSON")
    c.add_argument("--json", action="store_true", help="print the prediction as JSON")
    c.set_defaults(func=cmd_predict)

    c = sub.add_parser("replay", help="one prediction from a recorded minute feed")
    _add_common(c)
    c.add_argument("--minutes", type=Path, default=None, help="parquet/csv of 1-minute bars")
    c.add_argument("--bars", type=Path, default=None,
                   help="parquet of 4h bars to synthesize a minute feed from")
    c.add_argument("--out", type=Path, default=None, help="write the prediction as JSON")
    c.add_argument("--json", action="store_true", help="print the prediction as JSON")
    c.add_argument("--now", default=None,
                   help="simulate the wall clock (ISO timestamp); enables the "
                        "'a bar has just closed' gate on a recorded feed")
    c.set_defaults(func=cmd_replay)

    c = sub.add_parser("selftest", help="offline parity checks against the training project")
    c.add_argument("--training-root", type=Path, required=True,
                   help="path to the trained DL-Layer project root")
    c.add_argument("--keep-going", action="store_true",
                   help="run every check even if one fails")
    c.add_argument("--log-level", default="INFO")
    c.set_defaults(func=cmd_selftest)

    c = sub.add_parser("make-seed", help="build a history seed from a clean 4h bar file")
    c.add_argument("--clean-bars", type=Path, required=True,
                   help="the training project's nq_4h_clean.parquet")
    c.add_argument("--bars", type=int, default=419,
                   help="how many bars to keep (default: %(default)s -- the model's "
                        "effective minimum)")
    c.add_argument("--out", type=Path, default=DEFAULT_SEED,
                   help="where to write the seed (default: %(default)s)")
    c.add_argument("--log-level", default="INFO")
    c.set_defaults(func=cmd_make_seed)

    args = p.parse_args(argv)
    setup_logging(getattr(args, "log_level", "INFO"), getattr(args, "log_file", None))
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001
        log.error("%s: %s", type(exc).__name__, exc)
        raise


if __name__ == "__main__":
    sys.exit(main())
