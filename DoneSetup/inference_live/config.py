"""Configuration for the live inference system.

Every tunable lives here. Nothing in this package reads global state, environment
variables or the old project's paths; the caller passes a `Config` explicitly.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Package root: <root>/inference_live/config.py  ->  <root>
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent

# Artifact drop locations (relative to the project dir that contains inference_live).
CKPT_DIR = PROJECT_DIR / "CKPTs"
SRC_DIR = PROJECT_DIR / "SRCs"


@dataclass(frozen=True)
class SlotTable:
    """The trained model's session-anchored slot geometry.

    A bar is 4 hours long, anchored to an America/New_York wall-clock start hour.
    `day_offset` is the calendar-day offset of the bar's local start date relative to
    the bar's CME `trade_date` (the business date).

    Verified against DATA/build_nq_4h.py: SLOT_START_H / SLOT_DAY_OFF.
    """
    timezone: str = "America/New_York"
    start_hours: Tuple[int, ...] = (18, 22, 2, 6, 10)
    day_offsets: Tuple[int, ...] = (-1, -1, 0, 0, 0)
    bar_hours: int = 4

    def __post_init__(self) -> None:
        if len(self.start_hours) != len(self.day_offsets):
            raise ValueError("start_hours and day_offsets must have the same length")
        if self.bar_hours <= 0:
            raise ValueError("bar_hours must be positive")

    @property
    def n_slots(self) -> int:
        return len(self.start_hours)

    def slot_for_hour(self, hour: int) -> int:
        """Map an ET wall-clock hour to a slot index, or -1 if the hour is not in the
        session (the 14:00-18:00 ET block that the training pipeline dropped).

        The session is a contiguous run of 4-hour blocks starting at `start_hours[0]`,
        so the slot is simply the block index measured from that start."""
        if not 0 <= hour < 24:
            raise ValueError(f"hour out of range: {hour}")
        offset_hours = (hour - self.start_hours[0]) % 24
        slot = offset_hours // self.bar_hours
        return slot if slot < self.n_slots else -1

    def validate(self) -> None:
        """Assert the table is a partition of the 24-hour clock into 5 x 4-hour blocks
        covering 18:00 -> 14:00 ET, with the 14:00-18:00 block absent."""
        covered = {h for h in range(24) if self.slot_for_hour(h) >= 0}
        expected = set(range(18, 24)) | set(range(0, 14))
        if covered != expected:
            raise ValueError(
                f"slot table covers {sorted(covered)}, expected {sorted(expected)}")
        for s, h in enumerate(self.start_hours):
            if self.slot_for_hour(h) != s:
                raise ValueError(f"slot {s} start hour {h} is inconsistent")
        for s in range(self.n_slots - 1):
            if (self.start_hours[s + 1] - self.start_hours[s]) % 24 != self.bar_hours:
                raise ValueError(
                    f"slots {s} and {s+1} are not contiguous {self.bar_hours}-hour blocks")


@dataclass(frozen=True)
class FeatureConfig:
    """Feature-engineering constants. Mirrors FE/nq4hq_features/config.py exactly."""
    ewma_half_life: int = 30
    ewma_min_periods: int = 30
    win_r_d: int = 5
    win_r_w: int = 30
    win_v_d: int = 5
    win_v_w: int = 30
    win_v_m: int = 120
    ema_span: int = 120
    rvol_sessions: int = 20
    gk_floor: float = 1e-10
    range_eps: float = 1e-12
    vol_floor: float = 1.0
    max_session_gap_days: int = 5
    var_shift: float = 9.0
    clip: float = 5.0
    # --- bar validity gates (mirrors DATA/build_nq_4h.py) -------------------------
    expected_minutes: int = 240
    min_coverage: float = 0.99
    max_missing_run: int = 1

    # --- LIVE gate policy ----------------------------------------------------------
    # The training gates above were calibrated on the vendor's minute feed: a bar is
    # accepted only if 238 of its 240 minutes are genuine and no two minutes are
    # missing in a row. A broker CFD feed is quoted less densely (the session open is
    # not quoted for its first few minutes, and the daily break sits at a different
    # place), so the training gates reject nearly-complete broker sessions.
    #
    # `use_live_gates=True` selects the relaxed policy below. It is OFF in the bare
    # FeatureConfig() so that the offline parity tests keep reproducing the training
    # pipeline exactly; the live Config turns it on.
    use_live_gates: bool = False
    live_min_coverage: float = 0.90          # slots 1..4: >= 216 of 240 genuine minutes
    live_max_missing_run: int = 60           # slots 1..4: longest tolerated hole
    live_slot0_min_coverage: float = 0.0     # slot 0: the coverage gate is disabled
    live_slot0_max_missing_run: int = 240    # slot 0: the missing-run gate is disabled
    live_slot0_min_observed_minutes: int = 2  # slot 0: accept from 2 genuine minutes on

    @classmethod
    def live(cls) -> "FeatureConfig":
        """The LIVE gate policy (relaxed for broker feeds). See the fields above."""
        return dataclasses.replace(cls(), use_live_gates=True)


@dataclass(frozen=True)
class ModelConfig:
    """The trained model's contract. Values here are the FROZEN M3 values."""
    lookback: int = 300
    K: int = 16
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.3
    attn_dropout: float = 0.1
    seed: int = 1729
    n_features: int = 11
    taus: Tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
    # The saved member predictions were produced under bf16 autocast on CUDA.
    # Reproducing fp32 on the same weights differs by ~2e-5 per member.
    use_bf16_autocast: bool = True
    device: str = "auto"          # "auto" | "cuda" | "cpu"


@dataclass(frozen=True)
class MT5Config:
    """MetaTrader 5 connection settings.

    `server_utc_offset_hours` is the broker's server offset from UTC. MT5 timestamps
    are BROKER SERVER TIME, not UTC: the API's clock runs `server_utc_offset_hours`
    AHEAD of true UTC (measured on this account: requesting a range that extends 6 h
    past the true present returns bars labelled up to true_now + 3 h). The data
    source converts in both directions -- server epoch -> true UTC on the way in, and
    true UTC -> server epoch on the way out (the range arguments are on the same
    server clock as the bar timestamps).
    """
    path: str | None = r"C:\Program Files\MetaTrader 5\terminal64.exe"
    symbol: str = "NDXUSD"
    timeframe_minutes: int = 1
    bootstrap_minutes: int = 90000
    initialize_timeout_ms: int = 60_000
    prefer_mid: bool = True
    volume_mode: str = "sum"
    server_utc_offset_hours: int = 3

@dataclass(frozen=True)
class CalibrationConfig:
    """How the trained calibration is applied live."""
    # "static":  frozen bias + frozen CQR offsets (exact reproduction of the offline path)
    # "aci":     additionally advance the ACI alpha vector with realized outcomes
    mode: str = "static"
    gamma: float = 0.005
    # Persisted ACI end-of-fold alpha vector (3 values) used as the ACI starting point.
    aci_alpha: Tuple[float, ...] = (0.10, 0.20, 0.50)


@dataclass(frozen=True)
class Config:
    """Top-level configuration for one live inference run."""
    symbol: str = "NDXUSD"
    slots: SlotTable = field(default_factory=SlotTable)
    features: FeatureConfig = field(default_factory=FeatureConfig.live)
    model: ModelConfig = field(default_factory=ModelConfig)
    mt5: MT5Config = field(default_factory=MT5Config)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)

    # Where the frozen artifacts live.
    checkpoint_dir: Path = CKPT_DIR
    # Which trained fold to ship. 2026 = the lockbox fold, trained on all data
    # through 2025.
    fold: str = "2026"
    architectures: Tuple[str, ...] = ("patchtst", "lstm")

    # History seed: a parquet with the tail of the trained 4h bar history plus the
    # matching feature rows, used to warm the recursive feature state. Optional.
    seed_bars_path: Path | None = None
    seed_features_path: Path | None = None

    # --- seed / live seam (Stage D) -------------------------------------------
    # The seed's volume is the vendor's contract-volume share; the live feed's volume
    # is the broker's tick count. The two differ by ~3 orders of magnitude, so the
    # `rvol` baseline (the median of the previous 20 same-slot bars) is poisoned for
    # as long as the median is still vendor-derived, and `rvol` pins at the +/-5 clip.
    # When this is on, the live bars' volume is rescaled into the seed's units using
    # a per-slot factor estimated from the bars the two frames share.
    align_volume_scale: bool = True
    volume_align_min_overlap: int = 5   # overlap bars required for a per-slot factor

    # --- warm-up ----------------------------------------------------------------
    # The model needs L bars in its window AND every row of that window to have all
    # 11 features finite. The long trailing features (EMA-120 on the price index,
    # `lrv_m`'s 120-bar trailing mean) are not finite until row 119, so the effective
    # minimum history is L + 119 = 419 bars, not L = 300.
    warmup_offset: int = 119

    # Where to write predictions.
    output_dir: Path = PROJECT_DIR / "out"

    def describe(self) -> str:
        c = self
        lines = [
            f"symbol              : {c.symbol}",
            f"timezone            : {c.slots.timezone}",
            f"slot start hours ET : {list(c.slots.start_hours)}",
            f"slot day offsets    : {list(c.slots.day_offsets)}",
            f"lookback L          : {c.model.lookback}",
            f"warm-up offset      : {c.warmup_offset}  (effective minimum "
            f"{c.model.lookback + c.warmup_offset} bars)",
            f"ensemble K          : {c.model.K}",
            f"taus                : {list(c.model.taus)}",
            f"fold                : {c.fold}",
            f"architectures       : {list(c.architectures)}",
            f"checkpoint dir      : {c.checkpoint_dir}",
            f"seed bars           : {c.seed_bars_path}",
            f"volume alignment    : {c.align_volume_scale}",
            f"mt5 server offset   : UTC+{c.mt5.server_utc_offset_hours} "
            f"(MT5 timestamps are broker server time)",
            f"mt5 symbol          : {c.mt5.symbol}",
            f"mt5 path            : {c.mt5.path or '(auto-discover)'}",
            f"bar gates           : "
            f"{'LIVE (relaxed)' if c.features.use_live_gates else 'TRAINING (strict)'}",
            f"calibration mode    : {c.calibration.mode}",
            f"bf16 autocast       : {c.model.use_bf16_autocast}",
            f"device              : {c.model.device}",
        ]
        return "\n".join(lines)

    def replace(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)
