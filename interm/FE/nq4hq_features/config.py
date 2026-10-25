"""All constants in one place. Changing anything here is a CR under PLAN.md §1.2."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- Adaptation (decided): slot 5 (terminal 3h bar) is absent -> 5 bars/session.
    session_bars: int = 5            # slots 0..4
    # --- PLAN.md §9.2 target scaling
    ewma_half_life: float = 30.0     # bars; EWMA of Garman-Klass variance
    ewma_min_periods: int = 30
    # --- PLAN.md §9.3.1 windows (6 -> 5 where the window means "one session")
    win_r_d: int = 5                 # r5_z  (was r6_z)
    win_r_w: int = 30                # r30_z (~1.2 weeks with 5-bar sessions; kept per PLAN)
    win_v_d: int = 5                 # lrv_d (was 6)
    win_v_w: int = 30                # lrv_w
    win_v_m: int = 120               # lrv_m
    ema_span: int = 120              # ema_dist
    rvol_sessions: int = 20          # rvol same-slot median lookback
    # --- Numerical guards
    gk_floor: float = 1e-10          # GK = 0 only when H == L; floor before log
    range_eps: float = 1e-12         # H-L below this counts as zero range (clv = 0)
    vol_floor: float = 1.0           # volume floor before log ratio
    # --- Contiguity: max calendar days between consecutive trade_dates
    max_session_gap_days: int = 5    # Fri->Mon = 3, Fri->Tue (Mon holiday) = 4


CFG = Config()

FEATURES_CONT = [
    "r1_z", "r5_z", "r30_z", "gap_z",
    "lgk_bar", "lrv_d", "lrv_w", "lrv_m",
    "clv", "ema_dist", "rvol",
]
FEATURES_CAL = ["slot_next", "dow_next"]
FEATURES_ALL = FEATURES_CONT + FEATURES_CAL
TARGET_RAW = "r_next"
TARGET_SCALED = "z_next_ewma"     # r_next / sigma_ewma (seasonal factor applied per fold)

REQUIRED_COLUMNS = [
    "interval_id", "bar_start_utc", "bar_end_utc", "trade_date", "calendar_date", "slot",
    "contract_id", "open_mid", "high_mid", "low_mid", "close_mid",
    "spread_open", "spread_mean", "spread_max", "spread_close", "volume",
    "expected_minutes", "observed_minutes", "quote_coverage", "active_seconds", "wall_seconds",
    "is_terminal_3h", "is_partial_calendar", "is_early_close", "is_holiday_session",
    "is_roll_session", "is_post_maintenance", "is_post_weekend", "is_post_holiday",
]
