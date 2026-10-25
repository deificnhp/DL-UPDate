"""NQ4H-Q causal feature & target pipeline (PLAN.md §9.2-9.3, 5-bar session adaptation)."""
from .config import CFG, FEATURES_CONT, FEATURES_CAL, FEATURES_ALL, TARGET_RAW, TARGET_SCALED
from .pipeline import build, compute_all
from .seasonality import fit_slot_seasonality, apply_slot_seasonality
