"""Quality control gates, effective state masking, and dynamical regime tagging."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ecoverse.core.constants import (
    ABIOTIC_INDICES,
    BIOTIC_INDICES,
    DT_DAYS,
    N_STATES,
    PRODUCER_INDICES,
    STATE_NAMES,
)
from ecoverse.core.spec import ForcingSpec, SystemSpec
from ecoverse.sim.solver import SimulationResult


@dataclass
class QualityReport:
    """Quality gate verdict, effective state survival mask, and dynamical regime tag."""

    passed: bool
    fail_reason: str
    effective_state_mask: np.ndarray  # bool (8,)
    regime_tag: str
    n_alive_producers: int
    n_varying_states: int


def classify_regime(
    x_true: np.ndarray,
    effective_state_mask: np.ndarray,
    forcing: ForcingSpec,
) -> str:
    """Classify trajectory into forced_periodic, endogenous_oscillation, or regime_shift."""
    # Focus on effective biotic states (fallback to all effective states)
    bio_idx = [i for i in BIOTIC_INDICES if effective_state_mask[i]]
    if not bio_idx:
        bio_idx = [i for i in range(N_STATES) if effective_state_mask[i]]
    if not bio_idx:
        return "forced_periodic"

    # 1. Check for persistent regime shift between early and late equal-season windows
    win = 180  # 30 days at 4h resolution
    for i in bio_idx:
        series = x_true[:, i].astype(np.float64)
        mean_all = float(np.mean(series))
        if mean_all < 1e-4:
            continue
        early = series[:win]
        late = series[-win:]
        m_early = float(np.mean(early))
        m_late = float(np.mean(late))
        within_std = 0.5 * (float(np.std(early)) + float(np.std(late))) + 1e-6
        rel_shift = abs(m_late - m_early) / (max(m_early, m_late, mean_all) + 1e-6)
        if rel_shift > 0.35 and abs(m_late - m_early) > 1.5 * within_std:
            return "regime_shift"

    # 2. Spectral separation between external forcing bands and endogenous oscillation bands
    if not forcing.seasonal and not forcing.diel:
        return "endogenous_oscillation"

    n_t = x_true.shape[0]
    freqs = np.fft.rfftfreq(n_t, d=DT_DAYS)  # cycles per day

    # Forced bands: annual + harmonic (<= 0.008 d^-1) and diel (~1.0, 2.0 d^-1)
    forced_mask = (freqs > 0.0) & (
        (freqs <= 0.008)
        | (np.abs(freqs - 1.0) <= 0.08)
        | (np.abs(freqs - 2.0) <= 0.08)
    )
    endo_mask = (freqs > 0.008) & (np.abs(freqs - 1.0) > 0.08) & (np.abs(freqs - 2.0) > 0.08) & (freqs <= 0.90)

    endo_powers = []
    forced_powers = []
    for i in bio_idx:
        s = x_true[:, i].astype(np.float64)
        s = s - np.mean(s)
        var = float(np.var(s))
        if var < 1e-10:
            continue
        fft_pow = np.abs(np.fft.rfft(s)) ** 2
        endo_powers.append(float(np.sum(fft_pow[endo_mask])))
        forced_powers.append(float(np.sum(fft_pow[forced_mask])))

    if not endo_powers:
        return "forced_periodic"

    tot_endo = float(np.sum(endo_powers))
    tot_forced = float(np.sum(forced_powers))
    if tot_endo / (tot_endo + tot_forced + 1e-12) > 0.35:
        return "endogenous_oscillation"
    return "forced_periodic"


def evaluate_quality(spec: SystemSpec, result: SimulationResult) -> QualityReport:
    """Apply the 4 hard quality gates and compute effective_state_mask."""
    empty_mask = np.zeros(N_STATES, dtype=bool)

    # Gate 1: Solver convergence & finiteness
    if not result.ok or result.x_burnin_last30 is None or result.x_true is None:
        return QualityReport(
            passed=False,
            fail_reason=result.fail_reason or "solver_failed",
            effective_state_mask=empty_mask,
            regime_tag="invalid",
            n_alive_producers=0,
            n_varying_states=0,
        )

    # Gate 2: Boundedness & physical positivity
    if result.raw_min < -1e-4 or result.raw_max > 500.0:
        return QualityReport(
            passed=False,
            fail_reason="bounds_violated",
            effective_state_mask=empty_mask,
            regime_tag="invalid",
            n_alive_producers=0,
            n_varying_states=0,
        )

    # Gate 3: Survival & competitive exclusion masking (Section 5.1.3)
    burn_means = np.mean(result.x_burnin_last30, axis=0)
    eff_mask = np.zeros(N_STATES, dtype=bool)

    for i in ABIOTIC_INDICES:
        if spec.active_states[STATE_NAMES[i]]:
            eff_mask[i] = True

    n_alive_producers = 0
    for i in BIOTIC_INDICES:
        if spec.active_states[STATE_NAMES[i]] and burn_means[i] >= 1e-3:
            eff_mask[i] = True
            if i in PRODUCER_INDICES:
                n_alive_producers += 1

    if n_alive_producers == 0:
        return QualityReport(
            passed=False,
            fail_reason="all_producers_extinct",
            effective_state_mask=eff_mask,
            regime_tag="invalid",
            n_alive_producers=0,
            n_varying_states=0,
        )

    # Gate 4: Non-trivial dynamics (CV >= 0.02 on at least 2 effective states)
    n_varying = 0
    x_formal = result.x_true.astype(np.float64)
    for i in range(N_STATES):
        if not eff_mask[i]:
            continue
        col = x_formal[:, i]
        m = float(np.mean(col))
        if m > 1e-9:
            cv = float(np.std(col)) / m
            if cv >= 0.02:
                n_varying += 1

    if n_varying < 2:
        return QualityReport(
            passed=False,
            fail_reason="low_cv_deadwater",
            effective_state_mask=eff_mask,
            regime_tag="invalid",
            n_alive_producers=n_alive_producers,
            n_varying_states=n_varying,
        )

    regime_tag = classify_regime(result.x_true, eff_mask, spec.forcing)
    return QualityReport(
        passed=True,
        fail_reason="",
        effective_state_mask=eff_mask,
        regime_tag=regime_tag,
        n_alive_producers=n_alive_producers,
        n_varying_states=n_varying,
    )
