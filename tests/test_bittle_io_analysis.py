"""Offline tests for the system-ID analysis in `step` and `bodyid`.

Each test synthesizes data from a known ground truth and asserts the estimator recovers
it — the same shapes the hardware will produce, minus the hardware.
"""

import math

import numpy as np
import pytest

from bittle_io.experiments.bodyid import knee_frequency, lockin
from bittle_io.experiments.friction import _analyse
from bittle_io.experiments.step import fold_and_fit


# -- equivalent-time step folding -------------------------------------------

def _synth_step(td, tau, amp, n=300, noise=0.15, seed=0, slew=None):
    """Sparse, jittered samples of a step response -- what folding N trials yields."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 0.7, n)
    if slew is None:
        y = np.where(t > td, amp * (1.0 - np.exp(-(t - td) / tau)), 0.0)
    else:
        y = np.clip(slew * np.maximum(t - td, 0.0), 0.0, amp)
    y = y + rng.normal(0.0, noise, n)
    return list(zip(t, y))


def test_fold_recovers_tau_and_dead_time():
    fit = fold_and_fit(_synth_step(td=0.03, tau=0.06, amp=8.0), amp=8.0)
    assert abs(fit["dead_time_ms"] - 30) <= 10
    assert 45 <= fit["tau_ms"] <= 75
    assert fit["r2"] > 0.95
    # f_3db consistent with tau
    assert fit["f3db_hz"] == pytest.approx(1.0 / (2 * math.pi * fit["tau_ms"] / 1000), rel=0.01)


def test_fold_recovers_slew_from_a_ramp():
    """Large-amplitude regime: the response is a velocity-saturated ramp."""
    fit = fold_and_fit(_synth_step(td=0.03, tau=0.0, amp=30.0, slew=120.0), amp=30.0)
    assert 95 <= fit["slew_deg_s"] <= 145


def test_fold_survives_outliers():
    """A serial link produces occasional garbage readings; medians must absorb them."""
    pts = _synth_step(td=0.03, tau=0.06, amp=8.0)
    pts += [(0.2, 90.0), (0.4, -60.0), (0.1, 90.0)]     # wild outliers
    fit = fold_and_fit(pts, amp=8.0)
    assert 40 <= fit["tau_ms"] <= 85


def test_fold_too_few_points_returns_empty():
    assert fold_and_fit([(0.1, 1.0)] * 5, amp=8.0) == {}


# -- lock-in ------------------------------------------------------------------

def test_lockin_recovers_amplitude_and_phase():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0.0, 3.0, 700))            # irregular, like USB arrivals
    truth_amp, truth_phase = 2.5, 0.4                  # rad
    y = truth_amp * np.sin(2 * np.pi * 3.0 * t + truth_phase)
    y = y + rng.normal(0.0, 0.2, len(t)) + 5.0         # noise + DC (mounting bias)
    amp, phase_deg = lockin(t, y, 3.0)
    assert amp == pytest.approx(truth_amp, rel=0.05)
    assert phase_deg == pytest.approx(math.degrees(truth_phase), abs=5.0)


def test_lockin_rejects_off_frequency():
    """The noise-floor estimate relies on a lock-in at 1.31f seeing ~nothing."""
    rng = np.random.default_rng(2)
    t = np.sort(rng.uniform(0.0, 3.0, 700))
    y = 2.5 * np.sin(2 * np.pi * 3.0 * t)
    on, _ = lockin(t, y, 3.0)
    off, _ = lockin(t, y, 3.0 * 1.31)
    assert off < on * 0.1


# -- knee frequency -----------------------------------------------------------

def test_knee_matches_first_order_rolloff():
    freqs = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
    fc = 3.0
    mags = [1.0 / math.sqrt(1.0 + (f / fc) ** 2) for f in freqs]
    knee = knee_frequency(freqs, mags)
    # The reference is the 0.5 Hz point (already slightly below DC), so the estimated
    # knee sits a bit above fc; the linear interpolation on a coarse grid adds more.
    assert 2.8 <= knee <= 3.5


def test_knee_none_when_flat_or_silent():
    freqs = [0.5, 1.0, 2.0]
    assert knee_frequency(freqs, [1.0, 0.99, 0.95]) is None   # never drops 3 dB
    assert knee_frequency(freqs, [0.0, 0.0, 0.0]) is None     # no signal
    assert knee_frequency([], []) is None


# -- friction ramp ------------------------------------------------------------
#
# The whole measurement lives in _analyse now: the accelerometer cannot see the slip
# instant (see friction.py's docstring for the arithmetic), so the number is read off the
# plateau the operator leaves when they stop raising the board. The first version took
# max(tilt) on the raw signal, which is a noise statistic -- these pin that it does not
# come back.

def _synth_ramp(hold_deg, rise_s=6.0, hold_s=4.0, fall=True, noise=0.45, seed=0, hz=250):
    """Board raised at a steady rate to `hold_deg`, held, then lowered -- plus the real
    tilt noise measured on this unit (0.45 deg std at 250 Hz)."""
    rng = np.random.default_rng(seed)
    rise = np.linspace(0.0, hold_deg, int(rise_s * hz))
    hold = np.full(int(hold_s * hz), hold_deg)
    parts = [rise, hold]
    if fall:
        parts.append(np.linspace(hold_deg, 0.0, int(3.0 * hz)))
    clean = np.concatenate(parts)
    tilt = clean + rng.normal(0.0, noise, len(clean))
    t = np.arange(len(clean)) / hz
    mag = 10.01 + rng.normal(0.0, 0.03, len(clean))   # this IMU rests at 10.01, not 9.81
    return list(zip(t, tilt, mag))


@pytest.mark.parametrize("truth", [9.0, 13.5, 24.0, 33.0])
def test_friction_recovers_plateau_angle(truth):
    theta, mu, why = _analyse(_synth_ramp(truth))
    assert why.startswith("plateau")
    assert abs(theta - truth) < 1.0, f"read {theta:.1f} for a {truth} deg plateau"
    assert abs(mu - math.tan(math.radians(truth))) < 0.03


def test_friction_plateau_beats_raw_max_under_noise():
    """The bug this replaced: max() of a noisy ramp overshoots the true angle."""
    trace = _synth_ramp(9.0, seed=3)
    raw_max = max(p[1] for p in trace)
    theta, _, _ = _analyse(trace)
    assert raw_max > 10.5                      # noise alone lifts the peak >1.5 deg
    assert abs(theta - 9.0) < abs(raw_max - 9.0) / 2


def test_friction_rejects_topple():
    _, _, why = _analyse(_synth_ramp(70.0))
    assert why.startswith("REJECTED")


def test_friction_reports_lower_bound_without_a_plateau():
    """Operator never stopped: a pure ramp has no held angle, so no slip was observed."""
    trace = _synth_ramp(20.0, rise_s=12.0, hold_s=0.0, fall=False)
    _, _, why = _analyse(trace)
    assert "LOWER BOUND" in why


def test_friction_ignores_a_board_that_barely_moved():
    assert _analyse(_synth_ramp(1.5)) is None
    assert _analyse(_synth_ramp(9.0)[:100]) is None      # too few samples to filter
