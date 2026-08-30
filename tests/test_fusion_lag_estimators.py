"""The two fusion-lag estimators must recover a lag they were GIVEN.

Motivation: the actuated `--phase fusion` run returned "21 ms" from data whose premise did not
hold, and nothing in the code objected. The tilt phase answers that with gates, but a gate is
only worth having if the estimator behind it is correct in the first place -- otherwise a clean
recording still yields a wrong number, quietly. So these tests hand both estimators a synthetic
accel/fused pair with a lag known to the millisecond and require them to find it.

The synthetic signal deliberately mimics the real thing: ~250 Hz sampling, a few degrees of
irregular multi-frequency "hand tilt" below 1 Hz, and independent per-channel noise.
"""

import numpy as np
import pytest

from bittle_io.experiments import latency as L

FS = 250.0
DT = 1.0 / FS


def _tilt_signal(seconds=12.0, lag_ms=20.0, gain=1.0, noise_deg=0.05, seed=0):
    """Accel-like and fused-like channels differing only by a known lag (plus noise).

    Built by generating one continuous waveform and sampling it twice at times offset by the
    lag, so the lag is exact rather than quantised to the sample grid -- which is the case the
    sub-sample refinement exists to handle.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, seconds, DT)
    # Three incommensurate slow components: irregular enough that the correlation peak is
    # sharp, unlike the broad peak a pure sine would give.
    comps = [(0.31, 12.0, 0.0), (0.47, 7.0, 1.1), (0.73, 4.0, 2.4)]

    def wave(tt):
        return sum(A * np.sin(2 * np.pi * f * tt + p) for f, A, p in comps)

    accel = wave(t) + rng.normal(0.0, noise_deg, len(t))
    fused = gain * wave(t - lag_ms / 1000.0) + rng.normal(0.0, noise_deg, len(t))
    return t, accel, fused


@pytest.mark.parametrize("lag_ms", [0.0, 8.0, 20.0, 44.0, 80.0])
def test_both_estimators_recover_known_lag(lag_ms):
    _, a, b = _tilt_signal(lag_ms=lag_ms)
    est = L._window_estimate(DT, a, b)
    # 4 ms is one sample period; both estimators must do better than the raw grid, and on
    # clean data they manage ~0.3 ms.
    assert est["lag_deriv_ms"] == pytest.approx(lag_ms, abs=2.0)
    assert est["lag_phase_ms"] == pytest.approx(lag_ms, abs=2.0)
    assert est["corr"] > 0.99
    assert est["gain"] == pytest.approx(1.0, abs=0.05)


def test_estimators_agree_with_each_other():
    """The run's confidence check is deriv-vs-phase agreement, so the agreement itself has to
    be tight on data where both are right -- otherwise the 10 ms disagreement threshold in
    _tilt would fire on good recordings."""
    for lag in (5.0, 20.0, 60.0):
        _, a, b = _tilt_signal(lag_ms=lag, seed=int(lag))
        est = L._window_estimate(DT, a, b)
        assert abs(est["lag_deriv_ms"] - est["lag_phase_ms"]) < 3.0


def test_negative_lag_is_representable():
    """A gyro-driven filter can PREDICT, leading the accelerometer. An estimator that clamps
    at zero would report 0 ms and hide the fact, so both must express negative lag."""
    _, a, b = _tilt_signal(lag_ms=-24.0)
    est = L._window_estimate(DT, a, b)
    assert est["lag_deriv_ms"] == pytest.approx(-24.0, abs=2.0)
    assert est["lag_phase_ms"] == pytest.approx(-24.0, abs=2.0)


def test_cross_correlation_is_unusable_here_which_is_why_it_was_dropped():
    """Pins the reason `--phase fusion`'s estimator was not reused.

    On a band-limited quasi-static tilt, shifting by one 4 ms sample barely changes the
    overlap, so the correlation ridge is flat and its argmax is noise. This test asserts the
    failure rather than the fix: if someone reintroduces cross-correlation as the estimator,
    this is the evidence they need to read first. The peak VALUE is still trustworthy, which
    is why _window_estimate keeps it as a gate.
    """
    _, a, b = _tilt_signal(lag_ms=20.0, noise_deg=0.0)
    ad, bd = a - a.mean(), b - b.mean()
    max_k = int(L.TILT_MAX_LAG_MS / 1000.0 / DT)
    ks = np.arange(-max_k, max_k + 1)
    cs = np.array([L._shifted_corr(ad, bd, int(k)) for k in ks])
    j = int(np.argmax(cs))
    # The ridge is flat: the peak's immediate neighbours are within a whisker of it.
    assert cs[j] - max(cs[j - 1], cs[j + 1]) < 1e-3
    # ...and the peak value is still a perfectly good agreement measure.
    assert cs[j] > 0.99


def test_gain_gate_catches_a_disagreeing_channel():
    """The failure mode of the actuated phase: the accel channel swung far more than the fused
    one because it was measuring tangential acceleration, not tilt. That shows up as gain far
    below 1, which is what the gate reads."""
    _, a, b = _tilt_signal(lag_ms=20.0, gain=0.2)
    est = L._window_estimate(DT, a, b)
    assert est["gain"] == pytest.approx(0.2, abs=0.05)
    assert not (0.7 <= est["gain"] <= 1.4)


def test_dominant_frequency_is_the_quasistatic_gate():
    slow = np.sin(2 * np.pi * 0.5 * np.arange(0.0, 12.0, DT))
    fast = np.sin(2 * np.pi * 4.0 * np.arange(0.0, 12.0, DT))
    assert L._dominant_freq(slow, DT) == pytest.approx(0.5, abs=0.1)
    assert L._dominant_freq(fast, DT) == pytest.approx(4.0, abs=0.2)


def test_uncorrelated_channels_do_not_produce_a_confident_lag():
    """Noise-only windows must fail the correlation gate rather than fitting a lag to nothing."""
    rng = np.random.default_rng(3)
    n = int(12.0 * FS)
    est = L._window_estimate(DT, rng.normal(0, 1, n), rng.normal(0, 1, n))
    assert est["corr"] < 0.9


def test_uniform_resampling_preserves_the_lag_under_jitter():
    """Host timestamps jitter, but both channels ride the SAME line, so the jitter is
    common-mode. Resampling must not manufacture or destroy lag."""
    t, a, b = _tilt_signal(lag_ms=20.0)
    rng = np.random.default_rng(7)
    jittered = np.sort(t + rng.normal(0.0, 0.0015, len(t)))
    dt, grid, (au, bu) = L._uniform(jittered, a, b)
    assert dt == pytest.approx(DT, rel=0.05)
    assert len(grid) > 0.9 * len(t)
    est = L._window_estimate(dt, au, bu)
    assert est["lag_deriv_ms"] == pytest.approx(20.0, abs=3.0)
    assert est["lag_phase_ms"] == pytest.approx(20.0, abs=3.0)
