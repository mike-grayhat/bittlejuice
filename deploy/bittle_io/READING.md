# Reading list — background for the hardware side of this project

Everything `bittle_io` deals with (servos, serial streams, IMUs, system ID, sim-to-real)
maps onto a few well-covered topics. Each entry below is tied to something this package
actually hit.

## If you read only three things

1. **Tan et al., "Sim-to-Real: Learning Agile Locomotion For Quadruped Robots" (RSS
   2018)** — the closest paper to this exact project: a small quadruped with hobby-grade
   actuators. They measure servo response and latency, build the actuator model into the
   simulator, and add domain randomization. Most of `bittle_io`'s experiment suite is
   this paper's methodology applied to a Bittle.
2. **Steven W. Smith, *The Scientist and Engineer's Guide to Digital Signal
   Processing*** — free at dspguide.com, famously readable. Nyquist/aliasing (why 9 Hz
   polling caps a chirp at ~2 Hz), windowing and spectral leakage (the Hann window in
   `bodyid`'s lock-in), sampling. Chapters 3, 9, 11 cover everything used here.
3. **Åström & Murray, *Feedback Systems*** — free PDF from the authors. First-order
   lag, time constants, Bode plots, f_3db — the language `step` and `bodyid` report in.
   Chapters 1–2 plus the frequency-response chapter suffice.

## Servos and actuators

- **Hwangbo et al., "Learning Agile and Dynamic Motor Skills for Legged Robots"
  (Science Robotics 2019)** — introduced the "actuator net": learn the command→torque
  mapping from logged data instead of hand-fitting a time constant. The heavyweight
  version of what `step` approximates.
- Any **RC/hobby servo internals teardown** (Adafruit and Pololu both have short
  guides) — the PWM command signal, the internal position loop, deadband, and why
  Petoi's same-wire feedback trick forces the detach-to-read behavior documented in
  `link.py`.

## Embedded / serial — why streams drop and timing gaps exist

- **Elecia White, *Making Embedded Systems* (O'Reilly)** — the most approachable "why
  firmware behaves like that" book: serial buffers, blocking loops, why a busy MCU drops
  a byte. The 20 ms `COMMAND_GAP` and the stream-stops-on-token behavior stop feeling
  mysterious a few chapters in.

## IMUs and orientation

- **VectorNav's online IMU/AHRS article library** — free and well written: accelerometer
  vs gyro roles, why gravity direction needs fusion, drift, bias. Maps directly to the
  projected-gravity-from-roll/pitch vs from-accel cross-check in `imu --phase tilt`.
- **Madgwick's AHRS orientation-filter report** — the classic readable treatment of
  fusing gyro + accel into attitude; what the fusion running on the robot is doing.
- When `mj_vec_env._inv_rotate` feels opaque: the rotation chapters of **Kuipers,
  *Quaternions and Rotation Sequences***, or any 3D-math primer. One careful read pays
  off permanently.

## The measurement tricks used here

- **Zurich Instruments, "Principles of Lock-in Detection"** (free primer) — what
  `bodyid` does: extract a small known-frequency signal from noise.
- **Tektronix, "XYZs of Oscilloscopes"** (free) — includes equivalent-time sampling,
  the trick behind `step`.

## Legged robots + RL

- **Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep RL" (CoRL
  2021)** — the training recipe `rsl-rl-lib` implements; explains the
  obs/reward/randomization structure `mj_vec_env.py` mirrors.
- **Peng et al., "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"
  (ICRA 2018)** — why "measure, then randomize around the measurement" is the strategy,
  not precise identification.
- **Russ Tedrake, *Underactuated Robotics*** (free MIT course + notes) — the deep end:
  dynamics, contact, why walking is hard at all. Skim now, return later.

## For this kind of debugging specifically

The list above is background on the *topics*. This one is about the skill that actually
decided outcomes here: characterizing undocumented hardware empirically.

1. **David Agans, *Debugging: The 9 Indispensable Rules*** — short and immediately
   applicable. Rule 3, "Quit Thinking and Look," is exactly what the `fp` hypothesis in
   this package violated: reasoning from firmware source said the one-shot API must be
   faster, measurement said 1.7x slower (see `SerialLink.poll_mode`). The book is about
   that failure mode.
2. **A logic analyzer** — not a book, and higher leverage than any of them. A cheap clone
   plus PulseView would settle the `delay(15)` question in twenty minutes by watching the
   pin, where source-reading only produces hypotheses.
3. **Elecia White, *Making Embedded Systems*** — blocking calls, interrupt latency, why a
   busy MCU silently drops a byte. Explains most of what `COMMAND_GAP` works around.
4. **ESP32 Technical Reference Manual**, LEDC and RMT chapters — where the real answer to
   the attach-settling delay lives, and RMT is the peripheral that would replace the
   blocking pulse measurement in `readFeedback()`.
5. **Jack Ganssle** (ganssle.com, embedded.com archives) — the practical writer on
   embedded timing. His debouncing guide is the canonical "sweep it and plot the success
   rate instead of guessing a delay," which is the method that found our 20 ms gap.

If you go further on identification: **Ljung, *System Identification: Theory for the
User*** is the standard reference. Heavier than this project needs.

## Suggested order

Tan et al. first (it *is* this project), then Åström & Murray ch. 1–2 and dspguide ch. 3
as background for interpreting `step`/`bodyid` outputs, then White the next time the
serial link misbehaves. The rest as they become relevant.

Nothing here replaces the workflow that found this package's bugs: reading the firmware
source side by side with measured behavior. That skill comes from doing exactly that.
