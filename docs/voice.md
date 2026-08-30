# The voice module

**You should not need this file.** On the image this repo ships, voice works out of the box:
it arms itself at boot without announcing anything, it answers when you speak to it, and it
stays out of the way while a policy is running. This is the record of what it took to get
there, kept because the behaviour is undocumented upstream and someone hitting it on stock
firmware will otherwise spend a weekend on it.

Four of the patches in [firmware.md](firmware.md) are about the module rather than about
realtime mode. Two are plain bug fixes worth having even if you never run a policy; two are
about keeping a talking robot and a policy loop from fighting over the same servos.

## Voice recognition was dead after every power cycle

The module arms on a genuine language *change*, and it cold-boots already in English — so an
English-default robot needs `voiceSyncAtStartup()` to walk it through one. Upstream commit
`b189c34` ("Streamline code") deleted that branch one day after `85f9eb7` added it, which
left English robots arming nothing on boot. Restored here.

Petoi's own documentation explains why this cannot simply be tuned away. The module accepts
exactly seven commands — `XAa`/`XAb` (set the **default** language), `XAc`/`XAd` (reply tone
and reaction on/off), `XAe`/`XAf` (custom command mode), `XAg` (delete custom commands) — and
**there is no volume or mute command**. It also resets to its default language when the robot
restarts, which is the trap: the commands that arm it are the same ones that rewrite that
default. Ending the sequence in English makes English the default, so the next boot resets to
English and `XAa` is no longer a change.

## An unframed byte was being sprayed into the module's command UART

`printToAllPorts()` in `src/io.h` writes an unframed `k`/`m` completion byte after every
finished motion, gated on `moduleActivatedQ[1]`:

```c
if (moduleActivatedQ[1] && (textResponse == "k" || textResponse == "m")) {
    SERIAL_VOICE.write((uint8_t)textResponse.charAt(0));   // no terminator, by design
```

The module's command protocol is line-based (`XAc\n`), so an unframed `k` prefixes whatever
arrives next: it reads `kXAc` and drops the line. Every completed skill leaves another one,
and the visible effect is that the *first* command of any sequence fails while later ones
land.

It is a board-revision bug, not a bad idea, and it is fixed by **destination**:

| board | `SERIAL_VOICE` | Grove / AI module |
|---|---|---|
| BiBoard V0.1 / V0.2 | `Serial2` | **also `Serial2`** — upstream's write was correct here |
| BiBoard V1.0 | `Serial1` (RX26/TX25) | `Serial2` (RX9/TX10) — they were split, the line was not updated |

Grove is `Serial2` on every revision and `moduleActivatedQ[0]` is Grove_Serial, so writing
there gated on that flag is correct on V0.x and V1.0 alike: the voice module stops being
sprayed with bytes, and Xiaozhi keeps working on both. `-DXIAOZHI_COMPLETION_ECHO=0` disables
the echo outright.

With the channel clean, `VOICE_ARM_SEQUENCE` defaults to **1** (`XAc` alone, no spoken
announcements at boot) — confirmed on hardware, and the reason a robot flashed with this
image is silent at boot where an earlier build announced two language switches.
`build_firmware.sh` exposes the setting; 3 (`XAc` → `XAb` → `XAa`) remains the fallback if a
cold boot ever fails to arm.

## Voice reactions are gated off during a policy run

In realtime mode `read_voice()` drops the reaction but still acknowledges it. A recognised
word otherwise queues a 2.5 s skill that drives the servos against the policy — and any
bystander can trigger it by speaking.

The acknowledgement is not optional. Without it the board says nothing at all while someone
talks during a run, and "no reaction, no log" is indistinguishable from a module that never
heard anything. `control_loop.py` counts the line as chatter.

This gate is also why `control_loop.py` no longer sends `XAd`. That token does not just clear
the firmware's `enableVoiceQ` flag, it forwards to the module, whose state outlives both the
run and a reboot — so re-arming afterwards needs the language sequence, which the module says
*out loud*. The realtime gate is silent, leaves nothing persistent, and lifts the moment `Xr`
arrives. An image with `XR` but without the voice gate would be fought by voice-triggered
skills, so `build_firmware.sh` checks for both.

## Voice cannot reach the calibration token

`read_voice()` refuses `c` (`T_SERVO_CALIBRATE`). It is the front half of the only sequence
that overwrites the per-robot joint offsets in NVS (`c` then `s`), those offsets are
factory-set and not reproducible at home, and `SerialLink.send()` already refuses the token
host-side for that reason. Preset index 59 maps to it, and it fired once per session during
ordinary testing from speech that was not aimed at it — so leaving it reachable by voice
bypassed the host guard entirely.

## Battery cycling with USB attached

The voice module is powered from the battery rail. The main board, while a USB cable is
attached, is not. So switching the battery off and on cold-starts the **module** without
restarting the **board**: `setup()` is never re-entered, `voiceSyncAtStartup()` never runs
again, and the module comes back deaf with nothing in the log to say so. There is no boot to
inspect, because there was no boot:

```
Low power: 2.93V. The robot won't move.
Got 7.74 V power
Reactivating servo PWM signals after power restoration...
```

That handler in `reaction.h` already exists to repair the servos after exactly this event.
The patch re-arms the voice module in the same place. It is skipped while realtime mode is
on: a host loop does not want voice reactions, and the arming sequence blocks for ~2.4 s,
which would cost a hundred control ticks.

Petoi's remedy for an unresponsive module is a full power cycle — disconnect USB, long-press
the battery button — which matches what we found, since a battery cycle alone never restarts
a USB-powered board.
